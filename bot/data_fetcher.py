import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

OUTPUT_DIR = "analiz_ciktisi"
COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD"
}
LOOKBACK_HOURS = 2000  # 83 gün

def fetch_yahoo_1h_data(symbol="BTC-USD", hours=2000):
    """Yahoo Finance'ten 1 saatlik veri çek"""
    try:
        # Yahoo Finance 1 saatlik interval destekliyor
        ticker = yf.Ticker(symbol)
        
        # Son 'hours' kadar saatlik veri (gün cinsinden hesapla)
        days = hours / 24
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = ticker.history(start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            print(f"   ⚠️ {symbol} için veri yok")
            return None
        
        # Sütunları düzenle
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"   📊 {len(df)} saatlik veri alındı")
        return df
    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:100]}")
        return None

def save_outputs(df, coin_name, symbol):
    """Excel ve JSON kaydet"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Excel
    excel_name = f"{OUTPUT_DIR}/output_{coin_name}_1h.xlsx"
    df.to_excel(excel_name, index=False)
    
    # JSON özet
    summary = {
        "coin": coin_name,
        "symbol": symbol,
        "last_price": float(df.iloc[-1]['close']),
        "highest_24h": float(df.tail(24)['high'].max()),
        "lowest_24h": float(df.tail(24)['low'].min()),
        "volume_24h": float(df.tail(24)['volume'].sum()),
        "data_points": len(df),
        "timestamp": datetime.now().isoformat()
    }
    
    json_name = f"{OUTPUT_DIR}/output_{coin_name}_1h.json"
    with open(json_name, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return excel_name, json_name

def main():
    print("=" * 60)
    print("🚀 TREND BOT - Yahoo Finance ile")
    print("=" * 60)
    print(f"📋 Coinler: {list(COINS.keys())}")
    print(f"⏰ Hedef: {LOOKBACK_HOURS} saat")
    
    successful = 0
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")
        df = fetch_yahoo_1h_data(symbol, hours=LOOKBACK_HOURS)
        
        if df is not None and len(df) > 10:
            excel_file, json_file = save_outputs(df, coin_name, symbol)
            successful += 1
            print(f"   ✅ {excel_file}")
            print(f"   📈 Son fiyat: ${df.iloc[-1]['close']:.2f}")
    
    print("\n" + "=" * 60)
    print(f"✅ Tamamlandı! {successful}/{len(COINS)} coin başarılı")
    print(f"📁 Çıktılar: {OUTPUT_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
