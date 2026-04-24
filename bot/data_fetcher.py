import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os

print("=" * 60)
print("🚀 TREND BOT - Yahoo Finance")
print("=" * 60)

OUTPUT_DIR = "veri"

COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD"
}
LOOKBACK_HOURS = 2000

def fetch_yahoo_data(symbol="BTC-USD", hours=2000):
    print(f"   📡 {symbol} çekiliyor...")
    
    try:
        ticker = yf.Ticker(symbol)
        days = hours / 24 + 1
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df = ticker.history(start=start, end=end, interval="1h")
        
        if df.empty:
            print(f"   ⚠️ {symbol} için veri yok")
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # ZAMAN DİLİMİNİ TEMİZLE (Excel hatasını çözer)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"   ✅ {len(df)} saatlik veri")
        return df
    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:80]}")
        return None

def add_indicators(df):
    """Teknik göstergeler ekle"""
    if len(df) < 24:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()
    df['sma_72'] = df['close'].rolling(72).mean()
    df['sma_168'] = df['close'].rolling(168).mean()
    
    # RSI (14 saat)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_24']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Saatlik getiri
    df['hourly_return'] = df['close'].pct_change(1) * 100
    
    # Trend etiketi
    def get_trend(row):
        if pd.isna(row['sma_24']):
            return "YETERSİZ"
        if row['close'] > row['sma_24'] and row['close'] > row['sma_168']:
            return "📈 GÜÇLÜ YÜKSELİŞ"
        elif row['close'] > row['sma_24']:
            return "📈 ZAYIF YÜKSELİŞ"
        elif row['close'] < row['sma_24'] and row['close'] < row['sma_168']:
            return "📉 GÜÇLÜ DÜŞÜŞ"
        elif row['close'] < row['sma_24']:
            return "📉 ZAYIF DÜŞÜŞ"
        else:
            return "➡️ YATAY"
    
    df['trend'] = df.apply(get_trend, axis=1)
    
    # RSI durumu
    def get_rsi_status(rsi):
        if pd.isna(rsi):
            return "VERİ YOK"
        elif rsi > 70:
            return "🔴 AŞIRI ALIM"
        elif rsi < 30:
            return "🟢 AŞIRI SATIM"
        else:
            return "⚪ NÖTR"
    
    df['rsi_status'] = df['rsi_14'].apply(get_rsi_status)
    
    return df

def save_outputs(df, coin_name, symbol):
    """Excel ve JSON kaydet"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    df = add_indicators(df)
    
    # Excel (zaman dilimi temizlendi, artık çalışır)
    excel_name = f"{OUTPUT_DIR}/{coin_name}_1h.xlsx"
    df.to_excel(excel_name, index=False)
    
    # JSON özet
    son = df.iloc[-1]
    summary = {
        "coin": coin_name,
        "symbol": symbol,
        "last_price": float(son['close']),
        "trend": son['trend'],
        "rsi_14": float(son['rsi_14']) if pd.notna(son['rsi_14']) else None,
        "rsi_status": son['rsi_status'],
        "macd": float(son['macd']) if pd.notna(son['macd']) else None,
        "macd_signal": float(son['macd_signal']) if pd.notna(son['macd_signal']) else None,
        "hourly_return": float(son['hourly_return']) if pd.notna(son['hourly_return']) else None,
        "sma_24": float(son['sma_24']) if pd.notna(son['sma_24']) else None,
        "sma_168": float(son['sma_168']) if pd.notna(son['sma_168']) else None,
        "data_points": len(df),
        "timestamp": datetime.now().isoformat()
    }
    
    json_name = f"{OUTPUT_DIR}/{coin_name}_1h.json"
    with open(json_name, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return excel_name, json_name

def main():
    # Klasörü oluştur
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 {OUTPUT_DIR}/ klasörü oluşturuldu")
    
    print(f"📋 Coin: {len(COINS)} adet")
    print(f"⏰ Hedef: {LOOKBACK_HOURS} saat ({LOOKBACK_HOURS//24} gün)")
    print("-" * 60)
    
    results = []
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()}")
        df = fetch_yahoo_data(symbol, LOOKBACK_HOURS)
        
        if df is not None and len(df) > 0:
            excel_file, json_file = save_outputs(df, coin_name, symbol)
            results.append({
                "coin": coin_name,
                "last_price": float(df.iloc[-1]['close']),
                "trend": df.iloc[-1]['trend'],
                "data_points": len(df)
            })
            print(f"   ✅ {excel_file}")
            print(f"   📈 ${df.iloc[-1]['close']:,.2f} | {df.iloc[-1]['trend']}")
        else:
            print(f"   ❌ {coin_name} için veri alınamadı")
    
    # Master rapor
    if results:
        master_df = pd.DataFrame(results)
        master_df.to_excel(f"{OUTPUT_DIR}/master_rapor.xlsx", index=False)
        
        with open(f"{OUTPUT_DIR}/master_rapor.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Master rapor: {OUTPUT_DIR}/master_rapor.xlsx")
    
    print("\n" + "=" * 60)
    print(f"✅ {len(results)}/{len(COINS)} coin başarılı")
    print(f"📁 Tüm çıktılar: {OUTPUT_DIR}/")
    
    # Dosyaları listele
    print("\n📄 Oluşan dosyalar:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"   - {f}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
