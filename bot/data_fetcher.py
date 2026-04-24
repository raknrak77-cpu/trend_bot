import requests
import pandas as pd
from datetime import datetime
import json
import time
import os

# ============= AYARLAR =============
OUTPUT_DIR = "analiz_ciktisi"
COINS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "cardano": "ADAUSDT"
}
LOOKBACK_HOURS = 2000  # Bybit max 2000 limit
# ===================================

def create_output_dir():
    """Çıktı klasörünü oluştur"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    return OUTPUT_DIR

def fetch_bybit_1h_data(symbol="BTCUSDT", limit=2000):
    """
    Bybit'ten 1 saatlik OHLCV verisi çeker.
    Türkiye'den erişilebilir, ücretsiz, anahtar gerekmez.
    """
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": "60",  # 60 dakika = 1 saat
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if data.get('retCode') == 0:
            klines = data.get('result', {}).get('list', [])
            
            if not klines:
                print(f"   ⚠️ {symbol} için veri yok")
                return None
            
            rows = []
            for k in klines:
                # Bybit format: [timestamp, open, high, low, close, volume, ...]
                rows.append({
                    'timestamp': pd.to_datetime(int(k[0]), unit='ms'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp')
            df = df.tail(limit)
            df['symbol'] = symbol
            
            print(f"   📊 {len(df)} saatlik veri alındı ({df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']})")
            return df
        else:
            print(f"   ❌ Bybit hatası: {data.get('retMsg')}")
            return None
    else:
        print(f"   ❌ HTTP {response.status_code}")
        return None

def add_technical_indicators(df):
    """Teknik göstergeler ekle"""
    if len(df) < 24:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()
    df['sma_72'] = df['close'].rolling(72).mean()
    df['sma_168'] = df['close'].rolling(168).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_24']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Saatlik getiri
    df['hourly_return'] = df['close'].pct_change(1) * 100
    
    # Trend durumu
    def get_trend(row):
        if pd.isna(row['sma_24']):
            return "⚠️ YETERSİZ"
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

def save_outputs(df, coin_name, symbol, output_dir):
    """Excel ve JSON olarak kaydet"""
    df = add_technical_indicators(df)
    
    # Excel
    excel_name = f"{output_dir}/output_{coin_name}_1h.xlsx"
    with pd.ExcelWriter(excel_name, engine='openpyxl') as writer:
        # Ana veri
        df.to_excel(writer, sheet_name='Tüm_Veri', index=False)
        # Son 24 saat
        df.tail(24).to_excel(writer, sheet_name='Son_24_Saat', index=False)
        # Son 168 saat (1 hafta)
        df.tail(168).to_excel(writer, sheet_name='Son_1_Hafta', index=False)
    
    # JSON - son durum
    son_satir = df.iloc[-1]
    summary = {
        "coin": coin_name,
        "symbol": symbol,
        "timeframe": "1h",
        "data_points": len(df),
        "hours_requested": LOOKBACK_HOURS,
        "last_price": float(son_satir['close']),
        "trend": son_satir['trend'],
        "rsi_14": float(son_satir['rsi_14']) if pd.notna(son_satir['rsi_14']) else None,
        "rsi_status": son_satir['rsi_status'],
        "macd": float(son_satir['macd']) if pd.notna(son_satir['macd']) else None,
        "macd_signal": float(son_satir['macd_signal']) if pd.notna(son_satir['macd_signal']) else None,
        "atr_14": float(son_satir['atr_14']) if pd.notna(son_satir['atr_14']) else None,
        "sma_24": float(son_satir['sma_24']) if pd.notna(son_satir['sma_24']) else None,
        "sma_168": float(son_satir['sma_168']) if pd.notna(son_satir['sma_168']) else None,
        "hourly_return_pct": float(son_satir['hourly_return']) if pd.notna(son_satir['hourly_return']) else None,
        "volume_24h": float(df.tail(24)['volume'].sum()),
        "timestamp": datetime.now().isoformat()
    }
    
    json_name = f"{output_dir}/output_{coin_name}_1h.json"
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return excel_name, json_name, summary

def create_master_report(all_summaries, output_dir):
    """Tüm coinlerin özet raporu"""
    if not all_summaries:
        return None, None
    
    master_df = pd.DataFrame(all_summaries)
    master_excel = f"{output_dir}/master_report_1h.xlsx"
    master_df.to_excel(master_excel, index=False)
    
    master_json = f"{output_dir}/master_report_1h.json"
    with open(master_json, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    return master_excel, master_json

def main():
    output_dir = create_output_dir()
    
    print("=" * 60)
    print("🚀 TREND BOT - 1 SAATLİK VERİ ANALİZİ")
    print("📍 Veri Kaynağı: Bybit (Türkiye'den Erişilebilir)")
    print("=" * 60)
    print(f"📋 İncelenecek coinler: {list(COINS.keys())}")
    print(f"⏰ Hedef: {LOOKBACK_HOURS} saat ({LOOKBACK_HOURS//24} gün)")
    print(f"📁 Çıktı klasörü: {output_dir}/")
    print(f"🕐 Başlangıç: {datetime.now().isoformat()}")
    print("-" * 60)
    
    all_summaries = []
    successful = 0
    failed = 0
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")
        df = fetch_bybit_1h_data(symbol, limit=LOOKBACK_HOURS)
        
        if df is not None and len(df) >= 24:
            excel_file, json_file, summary = save_outputs(df, coin_name, symbol, output_dir)
            all_summaries.append(summary)
            successful += 1
            
            print(f"   ✅ Excel: {excel_file}")
            print(f"   ✅ JSON: {json_file}")
            print(f"   📈 Son fiyat: ${summary['last_price']:,.2f}")
            print(f"   📊 Trend: {summary['trend']}")
            print(f"   📊 RSI(14): {summary['rsi_14']:.1f} {summary['rsi_status']}")
        else:
            failed += 1
            print(f"   ❌ {coin_name} atlandı")
        
        time.sleep(0.5)
    
    # Ana rapor
    if all_summaries:
        master_excel, master_json = create_master_report(all_summaries, output_dir)
        print("\n" + "=" * 60)
        print("🎉 ANALİZ TAMAMLANDI!")
        print(f"✅ Başarılı: {successful} coin")
        print(f"❌ Başarısız: {failed} coin")
        print(f"\n📁 Tüm çıktılar '{output_dir}/' klasöründe:")
        print(f"   📊 Her coin için: output_[coin]_1h.xlsx")
        print(f"   📄 Her coin için: output_[coin]_1h.json")
        print(f"   📊 Master rapor: {master_excel}")
        print(f"   📄 Master JSON: {master_json}")
    else:
        print("\n❌ Hiçbir coin için veri alınamadı!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
