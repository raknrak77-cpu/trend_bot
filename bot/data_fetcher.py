import requests
import pandas as pd
from datetime import datetime
import json
import time

# ============= AYARLAR =============
COINS = {
    "bitcoin": "btc-bitcoin",
    "ethereum": "eth-ethereum",
    "binancecoin": "bnb-binance-coin",
    "solana": "sol-solana",
    "cardano": "ada-cardano"
}
LOOKBACK_HOURS = 2400  # 100 gün = 2400 saat
# ===================================

def fetch_coinpaprika_ohlcv(coin_id="btc-bitcoin", hours=2400):
    """
    CoinPaprika'dan 1 saatlik OHLCV verisi çeker.
    Direkt historical OHLCV endpoint'ini kullanır.
    """
    # CoinPaprika historical OHLCV endpoint'i
    # Doğrudan çalışır, market aramaya gerek yok
    url = f"https://api.coinpaprika.com/v1/coins/{coin_id}/ohlcv/historical"
    
    params = {
        "limit": hours
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if not data:
            print(f"   ⚠️ {coin_id} için veri yok")
            return None
        
        # Veriyi DataFrame'e dönüştür
        rows = []
        for item in data:
            # Timestamp format: "2024-04-20T00:00:00Z"
            timestamp_str = item.get('timestamp', '')
            if timestamp_str:
                # Saatlik veri için timestamp'i düzgün parse et
                timestamp = pd.to_datetime(timestamp_str)
            else:
                continue
                
            rows.append({
                'timestamp': timestamp,
                'open': float(item.get('open', 0)),
                'high': float(item.get('high', 0)),
                'low': float(item.get('low', 0)),
                'close': float(item.get('close', 0)),
                'volume': float(item.get('volume', 0))
            })
        
        if not rows:
            print(f"   ⚠️ {coin_id} için satır oluşturulamadı")
            return None
            
        df = pd.DataFrame(rows)
        df = df.sort_values('timestamp')
        
        # Hedeflenen saat sayısını al
        df = df.tail(hours)
        df['coin_id'] = coin_id
        
        print(f"   📊 {len(df)} saatlik veri alındı")
        return df
    else:
        print(f"   ❌ Hata {response.status_code}: {response.text[:100]}")
        return None

def add_technical_indicators(df):
    """Teknik göstergeler ekle"""
    if len(df) < 24:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()      # 1 gün
    df['sma_72'] = df['close'].rolling(72).mean()      # 3 gün
    df['sma_168'] = df['close'].rolling(168).mean()    # 7 gün
    
    # RSI (14 saatlik)
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
    
    # ATR (Average True Range)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Saatlik getiri
    df['hourly_return'] = df['close'].pct_change(1) * 100
    
    # Hacim değişimi
    df['volume_change'] = df['volume'].pct_change(1) * 100
    
    # Trend durumu
    def get_trend(row):
        if pd.isna(row['sma_24']) or pd.isna(row['sma_168']):
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
            return "AŞIRI ALIM"
        elif rsi < 30:
            return "AŞIRI SATIM"
        else:
            return "NÖTR"
    
    df['rsi_status'] = df['rsi_14'].apply(get_rsi_status)
    
    return df

def save_outputs(df, coin_name, coin_id, hours_requested):
    """Excel ve JSON olarak kaydet"""
    df = add_technical_indicators(df)
    
    # Excel
    excel_name = f"output_{coin_name}_1h.xlsx"
    df.to_excel(excel_name, index=False)
    
    # Excel'de son 24 saat ayrı sheet'te
    last_24h = df.tail(24)
    with pd.ExcelWriter(excel_name, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        last_24h.to_excel(writer, sheet_name='Son_24_Saat', index=False)
    
    # JSON - son durum
    son_satir = df.iloc[-1]
    summary = {
        "coin": coin_name,
        "coin_id": coin_id,
        "timeframe": "1h",
        "data_points": len(df),
        "hours_requested": hours_requested,
        "last_price": float(son_satir['close']) if pd.notna(son_satir['close']) else None,
        "trend": son_satir['trend'] if pd.notna(son_satir['trend']) else "VERİ YOK",
        "rsi_14": float(son_satir['rsi_14']) if pd.notna(son_satir['rsi_14']) else None,
        "rsi_status": son_satir['rsi_status'] if pd.notna(son_satir['rsi_status']) else "VERİ YOK",
        "macd": float(son_satir['macd']) if pd.notna(son_satir['macd']) else None,
        "macd_signal": float(son_satir['macd_signal']) if pd.notna(son_satir['macd_signal']) else None,
        "atr_14": float(son_satir['atr_14']) if pd.notna(son_satir['atr_14']) else None,
        "sma_24": float(son_satir['sma_24']) if pd.notna(son_satir['sma_24']) else None,
        "sma_168": float(son_satir['sma_168']) if pd.notna(son_satir['sma_168']) else None,
        "hourly_return_pct": float(son_satir['hourly_return']) if pd.notna(son_satir['hourly_return']) else None,
        "volume_24h": float(df.tail(24)['volume'].sum()),
        "volume_avg_24h": float(df.tail(24)['volume'].mean()),
        "timestamp": datetime.now().isoformat()
    }
    
    json_name = f"output_{coin_name}_1h.json"
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Son 24 saatlik detay JSON
    last_24h_json = last_24h[['timestamp', 'close', 'high', 'low', 'volume', 'hourly_return', 'rsi_14', 'trend']].copy()
    last_24h_json['timestamp'] = last_24h_json['timestamp'].dt.strftime('%Y-%m-%d %H:00')
    last_24h_file = f"output_{coin_name}_last24h.json"
    with open(last_24h_file, 'w', encoding='utf-8') as f:
        json.dump(last_24h_json.to_dict(orient='records'), f, indent=2, ensure_ascii=False)
    
    return excel_name, json_name, last_24h_file, summary

def create_master_report(all_summaries):
    """Tüm coinlerin özet raporu"""
    if not all_summaries:
        return None, None
    
    master_df = pd.DataFrame(all_summaries)
    master_excel = "master_report_1h.xlsx"
    master_df.to_excel(master_excel, index=False)
    
    master_json = "master_report_1h.json"
    with open(master_json, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    
    return master_excel, master_json

def main():
    print("=" * 60)
    print("🚀 TREND BOT - 1 SAATLİK VERİ ANALİZİ")
    print("📍 Veri Kaynağı: CoinPaprika")
    print("=" * 60)
    print(f"📋 İncelenecek coinler: {list(COINS.keys())}")
    print(f"⏰ Hedef: {LOOKBACK_HOURS} saat ({LOOKBACK_HOURS//24} gün)")
    print(f"🕐 Başlangıç: {datetime.now().isoformat()}")
    print("-" * 60)
    
    all_summaries = []
    successful = 0
    failed = 0
    
    for coin_name, coin_id in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({coin_id})")
        df = fetch_coinpaprika_ohlcv(coin_id, hours=LOOKBACK_HOURS)
        
        if df is not None and len(df) >= 24:
            excel_file, json_file, last24h_file, summary = save_outputs(df, coin_name, coin_id, LOOKBACK_HOURS)
            all_summaries.append(summary)
            successful += 1
            
            print(f"   ✅ Excel: {excel_file}")
            print(f"   ✅ JSON: {json_file}")
            print(f"   ✅ Son 24h: {last24h_file}")
            print(f"   📈 Son fiyat: ${summary['last_price']:,.2f}")
            print(f"   📊 Trend: {summary['trend']}")
            if summary.get('rsi_14'):
                print(f"   📊 RSI(14): {summary['rsi_14']:.1f} ({summary['rsi_status']})")
            print(f"   📊 24s Hacim: ${summary['volume_24h']:,.0f}")
        else:
            failed += 1
            print(f"   ❌ {coin_name} atlandı (yetersiz veri)")
        
        # Rate limit için bekle (ücretsiz API için iyi pratiktir)
        time.sleep(1)
    
    # Ana rapor
    if all_summaries:
        master_excel, master_json = create_master_report(all_summaries)
        print("\n" + "=" * 60)
        print("🎉 ANALİZ TAMAMLANDI!")
        print(f"✅ Başarılı: {successful} coin")
        print(f"❌ Başarısız: {failed} coin")
        print(f"\n📁 Çıktı dosyaları:")
        print(f"   📊 Her coin için: output_[coin]_1h.xlsx")
        print(f"   📄 Her coin için: output_[coin]_1h.json")
        print(f"   📋 Her coin için: output_[coin]_last24h.json")
        print(f"   📊 Master rapor: {master_excel}")
        print(f"   📄 Master JSON: {master_json}")
    else:
        print("\n❌ Hiçbir coin için veri alınamadı!")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
