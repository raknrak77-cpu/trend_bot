import requests
import pandas as pd
from datetime import datetime
import json
import time
import os
import sys

# ============= AYARLAR =============
OUTPUT_DIR = "analiz_ciktisi"
COINS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
}
LOOKBACK_HOURS = 100

# Binance Hong Kong IP'si (CloudFront üzerinden)
BINANCE_API_BASE = "https://api.binance.com"
# ===================================

def log(msg):
    """Hem ekrana yaz hem dosyaya"""
    print(msg)
    with open("debug.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")

def fetch_binance_1h_data(symbol="BTCUSDT", limit=100):
    log(f"   ➤ {symbol} için veri çekiliyor...")
    
    url = f"{BINANCE_API_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "limit": limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        log(f"   ➤ HTTP {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            log(f"   ➤ {len(data)} mum verisi alındı")
            
            rows = []
            for k in data:
                rows.append({
                    'timestamp': pd.to_datetime(k[0], unit='ms'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp')
            log(f"   ✅ {symbol} başarılı, {len(df)} satır")
            return df
        else:
            log(f"   ❌ HTTP {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        log(f"   ❌ İstisna: {str(e)}")
        return None

def save_outputs(df, coin_name, symbol, output_dir):
    """Excel ve JSON kaydet"""
    log(f"   ➤ {coin_name} çıktıları kaydediliyor...")
    
    try:
        # Klasörü kontrol et
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log(f"   ➤ Klasör oluşturuldu: {output_dir}")
        
        # Excel
        excel_name = f"{output_dir}/output_{coin_name}.xlsx"
        df.to_excel(excel_name, index=False)
        log(f"   ✅ Excel kaydedildi: {excel_name}")
        
        # JSON
        summary = {
            "coin": coin_name,
            "symbol": symbol,
            "last_price": float(df.iloc[-1]['close']),
            "data_points": len(df),
            "timestamp": datetime.now().isoformat()
        }
        json_name = f"{output_dir}/output_{coin_name}.json"
        with open(json_name, 'w') as f:
            json.dump(summary, f, indent=2)
        log(f"   ✅ JSON kaydedildi: {json_name}")
        
        return True
    except Exception as e:
        log(f"   ❌ Kayıt hatası: {str(e)}")
        return False

def main():
    # Log dosyasını temizle
    if os.path.exists("debug.log"):
        os.remove("debug.log")
    
    log("=" * 60)
    log("🚀 TREND BOT DEBUG - BAŞLANGIÇ")
    log(f"📁 Çalışma dizini: {os.getcwd()}")
    log(f"📁 Klasör içeriği: {os.listdir('.')}")
    log("=" * 60)
    
    # Çıktı klasörünü oluştur
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        log(f"✅ {OUTPUT_DIR} klasörü oluşturuldu")
    
    successful = 0
    
    for coin_name, symbol in COINS.items():
        log(f"\n🪙 {coin_name.upper()} ({symbol})")
        df = fetch_binance_1h_data(symbol, limit=LOOKBACK_HOURS)
        
        if df is not None and len(df) > 0:
            if save_outputs(df, coin_name, symbol, OUTPUT_DIR):
                successful += 1
                log(f"   📈 Son fiyat: ${df.iloc[-1]['close']:,.2f}")
        else:
            log(f"   ❌ {coin_name} için veri yok")
    
    log("\n" + "=" * 60)
    log(f"✅ Başarılı coin sayısı: {successful}/{len(COINS)}")
    log(f"📁 Çıktılar '{OUTPUT_DIR}/' klasöründe")
    log(f"📄 Debug log: debug.log")
    
    # Klasör içeriğini listele
    log(f"\n📁 {OUTPUT_DIR} klasör içeriği:")
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            log(f"   - {f}")
    else:
        log(f"   ❌ {OUTPUT_DIR} klasörü bulunamadı!")
    
    log("=" * 60)

if __name__ == "__main__":
    main()
