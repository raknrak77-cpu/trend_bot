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

LOOKBACK_HOURS = 1000  # Binance limiti 1000, test için yeterli

# 🎯 KRİTİK DEĞİŞİKLİK: API base URL'i Hong Kong sunucusu
# Bu IP doğrudan Hong Kong'daki Binance uç noktasına ait
BINANCE_API_BASE = "https://13.32.36.6"  # Alternatif: "https://api.binance.com" (orijinal)
# ===================================

def fetch_binance_hk_1h_data(symbol="BTCUSDT", limit=1000):
    """
    Binance Hong Kong sunucusundan 1 saatlik veri çeker.
    Coğrafi engelleri aşmak için alternatif IP kullanır.
    """
    url = f"{BINANCE_API_BASE}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "limit": limit
    }
    
    # SSL doğrulamasını kapat (özel IP için gerekli olabilir)
    response = requests.get(url, params=params, verify=False)
    
    if response.status_code == 200:
        data = response.json()
        
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
        print(f"   📊 {len(df)} saatlik veri alındı")
        return df
    else:
        print(f"   ❌ Hata {response.status_code}: {response.text[:100]}")
        return None

# Diğer fonksiyonlar (add_technical_indicators, save_outputs, create_master_report) 
# önceki kodla AYNI, sadece `fetch_bybit_1h_data` yerine yukarıdaki fonksiyon kullanılacak
