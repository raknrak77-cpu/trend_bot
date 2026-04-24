import requests
import pandas as pd
from datetime import datetime
import json

# ============= AYARLAR =============
COINS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
    "cardano": "ADAUSDT"
}
LOOKBACK_HOURS = 2400  # 100 gün = 2400 saat
# ===================================

def fetch_binance_1h_data(symbol="BTCUSDT", limit=1000):
    """Binance'ten 1 saatlik veri çek (her seferde max 1000)"""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "limit": limit
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Sadece ihtiyacımız olan sütunları al
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
    else:
        print(f"❌ {symbol} için hata: {response.status_code}")
        return None

def fetch_all_hours(symbol, target_hours=2400):
    """2400 saati parçalar halinde çek (Binance limit 1000)"""
    all_dfs = []
    remaining = target_hours
    
    while remaining > 0:
        limit = min(1000, remaining)
        df = fetch_binance_1h_data(symbol, limit=limit)
        if df is not None:
            all_dfs.append(df)
            remaining -= limit
            # Rate limit için kısa bekleme
            import time
            time.sleep(0.5)
        else:
            break
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        # Tekrarlanan satırları temizle
        combined = combined.drop_duplicates(subset=['timestamp'])
        combined = combined.sort_values('timestamp')
        return combined.tail(target_hours)
    
    return None

def add_trend_indicators(df):
    """Trend göstergeleri ekle (1 saatlik için uyarlanmış)"""
    # 24 saatlik SMA (1 gün)
    df['sma_24'] = df['close'].rolling(24).mean()
    # 168 saatlik SMA (1 hafta)
    df['sma_168'] = df['close'].rolling(168).mean()
    
    # Trend durumu
    df['trend'] = df.apply(
        lambda row: '📈 YÜKSELİŞ' if row['close'] > row['sma_24'] else '📉 DÜŞÜŞ',
        axis=1
    )
    
    return df

def save_outputs(df, coin_name, symbol):
    """Excel ve JSON olarak kaydet"""
    df = add_trend_indicators(df)
    
    excel_name = f"output_{coin_name}_1h.xlsx"
    df.to_excel(excel_name, index=False)
    
    son_satir = df.iloc[-1]
    summary = {
        "coin": coin_name,
        "symbol": symbol,
        "timeframe": "1h",
        "last_price": float(son_satir['close']),
        "trend": son_satir['trend'],
        "sma_24": float(son_satir['sma_24']) if pd.notna(son_satir['sma_24']) else None,
        "sma_168": float(son_satir['sma_168']) if pd.notna(son_satir['sma_168']) else None,
        "volume_24h": float(df.tail(24)['volume'].sum()),
        "data_points": len(df),
        "timestamp": datetime.now().isoformat()
    }
    
    json_name = f"output_{coin_name}_1h.json"
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return excel_name, json_name

if __name__ == "__main__":
    print("🚀 Trend Bot Başlıyor (1 Saatlik Veri ile)")
    print(f"📋 İncelenecek coinler: {list(COINS.keys())}")
    print(f"⏰ Geriye dönük saat: {LOOKBACK_HOURS} (={LOOKBACK_HOURS//24} gün)")
    print("-" * 50)
    
    all_results = []
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol}) işleniyor...")
        df = fetch_all_hours(symbol, target_hours=LOOKBACK_HOURS)
        
        if df is not None:
            excel_file, json_file = save_outputs(df, coin_name, symbol)
            
            son_satir = df.iloc[-1]
            result = {
                "coin": coin_name,
                "symbol": symbol,
                "last_price": float(son_satir['close']),
                "trend": son_satir['trend'],
                "data_points": len(df)
            }
            all_results.append(result)
            
            print(f"   ✅ Excel: {excel_file}")
            print(f"   ✅ JSON: {json_file}")
            print(f"   📈 Son fiyat: ${son_satir['close']:,.2f}")
            print(f"   📊 Trend: {son_satir['trend']}")
            print(f"   ⏰ Veri aralığı: {df.iloc[0]['timestamp']} → {df.iloc[-1]['timestamp']}")
        else:
            print(f"   ❌ {coin_name} atlandı (veri alınamadı)")
    
    # Ana rapor
    if all_results:
        master_df = pd.DataFrame(all_results)
        master_df.to_excel("master_report_1h.xlsx", index=False)
        print("\n" + "=" * 50)
        print("🎉 TAMAMLANDI!")
        print(f"\n📁 Oluşan dosyalar:")
        print(f"   📊 Ana rapor: master_report_1h.xlsx")
        for r in all_results:
            print(f"   📈 {r['coin']}: ${r['last_price']:,.2f} - {r['trend']}")
