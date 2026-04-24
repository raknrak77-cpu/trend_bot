import requests
import pandas as pd
from datetime import datetime
import json

def fetch_coin_data(coin_id="bitcoin", days=100):
    """CoinGecko'dan coin verisi çek"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": "usd", "days": days}
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['coin'] = coin_id
        return df
    else:
        print(f"❌ {coin_id} için hata: {response.status_code}")
        return None

def save_outputs(df, coin_name):
    """Excel ve JSON olarak kaydet"""
    # Excel
    excel_name = f"output_{coin_name}.xlsx"
    df.to_excel(excel_name, index=False)
    
    # JSON (son 30 günlük özet)
    summary = {
        "coin": coin_name,
        "last_price": float(df.iloc[-1]['close']),
        "highest_30d": float(df.tail(30)['high'].max()),
        "lowest_30d": float(df.tail(30)['low'].min()),
        "data_points": len(df)
    }
    
    json_name = f"output_{coin_name}.json"
    with open(json_name, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return excel_name, json_name

if __name__ == "__main__":
    print("🚀 Trend Bot Başlıyor...")
    test_coin = "bitcoin"
    
    df = fetch_coin_data(test_coin, days=100)
    if df is not None:
        excel_file, json_file = save_outputs(df, test_coin)
        print(f"✅ {test_coin} verisi çekildi")
        print(f"📊 Excel: {excel_file}")
        print(f"📄 JSON: {json_file}")
        print(f"\n📈 Son fiyat: {df.iloc[-1]['close']} USD")
        print(f"📅 Veri aralığı: {df.iloc[0]['timestamp'].date()} → {df.iloc[-1]['timestamp'].date()}")
