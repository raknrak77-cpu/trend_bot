import pandas as pd
import json
import os
from datetime import datetime

from data_fetcher import fetch_yahoo_data, add_features
from predictor import train_and_predict

print("=" * 70)
print("🚀 TREND BOT - LSTM (15 Coin | 14 Saat)")
print("=" * 70)

OUTPUT_DIR = "veri"
COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD",
    "ripple": "XRP-USD",
    "dogecoin": "DOGE-USD",
    "polkadot": "DOT-USD",
    "avalanche": "AVAX-USD",
    "shiba_inu": "SHIB-USD",
    "toncoin": "TON-USD",
    "chainlink": "LINK-USD",
    "uniswap": "UNI-USD",
    "litecoin": "LTC-USD",
    "aptos": "APT-USD"
}
LOOKBACK_HOURS = 3000
TARGET_HOURS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 48, 72]

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"📋 Coin: {len(COINS)}")
    print(f"🎯 Hedef saat: {len(TARGET_HOURS)}")
    print("=" * 70)
    
    all_predictions = []
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")
        
        df = fetch_yahoo_data(symbol, LOOKBACK_HOURS)
        if df is None or len(df) < 500:
            print(f"   ❌ Yetersiz veri")
            continue
        
        df = add_features(df)
        df = df.dropna()
        
        predictions = {}
        for hour in TARGET_HOURS:
            print(f"      🔮 {hour}h...", end=" ")
            pred, acc = train_and_predict(df, hour)
            if pred is not None:
                predictions[f"{hour}h"] = {
                    "expected_return_pct": round(pred * 100, 2),
                    "expected_price": round(float(df['close'].iloc[-1]) * (1 + pred), 2),
                    "direction": "📈 YUKARI" if pred > 0 else "📉 ASAGI",
                    "model_accuracy": round(acc, 1)
                }
                print(f"✅ %{round(pred*100,2)}")
            else:
                print(f"❌")
        
        if predictions:
            all_predictions.append({
                "coin": coin_name,
                "last_price": float(df['close'].iloc[-1]),
                "timestamp": datetime.now().isoformat(),
                "predictions": predictions
            })
    
    # Kaydet
    if all_predictions:
        with open(f"{OUTPUT_DIR}/tahminler.json", "w") as f:
            json.dump(all_predictions, f, indent=2)
        
        rows = []
        for p in all_predictions:
            for h, pred in p['predictions'].items():
                rows.append({
                    'coin': p['coin'],
                    'son_fiyat': p['last_price'],
                    'tahmin_saati': h,
                    'beklenen_getiri_%': pred['expected_return_pct'],
                    'beklenen_fiyat': pred['expected_price'],
                    'yon': pred['direction'],
                    'model_doğruluk_%': pred['model_accuracy']
                })
        pd.DataFrame(rows).to_excel(f"{OUTPUT_DIR}/tahminler.xlsx", index=False)
        
        print("\n" + "=" * 70)
        print(f"✅ {len(all_predictions)}/{len(COINS)} coin başarılı")
        print(f"📁 Çıktılar: {OUTPUT_DIR}/")
        print("=" * 70)

if __name__ == "__main__":
    main()
