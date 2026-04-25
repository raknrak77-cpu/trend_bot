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
    "ripple": "XRP-USD",
    "solana": "SOL-USD",
    "dogecoin": "DOGE-USD",
    "cardano": "ADA-USD",
    "toncoin": "TON-USD",
    "avalanche": "AVAX-USD",
    "chainlink": "LINK-USD",
    "polkadot": "DOT-USD"
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
            print(f"      🔮 {hour}h...", end=" ", flush=True)
            pred, acc = train_and_predict(df, hour)
            if pred is not None:
                # Güven yüzdesi: tahmin büyüdükçe güven artsın (max 85)
                confidence = min(abs(pred * 100) * 1.5, 85) + 15
                
                predictions[f"{hour}h"] = {
                    "expected_return_pct": round(pred * 100, 2),
                    "expected_price": round(float(df['close'].iloc[-1]) * (1 + pred), 2),
                    "direction": "📈 YUKARI" if pred > 0 else "📉 ASAGI",
                    "model_accuracy": round(acc, 1),
                    "confidence_pct": round(confidence, 1)
                }
                print(f"✅ %{round(pred*100,2)} (güven: %{round(confidence,1)})")
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
        # JSON
        with open(f"{OUTPUT_DIR}/tahminler.json", "w", encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        
        # Excel (ZENGİN SÜTUNLARLA)
        rows = []
        for p in all_predictions:
            for h, pred in p['predictions'].items():
                rows.append({
                    'coin': p['coin'],
                    'son_fiyat_usd': p['last_price'],
                    'tahmin_saati': h,
                    'beklenen_getiri_yuzde': pred['expected_return_pct'],
                    'beklenen_fiyat_usd': pred['expected_price'],
                    'yon': pred['direction'],
                    'guven_yuzde': pred['confidence_pct'],
                    'model_gecmis_dogruluk_yuzde': pred['model_accuracy'],
                    'tahmin_tarihi': p['timestamp']
                })
        
        df_excel = pd.DataFrame(rows)
        df_excel.to_excel(f"{OUTPUT_DIR}/tahminler.xlsx", index=False)
        
        print("\n" + "=" * 70)
        print(f"✅ {len(all_predictions)}/{len(COINS)} coin başarılı")
        print(f"📁 JSON: {OUTPUT_DIR}/tahminler.json")
        print(f"📊 Excel: {OUTPUT_DIR}/tahminler.xlsx")
        print("=" * 70)
        
        # Özet tablo
        print("\n📈 ÖZET TABLO (En yüksek getiri beklenenler):")
        summary_df = df_excel.nlargest(10, 'beklenen_getiri_yuzde')
        for _, row in summary_df.iterrows():
            print(f"   {row['coin']:12s} {row['tahmin_saati']:4s}: %{row['beklenen_getiri_yuzde']:6.2f} "
                  f"(güven: %{row['guven_yuzde']:.1f})")

if __name__ == "__main__":
    main()
