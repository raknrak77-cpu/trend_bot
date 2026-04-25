import pandas as pd
import json
import os
from datetime import datetime

from data_fetcher import fetch_yahoo_data, add_features
from predictor import train_and_predict

print("=" * 70)
print("🚀 TREND BOT v2 - LSTM + Konsensüs Filtresi")
print("=" * 70)

OUTPUT_DIR = "veri"
COINS = {
    "bitcoin":   "BTC-USD",
    "ethereum":  "ETH-USD",
    "ripple":    "XRP-USD",
    "solana":    "SOL-USD",     
}

LOOKBACK_HOURS = 3000
TARGET_HOURS   = [4, 12, 24, 36, 48, 72]   # 10 → 5 saat: az ama kaliteli


# ── Gerçek Confidence Hesabı ───────────────────────────────────────────────
def calc_confidence(pred_pct: float, direction_accuracy: float, vol_regime: float) -> float:
    """
    Eski formül: abs(pred) * 1.5 + 15   → anlamsız (büyük tahmin = güvenilir değil)
    Yeni formül: model accuracy + volatilite cezası + büyüklük bonusu

    Aralık: 0–100
    """
    base        = direction_accuracy          # model gerçek geçmiş doğruluğu
    vol_penalty = min((vol_regime - 1) * 15, 25) if vol_regime > 1 else 0
    size_bonus  = min(abs(pred_pct) * 2, 10)  # büyük tahmin: küçük bonus (max 10)
    confidence  = base - vol_penalty + size_bonus
    return round(max(10.0, min(confidence, 95.0)), 1)


# ── Konsensüs Oylama ───────────────────────────────────────────────────────
def consensus_signal(predictions: dict) -> dict:
    """
    Birden fazla zaman diliminin yönüne bak.
    En az %70'i aynı yönü gösteriyorsa sinyal ver, yoksa 'NÖTR' döner.
    """
    if not predictions:
        return {}

    directions = [v['direction'] for v in predictions.values()]
    total = len(directions)
    up    = directions.count("📈 YUKARI")
    down  = directions.count("📉 ASAGI")

    consensus_pct = max(up, down) / total * 100
    dominant = "📈 YUKARI" if up >= down else "📉 ASAGI"

    return {
        "consensus_direction": dominant if consensus_pct >= 70 else "⚖️ NÖTR",
        "consensus_strength_pct": round(consensus_pct, 1),
        "up_votes":   up,
        "down_votes": down,
        "total_votes": total
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📋 Coin: {len(COINS)}")
    print(f"🎯 Hedef saatler: {TARGET_HOURS}")
    print("=" * 70)

    all_predictions = []

    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")

        df = fetch_yahoo_data(symbol, LOOKBACK_HOURS)
        if df is None or len(df) < 500:
            print("   ❌ Yetersiz veri")
            continue

        df = add_features(df)
        df = df.dropna()

        # Son 6 saatin ortalama volatilite rejimi (confidence için)
        vol_regime_now = float(df['vol_regime'].dropna().tail(6).mean()) \
            if 'vol_regime' in df.columns else 1.0

        predictions = {}
        for hour in TARGET_HOURS:
            print(f"      🔮 {hour}h...", end=" ", flush=True)
            pred, acc = train_and_predict(df, hour)

            if pred is not None:
                pred_pct   = round(pred * 100, 2)
                confidence = calc_confidence(pred_pct, acc, vol_regime_now)

                predictions[f"{hour}h"] = {
                    "expected_return_pct": pred_pct,
                    "expected_price":      round(float(df['close'].iloc[-1]) * (1 + pred), 2),
                    "direction":           "📈 YUKARI" if pred > 0 else "📉 ASAGI",
                    "model_accuracy":      round(acc, 1),
                    "confidence_pct":      confidence,
                    "vol_regime":          round(vol_regime_now, 2),
                }
                print(f"✅ %{pred_pct:+.2f} | doğruluk: %{acc:.1f} | güven: %{confidence}")
            else:
                print("⏭️  atlandı (düşük kalite veya yüksek gürültü)")

        if predictions:
            consensus = consensus_signal(predictions)
            print(f"   📊 Konsensüs: {consensus.get('consensus_direction')} "
                  f"({consensus.get('consensus_strength_pct')}% - "
                  f"{consensus.get('up_votes')}↑ / {consensus.get('down_votes')}↓)")

            all_predictions.append({
                "coin":       coin_name,
                "last_price": float(df['close'].iloc[-1]),
                "timestamp":  datetime.now().isoformat(),
                "vol_regime": round(vol_regime_now, 2),
                "consensus":  consensus,
                "predictions": predictions
            })

    # ── Kaydet ────────────────────────────────────────────────────────────
    if not all_predictions:
        print("\n❌ Hiç başarılı tahmin üretilemedi.")
        return

    # JSON
    json_path = f"{OUTPUT_DIR}/tahminler.json"
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(all_predictions, f, indent=2, ensure_ascii=False)

    # Excel
    rows = []
    for p in all_predictions:
        for h, pred in p['predictions'].items():
            rows.append({
                'coin':                          p['coin'],
                'son_fiyat_usd':                 p['last_price'],
                'vol_regime':                    p['vol_regime'],
                'konsensus_yonu':                p['consensus'].get('consensus_direction', '-'),
                'konsensus_guc':                 p['consensus'].get('consensus_strength_pct', 0),
                'tahmin_saati':                  h,
                'beklenen_getiri_yuzde':         pred['expected_return_pct'],
                'beklenen_fiyat_usd':            pred['expected_price'],
                'yon':                           pred['direction'],
                'guven_yuzde':                   pred['confidence_pct'],
                'model_gecmis_dogruluk_yuzde':   pred['model_accuracy'],
                'tahmin_tarihi':                 p['timestamp']
            })

    df_excel = pd.DataFrame(rows)
    df_excel.to_excel(f"{OUTPUT_DIR}/tahminler.xlsx", index=False)

    print("\n" + "=" * 70)
    print(f"✅ {len(all_predictions)}/{len(COINS)} coin başarılı")
    print(f"📁 JSON : {json_path}")
    print(f"📊 Excel: {OUTPUT_DIR}/tahminler.xlsx")
    print("=" * 70)

    # ── Özet: sadece güvenilir sinyaller ──────────────────────────────────
    print("\n🏆 GÜVENİLİR SİNYALLER (konsensüs ≥ %70, güven ≥ %60):")
    for p in all_predictions:
        cons = p['consensus']
        if cons.get('consensus_direction') == "⚖️ NÖTR":
            continue
        for h, pred in p['predictions'].items():
            if pred['confidence_pct'] >= 60:
                print(f"   {p['coin']:12s} {h:4s}: {pred['direction']}  "
                      f"%{pred['expected_return_pct']:+.2f}  "
                      f"güven:%{pred['confidence_pct']}")


if __name__ == "__main__":
    main()
    
