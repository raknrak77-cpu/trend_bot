import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def train_and_predict(coin_name, veri_klasoru):
    """
    Her coin için basit bir regresyon modeli eğitir ve gelecek tahmini yapar.
    """
    # Excel'den veriyi oku
    excel_path = f"{veri_klasoru}/{coin_name}_1h.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"   ❌ {coin_name} için veri dosyası bulunamadı")
        return None
    
    df = pd.read_excel(excel_path)
    
    if len(df) < 100:
        print(f"   ⚠️ {coin_name} için yetersiz veri ({len(df)} satır)")
        return None
    
    print(f"   📊 {coin_name}: {len(df)} saatlik veri ile model eğitiliyor...")
    
    # Özellikler (kullanacağımız göstergeler)
    features = ['rsi_14', 'macd', 'macd_signal', 'sma_24', 'sma_168', 'hourly_return']
    
    # Hedef: gelecek 1, 4, 12, 24, 48, 72 saat sonraki fiyat
    target_hours = [1, 4, 12, 24, 48, 72]
    
    results = {
        "coin": coin_name,
        "timestamp": datetime.now().isoformat(),
        "last_price": float(df['close'].iloc[-1]),
        "predictions": {}
    }
    
    for hour in target_hours:
        # Hedef sütunu oluştur: 'hour' saat sonraki fiyat / şimdiki fiyat
        df[f'target_{hour}h'] = df['close'].shift(-hour) / df['close'] - 1
        
        # NaN'leri temizle
        df_clean = df[features + [f'target_{hour}h']].dropna()
        
        if len(df_clean) < 50:
            print(f"      ⚠️ {hour}h için yeterli veri yok ({len(df_clean)} satır)")
            continue
        
        # Model eğit
        X = df_clean[features].values
        y = df_clean[f'target_{hour}h'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Son satır ile tahmin yap
        son_veri = df[features].iloc[-1].values.reshape(1, -1)
        tahmin = float(model.predict(son_veri)[0])
        
        # Tahmini yüzdeye çevir
        yuzde_tahmin = tahmin * 100
        
        # Son 10 tahmin ile basit doğruluk hesapla (işaret bazlı)
        y_pred = model.predict(X[-10:])
        y_true = y[-10:]
        yon_dogruluk = np.mean((y_pred > 0) == (y_true > 0)) * 100
        
        results["predictions"][f"{hour}h"] = {
            "expected_return_pct": round(yuzde_tahmin, 2),
            "expected_price": round(float(df['close'].iloc[-1]) * (1 + tahmin), 2),
            "direction": "📈 YÜKSELİŞ" if tahmin > 0 else "📉 DÜŞÜŞ",
            "confidence_pct": round(min(abs(yuzde_tahmin) * 5, 85) + 15, 1),
            "direction_accuracy_10d": round(yon_dogruluk, 1)
        }
    
    return results

def predict_all_coins(veri_klasoru, output_dir):
    """Tüm coinler için tahmin yap"""
    print("\n" + "=" * 60)
    print("🔮 GELECEK TAHMİNLERİ (1-72 saat)")
    print("=" * 60)
    
    # Hangi coinler var?
    coin_files = [f for f in os.listdir(veri_klasoru) if f.endswith('_1h.xlsx')]
    coins = [f.replace('_1h.xlsx', '') for f in coin_files]
    
    print(f"📋 İşlenecek coinler: {coins}")
    print("-" * 60)
    
    all_predictions = []
    
    for coin in coins:
        print(f"\n🪙 {coin.upper()}")
        result = train_and_predict(coin, veri_klasoru)
        
        if result:
            all_predictions.append(result)
            
            # Ekrana yazdır
            print(f"   📈 Son fiyat: ${result['last_price']:,.2f}")
            for h, pred in result['predictions'].items():
                yon_emoji = "🟢" if pred['direction'] == "📈 YÜKSELİŞ" else "🔴"
                print(f"      {h}: {yon_emoji} {pred['direction']} %{pred['expected_return_pct']} "
                      f"(güven: %{pred['confidence_pct']})")
    
    # Tahminleri kaydet
    if all_predictions:
        # JSON
        json_path = f"{output_dir}/tahminler.json"
        with open(json_path, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        # Excel
        rows = []
        for p in all_predictions:
            for h, pred in p['predictions'].items():
                rows.append({
                    'coin': p['coin'],
                    'son_fiyat': p['last_price'],
                    'tahmin_saati': h,
                    'beklenen_getiri_yuzde': pred['expected_return_pct'],
                    'beklenen_fiyat': pred['expected_price'],
                    'yon': pred['direction'],
                    'guven_yuzde': pred['confidence_pct']
                })
        
        df_predictions = pd.DataFrame(rows)
        excel_path = f"{output_dir}/tahminler.xlsx"
        df_predictions.to_excel(excel_path, index=False)
        
        print("\n" + "=" * 60)
        print("✅ Tahminler tamamlandı!")
        print(f"📁 JSON: {json_path}")
        print(f"📊 Excel: {excel_path}")
        print("=" * 60)
    
    return all_predictions
