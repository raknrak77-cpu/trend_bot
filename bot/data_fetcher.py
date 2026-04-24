import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression

print("=" * 60)
print("🚀 TREND BOT - Yahoo Finance + Tahmin Modeli")
print("=" * 60)

# Ana çıktı klasörü (geçici, sonra Actions tarafından tarihli klasöre taşınacak)
OUTPUT_DIR = "veri"

# Eğer Actions'tan timestamp geldiyse kullan, yoksa şimdiki zaman
RUN_TIMESTAMP = os.environ.get('RUN_TIMESTAMP', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

COINS = {
    "bitcoin": "BTC-USD",
    "ethereum": "ETH-USD",
    "binancecoin": "BNB-USD",
    "solana": "SOL-USD",
    "cardano": "ADA-USD",
    "ripple": "XRP-USD",
    "dogecoin": "DOGE-USD"
}
LOOKBACK_HOURS = 2000

# ============================================================
# 1. VERİ ÇEKME
# ============================================================

def fetch_yahoo_data(symbol="BTC-USD", hours=2000):
    """Yahoo Finance'ten 1 saatlik veri çek"""
    print(f"   📡 {symbol} çekiliyor...")
    
    try:
        ticker = yf.Ticker(symbol)
        days = hours / 24 + 1
        end = datetime.now()
        start = end - timedelta(days=days)
        
        df = ticker.history(start=start, end=end, interval="1h")
        
        if df.empty:
            print(f"   ⚠️ {symbol} için veri yok")
            return None
        
        df = df.reset_index()
        df = df.rename(columns={
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Zaman dilimini temizle (Excel hatası için)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        print(f"   ✅ {len(df)} saatlik veri alındı")
        return df
    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:80]}")
        return None


# ============================================================
# 2. TEKNİK GÖSTERGELER (Özellik Mühendisliği)
# ============================================================

def add_indicators(df):
    """Teknik göstergeler ekle (RSI, MACD, SMA, ATR, trend)"""
    if len(df) < 24:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()      # 1 gün
    df['sma_72'] = df['close'].rolling(72).mean()      # 3 gün
    df['sma_168'] = df['close'].rolling(168).mean()    # 7 gün
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_24'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # RSI (14 saat)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['macd'] = df['ema_12'] - df['ema_24']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ATR (Average True Range) - volatilite
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Saatlik getiri
    df['hourly_return'] = df['close'].pct_change(1) * 100
    
    # Hacim değişimi
    df['volume_change'] = df['volume'].pct_change(1) * 100
    
    # Trend etiketi
    def get_trend(row):
        if pd.isna(row['sma_24']):
            return "YETERSİZ"
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


# ============================================================
# 3. GELECEK TAHMİNİ (Linear Regression Model)
# ============================================================

def train_and_predict(df, coin_name):
    """Geçmiş veriyi kullanarak gelecek 1-72 saat için tahmin yap"""
    
    if len(df) < 100:
        print(f"      ⚠️ {coin_name} için yetersiz veri ({len(df)} satır)")
        return None
    
    # Kullanılacak özellikler
    features = ['rsi_14', 'macd', 'macd_signal', 'sma_24', 'sma_168', 'hourly_return', 'atr_14']
    
    # Hedef saatler
    target_hours = [1, 4, 12, 24, 48, 72]
    
    predictions = {}
    
    for hour in target_hours:
        # Hedef sütunu oluştur: 'hour' saat sonraki getiri
        df[f'target_{hour}h'] = df['close'].shift(-hour) / df['close'] - 1
        
        # NaN'leri temizle
        df_clean = df[features + [f'target_{hour}h']].dropna()
        
        if len(df_clean) < 50:
            continue
        
        # Model eğit
        X = df_clean[features].values
        y = df_clean[f'target_{hour}h'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Son satır ile tahmin yap
        son_veri = df[features].iloc[-1].values.reshape(1, -1)
        tahmin = float(model.predict(son_veri)[0])
        
        # Son 10 tahmin ile yön doğruluğu hesapla
        y_pred = model.predict(X[-10:])
        y_true = y[-10:]
        yon_dogruluk = np.mean((y_pred > 0) == (y_true > 0)) * 100
        
        predictions[f"{hour}h"] = {
            "expected_return_pct": round(tahmin * 100, 2),
            "expected_price": round(float(df['close'].iloc[-1]) * (1 + tahmin), 2),
            "direction": "📈 YÜKSELİŞ" if tahmin > 0 else "📉 DÜŞÜŞ",
            "confidence_pct": round(min(abs(tahmin * 100) * 2, 85) + 15, 1),
            "direction_accuracy_10d": round(yon_dogruluk, 1)
        }
    
    return predictions if predictions else None


# ============================================================
# 4. KAYDETME VE RAPORLAMA
# ============================================================

def save_outputs(df, coin_name, symbol, output_dir):
    """Excel, JSON ve tahminleri kaydet"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Teknik göstergeleri ekle
    df = add_indicators(df)
    
    # Excel (tüm veri)
    excel_name = f"{output_dir}/{coin_name}_1h.xlsx"
    df.to_excel(excel_name, index=False)
    
    # Tahmin yap
    print(f"   🔮 {coin_name} için gelecek tahmini yapılıyor...")
    predictions = train_and_predict(df, coin_name)
    
    # JSON özet (son durum + tahminler)
    son = df.iloc[-1]
    summary = {
        "coin": coin_name,
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "last_price": float(son['close']),
        "trend": son['trend'],
        "rsi_14": float(son['rsi_14']) if pd.notna(son['rsi_14']) else None,
        "rsi_status": son['rsi_status'],
        "macd": float(son['macd']) if pd.notna(son['macd']) else None,
        "macd_signal": float(son['macd_signal']) if pd.notna(son['macd_signal']) else None,
        "atr_14": float(son['atr_14']) if pd.notna(son['atr_14']) else None,
        "sma_24": float(son['sma_24']) if pd.notna(son['sma_24']) else None,
        "sma_168": float(son['sma_168']) if pd.notna(son['sma_168']) else None,
        "hourly_return": float(son['hourly_return']) if pd.notna(son['hourly_return']) else None,
        "volume_24h": float(df.tail(24)['volume'].sum()),
        "data_points": len(df),
        "predictions": predictions
    }
    
    json_name = f"{output_dir}/{coin_name}_1h.json"
    with open(json_name, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return excel_name, json_name, predictions


# ============================================================
# 5. ANA FONKSİYON
# ============================================================

def main():
    # Geçici çıktı klasörü
    temp_dir = OUTPUT_DIR
    
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    print(f"📋 Coin sayısı: {len(COINS)}")
    print(f"⏰ Geriye dönük: {LOOKBACK_HOURS} saat ({LOOKBACK_HOURS//24} gün)")
    print(f"🕐 Çalışma ID: {RUN_TIMESTAMP}")
    print("-" * 60)
    
    results = []
    all_predictions = []
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")
        df = fetch_yahoo_data(symbol, LOOKBACK_HOURS)
        
        if df is not None and len(df) > 0:
            excel_file, json_file, predictions = save_outputs(df, coin_name, symbol, temp_dir)
            
            results.append({
                "coin": coin_name,
                "last_price": float(df.iloc[-1]['close']),
                "trend": df.iloc[-1]['trend'] if len(df) > 0 else "VERİ YOK",
                "data_points": len(df)
            })
            
            if predictions:
                all_predictions.append({
                    "coin": coin_name,
                    "last_price": float(df.iloc[-1]['close']),
                    "predictions": predictions
                })
            
            print(f"   ✅ Excel: {excel_file}")
            print(f"   📈 Son fiyat: ${df.iloc[-1]['close']:,.2f}")
            print(f"   📊 Trend: {df.iloc[-1]['trend']}")
            
            # Tahminleri ekrana yazdır
            if predictions:
                print(f"   🔮 Tahminler:")
                for h, pred in predictions.items():
                    yon = "🟢" if pred['direction'] == "📈 YÜKSELİŞ" else "🔴"
                    print(f"      {h}: {yon} {pred['direction']} %{pred['expected_return_pct']} "
                          f"(güven: %{pred['confidence_pct']})")
        else:
            print(f"   ❌ {coin_name} için veri alınamadı")
    
    # Master rapor (tüm coinlerin son durumu)
    if results:
        master_df = pd.DataFrame(results)
        master_excel = f"{temp_dir}/master_rapor.xlsx"
        master_df.to_excel(master_excel, index=False)
        
        with open(f"{temp_dir}/master_rapor.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Master rapor: {master_excel}")
    
    # Tahminler master raporu
    if all_predictions:
        pred_rows = []
        for p in all_predictions:
            for h, pred in p['predictions'].items():
                pred_rows.append({
                    'coin': p['coin'],
                    'son_fiyat': p['last_price'],
                    'tahmin_saati': h,
                    'beklenen_getiri_yuzde': pred['expected_return_pct'],
                    'beklenen_fiyat': pred['expected_price'],
                    'yon': pred['direction'],
                    'guven_yuzde': pred['confidence_pct'],
                    'yon_dogruluk_10d': pred.get('direction_accuracy_10d', None)
                })
        
        pred_df = pd.DataFrame(pred_rows)
        pred_excel = f"{temp_dir}/tahminler.xlsx"
        pred_df.to_excel(pred_excel, index=False)
        
        pred_json = f"{temp_dir}/tahminler.json"
        with open(pred_json, 'w') as f:
            json.dump(all_predictions, f, indent=2)
        
        print(f"🔮 Tahminler raporu: {pred_excel}")
    
    print("\n" + "=" * 60)
    print(f"✅ {len(results)}/{len(COINS)} coin başarılı")
    print(f"📁 Tüm çıktılar: {temp_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
