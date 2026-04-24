import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow / LSTM kütüphaneleri
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 70)
print("🚀 TREND BOT - LSTM MODEL (15 Coin | 14 Saat Tahmin)")
print("=" * 70)

OUTPUT_DIR = "veri"

# ============================================================
# 15 COIN (Tam liste)
# ============================================================
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

LOOKBACK_HOURS = 3000      # 125 gün geçmiş
SEQUENCE_LENGTH = 168      # 1 haftalık sequence (168 saat)
TARGET_HOURS = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 36, 48, 72]  # 14 hedef

# ============================================================
# 1. VERİ ÇEKME
# ============================================================

def fetch_yahoo_data(symbol="BTC-USD", hours=3000):
    """Yahoo Finance'ten veri çek"""
    print(f"   📡 {symbol} çekiliyor...")
    try:
        ticker = yf.Ticker(symbol)
        days = hours / 24 + 7
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
        
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df = df.tail(hours)
        print(f"   ✅ {len(df)} saatlik veri alındı")
        return df
    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:80]}")
        return None


# ============================================================
# 2. ÖZELLİK MÜHENDİSLİĞİ
# ============================================================

def add_features_for_lstm(df):
    """Teknik göstergeler ekle"""
    if len(df) < SEQUENCE_LENGTH + 100:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()
    df['sma_72'] = df['close'].rolling(72).mean()
    df['sma_168'] = df['close'].rolling(168).mean()
    
    # RSI (14)
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
    
    # ATR (volatilite)
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Hacim değişimi
    df['volume_change'] = df['volume'].pct_change(1) * 100
    
    # Momentum (son 12 saat)
    df['momentum_12'] = df['close'].pct_change(12) * 100
    
    # Bollinger Bantları
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Saatlik getiri
    df['hourly_return'] = df['close'].pct_change(1) * 100
    
    return df


# ============================================================
# 3. LSTM MODELİ
# ============================================================

def create_lstm_model(input_shape):
    """LSTM modeli oluştur"""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(df, features, target_hour):
    """Sequence hazırla"""
    df[f'target_{target_hour}h'] = df['close'].shift(-target_hour) / df['close'] - 1
    data = df[features + [f'target_{target_hour}h']].dropna()
    
    if len(data) < SEQUENCE_LENGTH + 50:
        return None, None, None
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(data[features].values)
    y_scaled = scaler_y.fit_transform(data[[f'target_{target_hour}h']].values)
    
    X, y = [], []
    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        X.append(X_scaled[i:i + SEQUENCE_LENGTH])
        y.append(y_scaled[i + SEQUENCE_LENGTH])
    
    return np.array(X), np.array(y), (scaler_X, scaler_y)

def train_and_predict_lstm(df, coin_name, target_hour):
    """LSTM eğit ve tahmin yap"""
    features = ['close', 'volume', 'rsi_14', 'macd', 'atr_14', 
                'hourly_return', 'bb_position', 'momentum_12']
    
    X, y, scalers = prepare_sequences(df, features, target_hour)
    
    if X is None or len(X) < 100:
        return None, None
    
    # Train/validation split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Model eğit
    model = create_lstm_model((SEQUENCE_LENGTH, len(features)))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=50, 
              batch_size=32,
              callbacks=[early_stop],
              verbose=0)
    
    # Tahmin
    last_sequence = X[-1:]
    prediction_scaled = model.predict(last_sequence, verbose=0)
    prediction = float(scalers[1].inverse_transform(prediction_scaled)[0, 0])
    
    # Validation doğruluğu
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_actual = scalers[1].inverse_transform(y_val)
    y_val_pred_actual = scalers[1].inverse_transform(y_val_pred)
    
    direction_accuracy = np.mean(
        ((y_val_pred_actual > 0) == (y_val_actual > 0)).flatten()
    ) * 100
    
    return prediction, direction_accuracy


# ============================================================
# 4. ANA FONKSİYON
# ============================================================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print(f"📋 Toplam coin: {len(COINS)}")
    print(f"⏰ Geçmiş: {LOOKBACK_HOURS} saat ({LOOKBACK_HOURS//24} gün)")
    print(f"🎯 Hedef saatler: {len(TARGET_HOURS)} adet ({TARGET_HOURS[0]}-{TARGET_HOURS[-1]}h)")
    print(f"🧠 LSTM sequence: {SEQUENCE_LENGTH} saat (1 hafta)")
    print("=" * 70)
    
    all_predictions = []
    successful_coins = 0
    
    for coin_name, symbol in COINS.items():
        print(f"\n🪙 {coin_name.upper()} ({symbol})")
        df = fetch_yahoo_data(symbol, LOOKBACK_HOURS)
        
        if df is None or len(df) < SEQUENCE_LENGTH:
            print(f"   ❌ Yetersiz veri, atlanıyor")
            continue
        
        df = add_features_for_lstm(df)
        df = df.dropna()
        
        predictions = {}
        
        for target_hour in TARGET_HOURS:
            print(f"      🔮 {target_hour}h LSTM eğitiliyor...", end=" ")
            pred, accuracy = train_and_predict_lstm(df, coin_name, target_hour)
            
            if pred is not None:
                predictions[f"{target_hour}h"] = {
                    "expected_return_pct": round(pred * 100, 2),
                    "expected_price": round(float(df['close'].iloc[-1]) * (1 + pred), 2),
                    "direction": "📈 YUKARI" if pred > 0 else "📉 ASAGI",
                    "model_accuracy": round(accuracy, 1)
                }
                print(f"✅ %{round(pred*100,2)} (doğruluk: %{round(accuracy,1)})")
            else:
                print(f"❌ Yetersiz veri")
        
        if predictions:
            all_predictions.append({
                "coin": coin_name,
                "last_price": float(df['close'].iloc[-1]),
                "timestamp": datetime.now().isoformat(),
                "predictions": predictions
            })
            successful_coins += 1
            
            # Özet
            print(f"\n   📈 Son fiyat: ${df['close'].iloc[-1]:,.2f}")
            print(f"   📊 Başarılı tahmin: {len(predictions)}/{len(TARGET_HOURS)} saat")
    
    # SONUÇLARI KAYDET
    if all_predictions:
        # JSON
        json_path = f"{OUTPUT_DIR}/tahminler.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        
        # Excel (sadece tahminler)
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
                    'model_doğruluk_yuzde': pred['model_accuracy']
                })
        
        pred_df = pd.DataFrame(rows)
        excel_path = f"{OUTPUT_DIR}/tahminler.xlsx"
        pred_df.to_excel(excel_path, index=False)
        
        print("\n" + "=" * 70)
        print("🎉 LSTM MODEL TAMAMLANDI!")
        print(f"✅ Başarılı coin: {successful_coins}/{len(COINS)}")
        print(f"📁 JSON: {json_path}")
        print(f"📊 Excel: {excel_path}")
        print("=" * 70)
    else:
        print("\n❌ Hiçbir tahmin yapılamadı!")

if __name__ == "__main__":
    main()
