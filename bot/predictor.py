import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

SEQUENCE_LENGTH = 168  # 1 hafta

def create_lstm_model(input_shape):
    """LSTM modeli oluştur - Gelişmiş versiyon"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(df, features, target_hour):
    """LSTM için sequence hazırla"""
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

def train_and_predict(df, target_hour):
    """Model eğit ve tahmin yap - YENİ ÖZELLİKLER EKLENDİ"""
    
    # 🆕 GELİŞMİŞ ÖZELLİKLER (15 özellik)
    features = [
        # Temel fiyat ve hacim
        'close', 'volume',
        
        # Trend göstergeleri
        'sma_24', 'sma_168',
        
        # Momentum göstergeleri
        'rsi_14', 'macd', 'macd_signal',
        
        # Volatilite
        'atr_14', 'bb_position',
        
        # Getiri ve hacim değişimi
        'hourly_return', 'volume_change', 'momentum_12'
    ]
    
    X, y, scalers = prepare_sequences(df, features, target_hour)
    
    if X is None or len(X) < 100:
        return None, None
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
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
    
    # Doğruluk
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_actual = scalers[1].inverse_transform(y_val)
    y_val_pred_actual = scalers[1].inverse_transform(y_val_pred)
    
    direction_accuracy = np.mean(
        ((y_val_pred_actual > 0) == (y_val_actual > 0)).flatten()
    ) * 100
    
    return prediction, direction_accuracy
