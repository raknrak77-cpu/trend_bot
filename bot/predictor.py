import numpy as np
from sklearn.preprocessing import RobustScaler   # MinMaxScaler → RobustScaler (outlier'lara dayanıklı)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

SEQUENCE_LENGTH = 72   # 168 → 72: kripto'da 1 hafta eskisi artık "bayat"

# ── Genişletilmiş feature seti ──────────────────────────────────────────────
FEATURES = [
    'close', 'volume',
    'rsi_14', 'macd', 'macd_hist',
    'atr_14',
    'hourly_return', 'bb_position', 'bb_width',
    'momentum_12', 'momentum_24',
    'lag_1h', 'lag_4h', 'lag_24h',
    'vol_regime', 'volume_ratio',
    'price_vs_sma24', 'price_vs_sma72',
    'hour_sin', 'hour_cos',
]

# Gürültü filtreleme eşikleri
MIN_DIRECTION_ACCURACY = 52.0   # %52 altı modeller atlanır (yazı tura = %50)
HIGH_VOLATILITY_THRESHOLD = 1.5  # vol_regime > 1.5 → tahmin güvenilmez


def create_lstm_model(input_shape):
    """
    İyileştirmeler:
    - BatchNormalization: eğitim kararlılığı
    - L2 regularization: overfit azaltma
    - Dropout 0.2 → 0.3: gürültü azaltma
    - 2. LSTM katmanı küçültüldü (64 → 32): daha az parametre = daha az overfit
    """
    model = Sequential([
        LSTM(96, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber',   # MSE → Huber: outlier'lara dayanıklı
                  metrics=['mae'])
    return model


def prepare_sequences(df, features, target_hour):
    """
    Sequence hazırlama.
    YENİ: RobustScaler kullan (kripto'nun yüksek outlier'larına karşı)
    """
    available = [f for f in features if f in df.columns]
    if len(available) < 10:
        return None, None, None

    df[f'target_{target_hour}h'] = df['close'].shift(-target_hour) / df['close'] - 1
    data = df[available + [f'target_{target_hour}h']].dropna()

    if len(data) < SEQUENCE_LENGTH + 100:
        return None, None, None

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_scaled = scaler_X.fit_transform(data[available].values)
    y_scaled = scaler_y.fit_transform(data[[f'target_{target_hour}h']].values)

    X, y = [], []
    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        X.append(X_scaled[i:i + SEQUENCE_LENGTH])
        y.append(y_scaled[i + SEQUENCE_LENGTH])

    return np.array(X), np.array(y), (scaler_X, scaler_y, available)


def train_and_predict(df, target_hour):
    """
    Model eğit ve tahmin yap.

    YENİLER:
    1. Yüksek volatilite rejiminde None döner (gürültülü ortamda tahmin yapma)
    2. ReduceLROnPlateau: sıkışınca learning rate düşür
    3. MIN_DIRECTION_ACCURACY filtresi
    4. Prediction'ı son 3 sequence'ın ortalaması (tek nokta gürültüsü azaltır)
    """

    # ── Volatilite rejimi kontrolü ─────────────────────────────────────────
    if 'vol_regime' in df.columns:
        recent_vol = df['vol_regime'].dropna().tail(6).mean()
        if recent_vol > HIGH_VOLATILITY_THRESHOLD:
            return None, None   # yüksek gürültü → geç

    X, y, meta = prepare_sequences(df, FEATURES, target_hour)

    if X is None or len(X) < 150:
        return None, None

    scaler_X, scaler_y, used_features = meta

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = create_lstm_model((SEQUENCE_LENGTH, X.shape[2]))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-5)
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # ── Yön doğruluğu filtresi ─────────────────────────────────────────────
    y_val_pred = model.predict(X_val, verbose=0)
    y_val_actual = scaler_y.inverse_transform(y_val)
    y_val_pred_actual = scaler_y.inverse_transform(y_val_pred)

    direction_accuracy = float(np.mean(
        ((y_val_pred_actual > 0) == (y_val_actual > 0)).flatten()
    ) * 100)

    if direction_accuracy < MIN_DIRECTION_ACCURACY:
        return None, None   # model yazı-tura seviyesinde → at

    # ── Ensemble tahmin: son 3 sequence ortalaması ─────────────────────────
    last_sequences = X[-3:]
    preds_scaled = model.predict(last_sequences, verbose=0)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    prediction = float(np.mean(preds))   # ortalama al → tek nokta spike'ını önle

    return prediction, direction_accuracy
    
