import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_yahoo_data(symbol="BTC-USD", hours=3000):
    """Yahoo Finance'ten saatlik veri çek"""
    try:
        ticker = yf.Ticker(symbol)
        days = hours / 24 + 10
        end = datetime.now()
        start = end - timedelta(days=days)

        df = ticker.history(start=start, end=end, interval="1h")

        if df.empty:
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
        df = df.tail(hours).reset_index(drop=True)
        return df

    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:80]}")
        return None


def add_features(df):
    """
    Teknik göstergeler + gürültü azaltma feature'ları.

    YENİLER:
    - lag_1h / lag_4h / lag_24h  → geçmiş getiri bilgisi (LSTM için kritik)
    - vol_regime               → volatilite rejimi (yüksek/düşük gürültü tespiti)
    - volume_ratio             → ani hacim spikeleri
    - price_vs_sma             → fiyatın ortalamalara göre pozisyonu
    - hour_of_day              → saatlik mevsimsellik
    """
    if len(df) < 200:
        return df

    # ── Hareketli Ortalamalar ───────────────────────────────────────────────
    df['sma_24']  = df['close'].rolling(24).mean()
    df['sma_72']  = df['close'].rolling(72).mean()
    df['sma_168'] = df['close'].rolling(168).mean()

    # Fiyatın SMA'ya göre normalize pozisyonu (gürültüsüz trend sinyali)
    df['price_vs_sma24']  = (df['close'] - df['sma_24'])  / df['sma_24']
    df['price_vs_sma72']  = (df['close'] - df['sma_72'])  / df['sma_72']
    df['price_vs_sma168'] = (df['close'] - df['sma_168']) / df['sma_168']

    # ── RSI ────────────────────────────────────────────────────────────────
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # ── MACD ───────────────────────────────────────────────────────────────
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']   # histogram sinyali

    # ── ATR ────────────────────────────────────────────────────────────────
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift()).abs()
    lc  = (df['low']  - df['close'].shift()).abs()
    df['atr_14'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    # ── Bollinger ──────────────────────────────────────────────────────────
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper']    = bb_mid + bb_std * 2
    df['bb_lower']    = bb_mid - bb_std * 2
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    df['bb_width']    = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)  # sıkışma tespiti

    # ── Getiri ve Momentum ─────────────────────────────────────────────────
    df['hourly_return'] = df['close'].pct_change(1)
    df['momentum_12']   = df['close'].pct_change(12)
    df['momentum_24']   = df['close'].pct_change(24)

    # YENİ: Lag features (LSTM'e geçmiş bağlamı ver)
    df['lag_1h']  = df['hourly_return'].shift(1)
    df['lag_4h']  = df['close'].pct_change(4).shift(1)
    df['lag_24h'] = df['close'].pct_change(24).shift(1)

    # ── Volatilite Rejimi ──────────────────────────────────────────────────
    # Son 24 saatin std'si / son 168 saatin std'si → 1'den büyükse yüksek gürültü
    roll_std_24  = df['hourly_return'].rolling(24).std()
    roll_std_168 = df['hourly_return'].rolling(168).std()
    df['vol_regime'] = roll_std_24 / (roll_std_168 + 1e-10)

    # ── Hacim Özellikleri ──────────────────────────────────────────────────
    df['volume_change'] = df['volume'].pct_change(1)
    df['volume_ratio']  = df['volume'] / (df['volume'].rolling(24).mean() + 1e-10)  # ortalamaya göre

    # ── Zaman Mevsimselliği ────────────────────────────────────────────────
    df['hour_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.hour / 24)

    return df
    
