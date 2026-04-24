import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_yahoo_data(symbol="BTC-USD", hours=3000):
    """Yahoo Finance'ten veri çek"""
    try:
        ticker = yf.Ticker(symbol)
        days = hours / 24 + 7
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
        df = df.tail(hours)
        return df
    except Exception as e:
        print(f"   ❌ Hata: {str(e)[:80]}")
        return None

def add_features(df):
    """Teknik göstergeler ekle"""
    if len(df) < 200:
        return df
    
    # Hareketli ortalamalar
    df['sma_24'] = df['close'].rolling(24).mean()
    df['sma_72'] = df['close'].rolling(72).mean()
    df['sma_168'] = df['close'].rolling(168).mean()
    
    # RSI
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
    
    # ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Bollinger
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Momentum ve getiri
    df['momentum_12'] = df['close'].pct_change(12) * 100
    df['hourly_return'] = df['close'].pct_change(1) * 100
    df['volume_change'] = df['volume'].pct_change(1) * 100
    
    return df
