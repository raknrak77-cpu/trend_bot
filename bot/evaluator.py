import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os

print("=" * 70)
print("📊 BACKTEST DEĞERLENDİRME (Geçmiş Veri ile Anında Test)")
print("=" * 70)

# ============================================================
# AYARLAR
# ============================================================
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

TEST_HOURS = 240  # Son 240 saat (10 gün) test edilecek
TARGET_HOURS = [2, 3, 4, 12, 16, 24, 28, 36, 48, 72]  # Tahmin edilecek saatler

# ============================================================
# VERİ ÇEKME (data_fetcher'dan bağımsız)
# ============================================================
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

# ============================================================
# TEKNİK GÖSTERGELER
# ============================================================
def add_features(df):
    """Teknik göstergeler ekle"""
    if len(df) < 200:
        return df
    
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

# ============================================================
# BASİT MODEL (Backtest için hızlı)
# ============================================================
def simple_predict(train_df, test_point, target_hour):
    """Basit bir tahmin modeli (ortalama + trend)"""
    if len(train_df) < 100:
        return None
    
    # Son 24 saatlik ortalama getiri
    avg_return = train_df['hourly_return'].tail(24).mean() / 100
    
    # Trend gücü (RSI bazlı)
    rsi = train_df['rsi_14'].iloc[-1]
    if rsi > 70:
        trend_factor = -0.002  # Aşırı alım, düşüş beklentisi
    elif rsi < 30:
        trend_factor = 0.002   # Aşırı satım, yükseliş beklentisi
    else:
        trend_factor = 0.001
    
    # Tahmin (basit)
    prediction = avg_return * (target_hour / 24) + trend_factor
    
    return prediction

# ============================================================
# BACKTEST
# ============================================================
def backtest_coin(coin_name, symbol):
    """Bir coin için backtest yap"""
    print(f"\n🪙 {coin_name.upper()} ({symbol})")
    
    # Veriyi çek
    df = fetch_yahoo_data(symbol, hours=3000)
    if df is None or len(df) < 500:
        print(f"   ❌ Yetersiz veri")
        return None
    
    df = add_features(df)
    df = df.dropna()
    
    if len(df) < TEST_HOURS + 100:
        print(f"   ❌ Test için yetersiz veri")
        return None
    
    # Eğitim ve test setlerine ayır
    train_df = df.iloc[:-TEST_HOURS].copy()
    test_df = df.iloc[-TEST_HOURS:].copy()
    
    print(f"   📊 Eğitim: {len(train_df)} saat, Test: {len(test_df)} saat")
    
    results = []
    
    # Her test noktası için tahmin yap
    for i, test_idx in enumerate(test_df.index):
        # Test noktasına kadar olan tüm veriyi eğitim olarak kullan
        current_train = df[df.index <= test_idx].copy()
        
        if len(current_train) < 200:
            continue
        
        for hour in TARGET_HOURS:
            # Tahmin yapılacak zaman noktası
            target_idx = test_idx + timedelta(hours=hour)
            
            # Eğer target_idx test setinin dışındaysa atla
            if target_idx not in test_df.index:
                continue
            
            # Tahmin yap
            pred_return = simple_predict(current_train, test_df.loc[test_idx], hour)
            
            if pred_return is None:
                continue
            
            # Gerçekleşen getiri
            current_price = test_df.loc[test_idx]['close']
            future_price = test_df.loc[target_idx]['close']
            actual_return = (future_price - current_price) / current_price
            
            # Başarı kontrolü
            direction_correct = (pred_return > 0) == (actual_return > 0)
            
            results.append({
                'coin': coin_name,
                'test_hour': test_idx.strftime('%Y-%m-%d %H:00'),
                'target_hour': hour,
                'current_price': round(current_price, 2),
                'predicted_price': round(current_price * (1 + pred_return), 2),
                'actual_price': round(future_price, 2),
                'predicted_return_pct': round(pred_return * 100, 2),
                'actual_return_pct': round(actual_return * 100, 2),
                'direction_correct': '✅ EVET' if direction_correct else '❌ HAYIR'
            })
    
    print(f"   ✅ {len(results)} tahmin değerlendirildi")
    return results

# ============================================================
# ANA FONKSİYON
# ============================================================
def main():
    print(f"📋 Test edilecek coin: {len(COINS)}")
    print(f"⏰ Geriye dönük test: {TEST_HOURS} saat ({TEST_HOURS//24} gün)")
    print(f"🎯 Hedef saatler: {TARGET_HOURS}")
    print("=" * 70)
    
    all_results = []
    
    for coin_name, symbol in COINS.items():
        results = backtest_coin(coin_name, symbol)
        if results:
            all_results.extend(results)
    
    if not all_results:
        print("\n❌ Hiçbir sonuç elde edilemedi!")
        return
    
    df = pd.DataFrame(all_results)
    
    # Rapor
    print("\n" + "=" * 70)
    print("📊 BACKTEST RAPORU (Gerçek Geçmiş Performans)")
    print("=" * 70)
    
    # Genel başarı
    overall_accuracy = (df['direction_correct'] == '✅ EVET').mean() * 100
    print(f"\n🎯 GENEL BAŞARI ORANI: %{overall_accuracy:.1f}")
    print(f"   Toplam değerlendirme: {len(df)}")
    
    # Saat bazında başarı
    print("\n⏰ SAAT BAZINDA BAŞARI ORANLARI:")
    for hour in sorted(df['target_hour'].unique()):
        hour_df = df[df['target_hour'] == hour]
        acc = (hour_df['direction_correct'] == '✅ EVET').mean() * 100
        print(f"   {hour:2d} saat: %{acc:.1f} ({len(hour_df)} tahmin)")
    
    # Coin bazında başarı
    print("\n🪙 COIN BAZINDA BAŞARI ORANLARI:")
    for coin in df['coin'].unique():
        coin_df = df[df['coin'] == coin]
        acc = (coin_df['direction_correct'] == '✅ EVET').mean() * 100
        print(f"   {coin:12s}: %{acc:.1f} ({len(coin_df)} tahmin)")
    
    # Ortalama hata
    df['error_pct'] = abs(df['predicted_return_pct'] - df['actual_return_pct'])
    print(f"\n📉 ORTALAMA HATA PAYI: %{df['error_pct'].mean():.2f}")
    
    # Sonuçları kaydet
    output_dir = "veri/backtest"
    os.makedirs(output_dir, exist_ok=True)
    
    excel_path = f"{output_dir}/backtest_raporu.xlsx"
    df.to_excel(excel_path, index=False)
    
    json_path = f"{output_dir}/backtest_raporu.json"
    with open(json_path, 'w') as f:
        json.dump({
            'genel_basari_orani': round(overall_accuracy, 1),
            'toplam_degerlendirme': len(df),
            'ortalama_hata_payi': round(df['error_pct'].mean(), 2),
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📁 Rapor: {excel_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
