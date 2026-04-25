import pandas as pd
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
import glob

print("=" * 70)
print("📊 PERFORMANS DEĞERLENDİRME BOTU v2")
print("=" * 70)

VERI_KLASORU = "veri"
COINS = {
    "bitcoin":   "BTC-USD",
    "ethereum":  "ETH-USD",
    "ripple":    "XRP-USD",
    "solana":    "SOL-USD",
}

# Cache: aynı sembol için tekrar tekrar API çağrısı yapma
_price_cache: dict = {}


def get_gerceklesen_fiyat(symbol: str, target_date: datetime) -> float | None:
    """Belirli tarihe en yakın saatlik kapanış fiyatını çek (cache'li)"""
    cache_key = f"{symbol}_{target_date.date()}"
    if cache_key not in _price_cache:
        try:
            ticker = yf.Ticker(symbol)
            start  = target_date - timedelta(days=1)
            end    = target_date + timedelta(days=2)
            df = ticker.history(start=start, end=end, interval="1h")
            if df.empty:
                _price_cache[cache_key] = None
                return None
            df = df.reset_index()
            df['ts'] = df['Datetime'].dt.tz_localize(None)
            _price_cache[cache_key] = df[['ts', 'Close']].copy()
        except Exception:
            _price_cache[cache_key] = None
            return None

    cached = _price_cache[cache_key]
    if cached is None or isinstance(cached, type(None)):
        return None

    cached['diff'] = (cached['ts'] - target_date).abs()
    row = cached.loc[cached['diff'].idxmin()]
    return float(row['Close'])


def get_gecmis_tahminler() -> list:
    """Tüm geçmiş tahmin JSON dosyalarını topla"""
    patterns = [
        f"{VERI_KLASORU}/**/tahminler.json",
        f"{VERI_KLASORU}/tahminler.json",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))

    files = list(set(files))
    files.sort(reverse=True)

    result = []
    for fp in files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
            parts = fp.replace("\\", "/").split("/")
            tarih = parts[-2] if len(parts) > 2 else "latest"
            result.append({'tarih': tarih, 'dosya': fp, 'veri': data})
        except Exception as e:
            print(f"   ⚠️  {fp} okunamadı: {e}")

    return result


def evaluate_one(coin_name, saat_int, beklenen_fiyat,
                 tahmin_tarihi, tahmin_ani_fiyat, dosya_tarih):
    """Tek bir tahmini değerlendir. BUG FIX: tahmin_ani_fiyat parametre olarak alınıyor."""
    symbol = COINS.get(coin_name)
    if not symbol:
        return None

    hedef_tarih = tahmin_tarihi + timedelta(hours=saat_int)
    if hedef_tarih > datetime.now():
        return None   # henüz gerçekleşmedi

    gercek_fiyat = get_gerceklesen_fiyat(symbol, hedef_tarih)
    if gercek_fiyat is None or tahmin_ani_fiyat == 0:
        return None

    beklenen_getiri   = beklenen_fiyat   / tahmin_ani_fiyat - 1
    gerceklesen_getiri = gercek_fiyat   / tahmin_ani_fiyat - 1
    yon_dogru          = (beklenen_getiri > 0) == (gerceklesen_getiri > 0)
    hata_payi          = abs(beklenen_getiri - gerceklesen_getiri)

    return {
        'coin':                     coin_name,
        'tahmin_saati':             saat_int,
        'tahmin_tarihi':            tahmin_tarihi.strftime('%Y-%m-%d %H:%M'),
        'tahmin_ani_fiyat':         round(tahmin_ani_fiyat, 4),
        'beklenen_fiyat':           round(beklenen_fiyat, 4),
        'gerceklesen_fiyat':        round(gercek_fiyat, 4),
        'beklenen_getiri_pct':      round(beklenen_getiri * 100, 2),
        'gerceklesen_getiri_pct':   round(gerceklesen_getiri * 100, 2),
        'yon_dogru':                '✅ EVET' if yon_dogru else '❌ HAYIR',
        'hata_payi_pct':            round(hata_payi * 100, 2),
        'kaynak_dosya':             dosya_tarih,
    }


def main():
    print("📂 Geçmiş tahminler taranıyor...")
    gecmis = get_gecmis_tahminler()

    if not gecmis:
        print("❌ Hiç tahmin dosyası bulunamadı! Önce main.py'yi çalıştırın.")
        return

    print(f"✅ {len(gecmis)} tahmin dosyası bulundu.")
    print("-" * 70)

    tum = []

    for dosya in gecmis:
        print(f"\n📁 {dosya['tarih']}")
        for coin_data in dosya['veri']:
            coin_name = coin_data['coin']
            ts_str    = coin_data.get('timestamp', '')
            ani_fiyat = coin_data.get('last_price', 0)

            try:
                tahmin_tarihi = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                # timezone-aware ise naive'e çevir
                if tahmin_tarihi.tzinfo is not None:
                    tahmin_tarihi = tahmin_tarihi.replace(tzinfo=None)
            except Exception:
                continue

            for saat_str, pred in coin_data.get('predictions', {}).items():
                saat_int      = int(saat_str.replace('h', ''))
                beklenen_fiyat = pred.get('expected_price', 0)

                sonuc = evaluate_one(
                    coin_name, saat_int, beklenen_fiyat,
                    tahmin_tarihi, ani_fiyat, dosya['tarih']
                )
                if sonuc:
                    tum.append(sonuc)

        yeni = [d for d in tum if d['kaynak_dosya'] == dosya['tarih']]
        print(f"   ✅ {len(yeni)} değerlendirme yapıldı")

    if not tum:
        print("\n❌ Henüz değerlendirilebilecek gerçekleşmiş tahmin yok.")
        return

    df = pd.DataFrame(tum)

    # ── Rapor ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 DEĞERLENDİRME RAPORU")
    print("=" * 70)

    genel_basari  = (df['yon_dogru'] == '✅ EVET').mean() * 100
    ort_hata      = df['hata_payi_pct'].mean()
    print(f"\n🎯 GENEL BAŞARI ORANI : %{genel_basari:.1f}  ({len(df)} tahmin)")
    print(f"📉 ORTALAMA HATA PAYI: %{ort_hata:.2f}")

    # Saat bazında başarı
    print("\n⏰ SAAT BAZINDA:")
    saat_bazli = (
        df.groupby('tahmin_saati')
          .apply(lambda x: (x['yon_dogru'] == '✅ EVET').mean() * 100)
          .sort_index()
    )
    for saat, basari in saat_bazli.items():
        n = len(df[df['tahmin_saati'] == saat])
        bar = "█" * int(basari / 5)
        print(f"   {saat:3d}h: %{basari:5.1f}  {bar}  ({n} tahmin)")

    # Coin bazında başarı
    print("\n🪙 COIN BAZINDA:")
    coin_bazli = (
        df.groupby('coin')
          .apply(lambda x: (x['yon_dogru'] == '✅ EVET').mean() * 100)
          .sort_values(ascending=False)
    )
    for coin, basari in coin_bazli.items():
        n = len(df[df['coin'] == coin])
        print(f"   {coin:15s}: %{basari:.1f}  ({n} tahmin)")

    # En iyi / en kötü
    dogru_df = df[df['yon_dogru'] == '✅ EVET']
    yanlis_df = df[df['yon_dogru'] == '❌ HAYIR']

    if not dogru_df.empty:
        print("\n🏆 EN BAŞARILI 5 TAHMİN:")
        for _, r in dogru_df.nlargest(5, 'gerceklesen_getiri_pct').iterrows():
            print(f"   {r['coin']:12s} {r['tahmin_saati']}h: "
                  f"tahmin %{r['beklenen_getiri_pct']:+.2f} → "
                  f"gerçek %{r['gerceklesen_getiri_pct']:+.2f}")

    if not yanlis_df.empty:
        print("\n📉 EN YANLIŞ 5 TAHMİN:")
        for _, r in yanlis_df.nlargest(5, 'hata_payi_pct').iterrows():
            print(f"   {r['coin']:12s} {r['tahmin_saati']}h: "
                  f"tahmin %{r['beklenen_getiri_pct']:+.2f} → "
                  f"gerçek %{r['gerceklesen_getiri_pct']:+.2f} "
                  f"(hata: %{r['hata_payi_pct']:.2f})")

    # ── Kaydet ────────────────────────────────────────────────────────────
    out_dir = f"{VERI_KLASORU}/performans"
    os.makedirs(out_dir, exist_ok=True)

    excel_path = f"{out_dir}/performans_raporu.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Tum_Degerlendirmeler', index=False)

        ozet = pd.DataFrame([
            {'metrik': 'Genel Başarı Oranı (%)',  'değer': round(genel_basari, 1)},
            {'metrik': 'Toplam Değerlendirme',     'değer': len(df)},
            {'metrik': 'Ortalama Hata Payı (%)',   'değer': round(ort_hata, 2)},
        ])
        ozet.to_excel(writer, sheet_name='Ozet', index=False)
        saat_bazli.reset_index().rename(
            columns={0: 'basari_orani_yuzde'}
        ).to_excel(writer, sheet_name='Saat_Bazli', index=False)
        coin_bazli.reset_index().rename(
            columns={0: 'basari_orani_yuzde'}
        ).to_excel(writer, sheet_name='Coin_Bazli', index=False)

    json_path = f"{out_dir}/performans_raporu.json"
    with open(json_path, 'w') as f:
        json.dump({
            'genel_basari_orani':   round(genel_basari, 1),
            'toplam_degerlendirme': len(df),
            'ortalama_hata_payi':   round(ort_hata, 2),
            'saat_bazli':           {int(k): round(v, 1) for k, v in saat_bazli.items()},
            'coin_bazli':           {k: round(v, 1) for k, v in coin_bazli.items()},
            'timestamp':            datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n📁 Raporlar kaydedildi:")
    print(f"   📊 Excel: {excel_path}")
    print(f"   📄 JSON : {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
                                                
