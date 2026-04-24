import pandas as pd
import json
import os
import yfinance as yf
from datetime import datetime, timedelta
import glob

print("=" * 70)
print("📊 PERFORMANS DEĞERLENDİRME BOTU")
print("=" * 70)

VERI_KLASORU = "veri"
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

def get_gecmis_tahminler():
    """Tüm geçmiş tahmin dosyalarını topla"""
    json_files = glob.glob(f"{VERI_KLASORU}/**/tahminler.json", recursive=True)
    json_files += glob.glob(f"{VERI_KLASORU}/tahminler.json", recursive=True)
    
    # En yeni önce gelecek şekilde sırala
    json_files.sort(reverse=True)
    
    tum_tahminler = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Dosya adından tarih bilgisini çıkar
                tarih = file_path.split('/')[-2] if '/veri/' in file_path else 'latest'
                tum_tahminler.append({
                    'tarih': tarih,
                    'dosya': file_path,
                    'veri': data
                })
        except Exception as e:
            print(f"   ⚠️ {file_path} okunamadı: {e}")
    
    return tum_tahminler

def get_gerceklesen_fiyat(symbol, target_date):
    """Belirli bir tarihteki gerçek fiyatı çek"""
    try:
        ticker = yf.Ticker(symbol)
        # Hedef tarihten 1 gün önce ve 1 gün sonra veri çek
        start = target_date - timedelta(days=1)
        end = target_date + timedelta(days=2)
        
        df = ticker.history(start=start, end=end, interval="1h")
        
        if df.empty:
            return None
        
        # Hedef tarihe en yakın fiyatı bul
        df = df.reset_index()
        df['timestamp'] = df['Datetime'].dt.tz_localize(None)
        df['diff'] = abs(df['timestamp'] - target_date)
        
        en_yakin = df.loc[df['diff'].idxmin()]
        return float(en_yakin['Close'])
    except Exception as e:
        return None

def evaluate_predictions(tahmin_dosyasi, coin_name, tahmin_saati, beklenen_fiyat, tahmin_tarihi):
    """Tek bir tahmini değerlendir"""
    # Hedef tarihi hesapla (tahmin tarihi + tahmin saati)
    hedef_tarih = tahmin_tarihi + timedelta(hours=tahmin_saati)
    
    # Gerçekleşen fiyatı al
    symbol = COINS.get(coin_name)
    if not symbol:
        return None
    
    gercek_fiyat = get_gerceklesen_fiyat(symbol, hedef_tarih)
    
    if gercek_fiyat is None:
        return None
    
    # Beklenen ve gerçekleşen getiri
    beklenen_getiri = beklenen_fiyat / tahmin_tarihi_fiyat - 1
    gerceklesen_getiri = gercek_fiyat / tahmin_tarihi_fiyat - 1
    
    # Doğru mu?
    yon_dogru = (beklenen_getiri > 0) == (gerceklesen_getiri > 0)
    hata_payi = abs(beklenen_getiri - gerceklesen_getiri)
    
    return {
        'coin': coin_name,
        'tahmin_saati': tahmin_saati,
        'tahmin_tarihi': tahmin_tarihi.strftime('%Y-%m-%d %H:%M'),
        'tahmin_fiyati': beklenen_fiyat,
        'gerceklesen_fiyat': gercek_fiyat,
        'beklenen_getiri_pct': round(beklenen_getiri * 100, 2),
        'gerceklesen_getiri_pct': round(gerceklesen_getiri * 100, 2),
        'yon_dogru': '✅ EVET' if yon_dogru else '❌ HAYIR',
        'hata_payi_pct': round(hata_payi * 100, 2),
        'tahmin_kaynagi': tahmin_dosyasi
    }

def main():
    print("📂 Geçmiş tahminler taranıyor...")
    gecmis_tahminler = get_gecmis_tahminler()
    
    if not gecmis_tahminler:
        print("❌ Hiç tahmin dosyası bulunamadı!")
        print("   Önce ana botu çalıştırın.")
        return
    
    print(f"✅ {len(gecmis_tahminler)} tahmin dosyası bulundu.")
    print("-" * 70)
    
    tum_degerlendirmeler = []
    
    for tahmin_dosyasi in gecmis_tahminler:
        print(f"\n📁 {tahmin_dosyasi['tarih']}")
        
        for coin_data in tahmin_dosyasi['veri']:
            coin_name = coin_data['coin']
            tahmin_tarihi_str = coin_data.get('timestamp', '')
            
            try:
                tahmin_tarihi = datetime.fromisoformat(tahmin_tarihi_str.replace('Z', '+00:00'))
            except:
                continue
            
            # Tahmin anındaki fiyat (globalde değil, burada tanımla)
            tahmin_ani_fiyat = coin_data['last_price']
            
            for saat, pred in coin_data['predictions'].items():
                saat_int = int(saat.replace('h', ''))
                beklenen_fiyat = pred['expected_price']
                
                # Bugünün tarihinden eski tahminleri değerlendirme
                if tahmin_tarihi + timedelta(hours=saat_int) > datetime.now():
                    continue
                
                degerlendirme = evaluate_predictions(
                    tahmin_dosyasi['tarih'],
                    coin_name,
                    saat_int,
                    beklenen_fiyat,
                    tahmin_tarihi
                )
                
                if degerlendirme:
                    # tahmin anı fiyatını ekle
                    degerlendirme['tahmin_ani_fiyat'] = tahmin_ani_fiyat
                    tum_degerlendirmeler.append(degerlendirme)
        
        print(f"   ✅ {len([d for d in tum_degerlendirmeler if d['tahmin_kaynagi'] == tahmin_dosyasi['tarih']])} değerlendirme yapıldı")
    
    # Rapor oluştur
    if not tum_degerlendirmeler:
        print("\n❌ Henüz değerlendirilebilecek tahmin yok.")
        print("   (Tahminlerin gerçekleşmesi için zaman gerekir)")
        return
    
    df = pd.DataFrame(tum_degerlendirmeler)
    
    # Özet istatistikler
    print("\n" + "=" * 70)
    print("📊 DEĞERLENDİRME RAPORU")
    print("=" * 70)
    
    # Genel başarı oranı
    genel_basari = (df['yon_dogru'] == '✅ EVET').mean() * 100
    print(f"\n🎯 GENEL BAŞARI ORANI: %{genel_basari:.1f}")
    print(f"   Toplam değerlendirme: {len(df)}")
    
    # Saat bazında başarı
    print("\n⏰ SAAT BAZINDA BAŞARI ORANLARI:")
    saat_bazli = df.groupby('tahmin_saati').apply(
        lambda x: (x['yon_dogru'] == '✅ EVET').mean() * 100
    ).sort_index()
    
    for saat, basari in saat_bazli.items():
        print(f"   {saat:2d} saat: %{basari:.1f} ({len(df[df['tahmin_saati']==saat])} tahmin)")
    
    # Coin bazında başarı
    print("\n🪙 COIN BAZINDA BAŞARI ORANLARI:")
    coin_bazli = df.groupby('coin').apply(
        lambda x: (x['yon_dogru'] == '✅ EVET').mean() * 100
    ).sort_values(ascending=False)
    
    for coin, basari in coin_bazli.items():
        print(f"   {coin:15s}: %{basari:.1f} ({len(df[df['coin']==coin])} tahmin)")
    
    # Ortalama hata payı
    ortalama_hata = df['hata_payi_pct'].mean()
    print(f"\n📉 ORTALAMA HATA PAYI: %{ortalama_hata:.2f}")
    
    # En başarılı ve başarısız tahminler
    print("\n🏆 EN BAŞARILI 5 TAHMİN:")
    basarili = df[df['yon_dogru'] == '✅ EVET'].nlargest(5, 'gerceklesen_getiri_pct')
    for _, row in basarili.iterrows():
        print(f"   {row['coin']} {row['tahmin_saati']}h: +%{row['gerceklesen_getiri_pct']} (tahmin: %{row['beklenen_getiri_pct']})")
    
    print("\n📉 EN BAŞARISIZ 5 TAHMİN:")
    basarisiz = df[df['yon_dogru'] == '❌ HAYIR'].nlargest(5, 'hata_payi_pct')
    for _, row in basarisiz.iterrows():
        print(f"   {row['coin']} {row['tahmin_saati']}h: tahmin %{row['beklenen_getiri_pct']} → gerçek %{row['gerceklesen_getiri_pct']}")
    
    # Raporu kaydet
    output_dir = f"{VERI_KLASORU}/performans"
    os.makedirs(output_dir, exist_ok=True)
    
    # Excel raporu
    excel_path = f"{output_dir}/performans_raporu.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Tum_Degerlendirmeler', index=False)
        
        # Özet sayfası
        ozet_df = pd.DataFrame([
            {'metrik': 'Genel Başarı Oranı', 'değer': f'%{genel_basari:.1f}'},
            {'metrik': 'Toplam Değerlendirme', 'değer': len(df)},
            {'metrik': 'Ortalama Hata Payı', 'değer': f'%{ortalama_hata:.2f}'}
        ])
        ozet_df.to_excel(writer, sheet_name='Ozet', index=False)
        
        saat_df = saat_bazli.reset_index()
        saat_df.columns = ['tahmin_saati', 'başarı_orani_yuzde']
        saat_df.to_excel(writer, sheet_name='Saat_Bazli', index=False)
        
        coin_df = coin_bazli.reset_index()
        coin_df.columns = ['coin', 'başarı_orani_yuzde']
        coin_df.to_excel(writer, sheet_name='Coin_Bazli', index=False)
    
    # JSON raporu
    json_path = f"{output_dir}/performans_raporu.json"
    with open(json_path, 'w') as f:
        json.dump({
            'genel_basari_orani': round(genel_basari, 1),
            'toplam_degerlendirme': len(df),
            'ortalama_hata_payi': round(ortalama_hata, 2),
            'saat_bazli': {int(k): round(v, 1) for k, v in saat_bazli.items()},
            'coin_bazli': {k: round(v, 1) for k, v in coin_bazli.items()},
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n📁 Rapor kaydedildi:")
    print(f"   📊 Excel: {excel_path}")
    print(f"   📄 JSON: {json_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
