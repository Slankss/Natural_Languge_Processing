import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Excel Dosyalarından Haberleri Okuyun
dosya_yolları = ["medicine_news.xlsx", "technical_news.xlsx"]
haber_verileri = []

for dosya_yolu in dosya_yolları:
    haber_df = pd.read_excel(dosya_yolu)
    haber_verileri.append(haber_df)

# Verileri Birleştirin
tüm_haberler = pd.concat(haber_verileri, ignore_index=True)

# Haber Metnini ve Kategorileri Ayırın
haber_metinleri = tüm_haberler["News"].tolist()
kategoriler = tüm_haberler["kategori"].tolist()

# Metni Ön İşleme (gerçek temizleme işlevleriyle değiştirin)
def metni_ön_işle(metin):
    # Noktalama işaretlerini, durdurma kelimelerini vb. kaldırma gibi temizleme adımlarını uygulayın.
    # Küçük harfe çevir
    metin = metin.lower()
    return metin

ön_işlenmiş_haberler = [metni_ön_işle(haber) for haber in haber_metinleri]

# TF-IDF Vektörizasyonu
vektörleştirici = TfidfVectorizer()
özellikler = vektörleştirici.fit_transform(ön_işlenmiş_haberler)

# Tren-Test Bölme
X_egitim, X_test, y_egitim, y_test = train_test_split(özellikler, kategoriler, test_boyutu=0.1)

# Naive Bayes Model Eğitimi
model = MultinomialNB()
model.fit(X_egitim, y_egitim)

# Görünmeyen Haberler Üzerinde Tahmin (gerçek görünmeyen haberlerle değiştirin)
görünmeyen_haber = ["Yeni bir yapay zeka algoritması geliştirildi..."]
görünmeyen_haber_df = pd.DataFrame({"haber_metni": [görünmeyen_haber]})
görünmeyen_haber_df["ön_işlenmiş_metin"] = görünmeyen_haber_df["haber_metni"].apply(metni_ön_işle)
görünmeyen_haber_özellikleri = vektörleştirici.transform(görünmeyen_haber_df["ön_işlenmiş_metin"])
tahminler = model.predict(görünmeyen_haber_özellikleri)

# Değerlendirme
doğruluk = accuracy_score(y_test, model.predict(X_test))
f1 = f1_score(y_test, model.predict(X_test), average='weighted')

print("Doğruluk:", doğruluk)
print("F1 Skoru:", f1)

# Görünmeyen haberler için tahmini yazdırın
print("Görünmeyen haberler için tahmin edilen kategori:", tahminler[0])