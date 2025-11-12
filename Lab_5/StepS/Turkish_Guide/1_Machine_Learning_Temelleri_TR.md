# Machine Learning (Makine Öğrenmesi) Temelleri

**Öğrenci:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025

---

## 🤖 Machine Learning (Makine Öğrenmesi) Nedir?

### Basit Tanım

**Machine Learning (ML)**, bilgisayarların açıkça programlanmadan **verilerden öğrenmesini** sağlayan bir yapay zeka dalıdır.

### Günlük Hayat Benzetmesi

**Normal Programlama:**

```
Siz: "Eğer sıcaklık 30°C'nin üzerindeyse, 'sıcak' yaz"
Bilgisayar: "Tamam, 30'un üzerinde her zaman 'sıcak' yazarım"
```

**Machine Learning:**

```
Siz: "İşte 1000 gün için sıcaklık ve insanların ne hissettiği verisi"
Bilgisayar: "Hmm... verilere bakıyorum... anladım! 28°C'den sonra
             insanlar genelde 'sıcak' diyor. Ama nem yüksekse
             25°C bile sıcak hissettiriyor!"
```

Fark görüyor musunuz? **Siz kuralları yazmıyorsunuz, bilgisayar kendisi öğreniyor!**

---

## 🎯 Neden Machine Learning Kullanıyoruz?

### Problem: Karmaşık İlişkiler

Diyelim ki **hava kalitesinden CO konsantrasyonunu** tahmin etmek istiyorsunuz.

**Klasik Programlama ile:**

```python
if sicaklik > 25 and nem < 50:
    CO = 2.5
elif sicaklik < 15 and nem > 70:
    CO = 1.8
elif ...  # 1000 tane daha kural!
```

❌ **Sorun:** CO'yu etkileyen o kadar çok faktör var ki (sıcaklık, nem, rüzgar, trafik, saat, mevsim...), tüm kuralları elle yazmak imkansız!

**Machine Learning ile:**

```python
# Sadece verileri ver, model kendi öğrensin!
model.fit(sicaklik_nem_verileri, CO_degerleri)
# Artık yeni verilerle tahmin yapabilir
tahmin = model.predict(yeni_veri)
```

✅ **Çözüm:** Model, verilerdeki karmaşık ilişkileri **kendisi keşfediyor**!

---

## 📚 Sklearn (Scikit-learn) Nedir?

### Kütüphane Tanımı

**Sklearn** (Scikit-learn), Python'da **makine öğrenmesi** yapmak için kullanılan **en popüler kütüphanedir**.

### Ne İşe Yarar?

Sklearn size **hazır ML araçları** sunar:

#### 1. **Modeller (Algoritmalar)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```

Bu modeller, verilerden öğrenip tahmin yapar.

#### 2. **Veri Hazırlama Araçları**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
```

Verileri modele vermeden önce hazırlar.

#### 3. **Model Değerlendirme**

```python
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

Modelinizin ne kadar iyi olduğunu ölçer.

#### 4. **Veri Ayırma**

```python
from sklearn.model_selection import train_test_split
```

Verilerinizi eğitim ve test setlerine ayırır.

### Benzetme: Sklearn = Mutfak Robotu

Yemek yapmayı düşünün:

- **Sklearn olmadan:** Her şeyi elle yaparsınız (doğrama, karıştırma, pişirme)
- **Sklearn ile:** Hazır mutfak robotu kullanırsınız (düğmeye basın, iş bitsin!)

Sklearn, ML'nin **zor kısımlarını** (matematiksel hesaplamalar, optimizasyon) **sizin için yapar**!

---

## 🎓 Machine Learning Türleri

### 1. Supervised Learning (Gözetimli Öğrenme) ⭐ BİZİM KULLANDIĞIMIZ

**Ne demek?**

- Modele hem **giriş** (input) hem de **çıkış** (output) veriyorsunuz
- Model, girişten çıkışa nasıl gidileceğini öğreniyor

**Örnek:**

```
Girdi:     Sıcaklık=25°C, Nem=60%
Çıktı:     CO = 2.3 mg/m³

Girdi:     Sıcaklık=18°C, Nem=45%
Çıktı:     CO = 1.8 mg/m³

... (7000+ örnek daha)
```

Model bu örneklere bakıp: **"Aha! Sıcaklık arttıkça CO genelde artıyor"** gibi ilişkiler öğreniyor.

**Lab 5'te:**

- Girdi: T (sıcaklık), RH (bağıl nem), AH (mutlak nem)
- Çıktı: CO(GT) konsantrasyonu
- Model, bu 3 değişkenden CO'yu tahmin etmeyi öğreniyor!

### 2. Unsupervised Learning (Gözetimsiz Öğrenme)

**Ne demek?**

- Sadece **giriş** veriyorsunuz, çıkış yok
- Model, verilerdeki **grupları/desenleri** kendisi buluyor

**Örnek:** Müşterileri benzer alışveriş alışkanlıklarına göre gruplamak

### 3. Reinforcement Learning (Pekiştirmeli Öğrenme)

**Ne demek?**

- Model deneme-yanılma ile öğreniyor
- Doğru yaptığında ödül, yanlış yaptığında ceza

**Örnek:** Oyun oynayan yapay zeka (AlphaGo, oyun botları)

---

## 🔍 Regression (Regresyon) Nedir?

### Tanım

**Regression**, supervised learning'in bir türüdür. **Sürekli bir sayı** tahmin etmeye yarar.

### Classification vs Regression

| Regression (Regresyon)                          | Classification (Sınıflandırma)        |
| ----------------------------------------------- | ---------------------------------------- |
| **Sayı** tahmin eder                     | **Kategori** tahmin eder           |
| Örnek: Ev fiyatı (250,000₺)                  | Örnek: Spam mi değil mi? (evet/hayır) |
| Örnek: Sıcaklık (23.5°C)                    | Örnek: Hayvan türü (kedi/köpek/kuş) |
| **Lab 5:** CO konsantrasyonu (2.3 mg/m³) | Örnek: Hastalık var mı? (var/yok)     |

### Lab 5'te Regression

Bizim problemimiz:

```
Girdiler:  T=25°C, RH=60%, AH=1.2 g/m³
Tahmin:    CO = 2.47 mg/m³  ← Bu bir SAYI!
```

CO konsantrasyonu **sürekli bir değer** (2.47, 1.83, 3.92...), bu yüzden **regression** kullanıyoruz!

---

## 🧮 Linear Regression (Doğrusal Regresyon)

### En Basit ML Modeli

**Linear Regression**, veriler arasında **düz çizgi** ilişkisi arar.

### Tek Değişkenli Örnek

Diyelim ki sadece **sıcaklık**tan CO tahmin ediyoruz:

```
Sıcaklık (°C)    CO (mg/m³)
10               1.5
15               1.8
20               2.1
25               2.4
30               2.7
```

**Linear Regression şunu yapar:**

1. Bu noktalara **en iyi uyan düz çizgiyi** bulur
2. Formül: **CO = a + b × Sıcaklık**
3. Örnek: **CO = 0.9 + 0.06 × Sıcaklık**

**Tahmin:**

```python
# Sıcaklık = 22°C olursa?
CO = 0.9 + 0.06 × 22 = 2.22 mg/m³
```

### Çok Değişkenli Örnek (Lab 5)

Lab 5'te **3 değişken** var: T, RH, AH

```
CO = β₀ + β₁×T + β₂×RH + β₃×AH
```

Model, **en iyi β değerlerini** (katsayıları) buluyor!

### Görsel Açıklama

**Tek değişken:**

```
CO ↑
   |        ×
   |      ×   ×
   |    ×       ×
   |  ×           ×
   |________________→ Sıcaklık

   Düz çizgi (linear)
```

**İki değişken:**

```
   CO ↑
      |    /
      |   / (düzlem)
      |  /
      |/_________→ RH
     / Sıcaklık
```

3 boyutlu bir **düzlem** bulunuyor!

---

## 🎯 Supervised Learning Süreci

### Adım Adım

#### 1. **Veri Toplama**

```python
# Örnek: 7344 satır hava kalitesi verisi
T, RH, AH, CO
25, 60, 1.2, 2.3
18, 45, 0.9, 1.8
...
```

#### 2. **Veriyi Ayırma**

```python
# %70 eğitim (training), %30 test (testing)
Eğitim: 5140 satır  # Model bunlarla öğrenecek
Test:   2204 satır  # Model bunları HİÇ görmeyecek!
```

**Neden ayırıyoruz?**

- Model eğitim verisinden öğrenir
- Ama gerçek performansı **test verisi** ile ölçeriz
- Böylece modelin **yeni verilerde** ne kadar iyi olduğunu görürüz!

#### 3. **Model Seçimi**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

#### 4. **Eğitim (Training)**

```python
model.fit(X_train, y_train)
# Model, eğitim verisindeki ilişkileri öğreniyor!
# Matematiksel olarak: en iyi β katsayılarını buluyor
```

#### 5. **Tahmin (Prediction)**

```python
tahminler = model.predict(X_test)
# Model, test verisinde tahmin yapıyor
```

#### 6. **Değerlendirme (Evaluation)**

```python
hata = mean_squared_error(y_test, tahminler)
# Tahminler ne kadar doğruydu?
```

---

## 🔬 Lab 5'te Ne Yapıyoruz?

### Özet

1. **Veri:** Hava kalitesi ölçümleri (İtalya, 7344 örnek)
2. **Girdiler:** Sıcaklık (T), Bağıl Nem (RH), Mutlak Nem (AH)
3. **Çıktı:** CO konsantrasyonu
4. **Model:** Polynomial Regression (1-10. dereceden)
5. **Amaç:** Hangi karmaşıklık seviyesi en iyi tahmin yapıyor?

### Neden Polynomial?

**Linear (Degree 1):**

```
CO = β₀ + β₁×T + β₂×RH + β₃×AH
```

Bu çok basit olabilir, gerçek ilişki daha karmaşıksa?

**Polynomial (Degree 2):**

```
CO = β₀ + β₁×T + β₂×RH + β₃×AH
     + β₄×T² + β₅×T×RH + β₆×T×AH
     + β₇×RH² + β₈×RH×AH + β₉×AH²
```

Daha karmaşık ilişkileri yakalayabilir!

**Degree 3, 4, 5...:** Daha da karmaşık!

**Soru:** Hangi karmaşıklık seviyesi en iyisi?
**Cevap:** Lab 5'in ana konusu bu! (Bias-Variance Tradeoff)

---

## 💡 Önemli Kavramlar

### Training (Eğitim)

Model, **eğitim verisinden öğrenir**.

**Analoji:** Sınava çalışmak

- Eğitim verisi = Çalıştığınız sorular
- Model öğreniyor = Konuyu kavramaya çalışıyorsunuz

### Testing (Test)

Model, **test verisinde** performans gösterir.

**Analoji:** Gerçek sınav

- Test verisi = Sınavdaki sorular (hiç görmediniz!)
- Bu, gerçek performansınızı ölçer

### Generalization (Genelleme)

Modelin **yeni, görmediği veriler**de iyi çalışması.

**Kötü örnek:** Sınav sorularını ezberleme

- Çalıştığınız soruları %100 doğru yaparsınız
- Ama sınavda farklı sorular çıkınca başarısız olursunuz
- ❌ Kötü genelleme!

**İyi örnek:** Konuyu anlama

- Çalıştığınız soruları %85 doğru yaparsınız
- Sınavda yeni sorularda da %80 doğru yaparsınız
- ✅ İyi genelleme!

---

## 🎓 Sklearn'in Avantajları

### 1. **Kolay Kullanım**

```python
# 3 satırda model eğit!
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 2. **Hızlı**

- C ve Cython ile yazılmış (çok hızlı!)
- Büyük verilerde bile çalışır

### 3. **Güvenilir**

- Dünya çapında milyonlarca kişi kullanıyor
- İyi test edilmiş algoritmalar

### 4. **Kapsamlı**

- 100+ farklı algoritma
- Preprocessing, metrics, model selection

### 5. **Dokümantasyon**

- Harika örnekler ve açıklamalar
- scikit-learn.org

---

## 📊 Özet Tablo

| Kavram                          | Açıklama                        | Lab 5'te Karşılığı        |
| ------------------------------- | --------------------------------- | ------------------------------ |
| **Machine Learning**      | Bilgisayarın veriden öğrenmesi | CO tahmin modeli               |
| **Sklearn**               | Python ML kütüphanesi           | Kullandığımız araçlar     |
| **Supervised Learning**   | Girdi ve çıktı ile öğrenme   | T,RH,AH → CO                  |
| **Regression**            | Sayı tahmini                     | CO konsantrasyonu (2.3 mg/m³) |
| **Linear Regression**     | Düz çizgi ilişkisi             | Degree 1 model                 |
| **Polynomial Regression** | Eğrisel ilişkiler               | Degree 2-10 modeller           |
| **Training**              | Modeli eğitme                    | 5140 örnekle öğrenme        |
| **Testing**               | Model performansı                | 2204 örnekle test             |
| **Generalization**        | Yeni verilerle başarı           | Test hatası düşük olmalı  |

---

## 🚀 Sonraki Adım

Şimdi **"Projede Yapay Zeka"** belgesine geçin ve bu kavramların **Lab 5'te nasıl kullanıldığını** görün!

---

**Hazırlayan:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025
