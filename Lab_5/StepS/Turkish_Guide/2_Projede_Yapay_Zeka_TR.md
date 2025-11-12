# Lab 5'te Yapay Zeka Nasıl Kullanılıyor?

**Öğrenci:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025

---

## 🎯 Bu Dokümanda Ne Öğreneceksiniz?

1. Lab 5'te **tam olarak ne yapıyoruz**?
2. **Yapay zeka** kodun neresinde?
3. Model **nasıl öğreniyor**?
4. **Training ve Testing** neden ayrı?
5. Adım adım **ne oluyor**?

---

## 🌍 Projenin Genel Resmi

### Problem Tanımı

**Soru:** İtalya'daki bir hava kalitesi istasyonundan **CO (Karbon Monoksit) konsantrasyonunu** tahmin edebilir miyiz?

**Elimizde ne var?**
- 📊 9,471 saatlik ölçüm verisi
- 🌡️ Sıcaklık (T)
- 💧 Bağıl Nem (RH - Relative Humidity)
- 💦 Mutlak Nem (AH - Absolute Humidity)
- 🏭 CO konsantrasyonu (tahmin etmek istediğimiz)

**Ne istiyoruz?**
Yeni bir ölçümde (örneğin: T=23°C, RH=55%, AH=1.1 g/m³), CO'nun ne olacağını **tahmin etmek**!

---

## 🤖 Yapay Zeka Nerede Kullanılıyor?

### Kodun AI Kısmı

```python
# 1. MODEL OLUŞTURMA (AI burada başlıyor!)
from sklearn.linear_model import LinearRegression
model = LinearRegression()  # ← Yapay zeka modeli!

# 2. MODEL EĞİTİMİ (AI öğreniyor!)
model.fit(X_train, y_train)  # ← Burası sihir!

# 3. TAHMİN (AI bilgisini kullanıyor!)
predictions = model.predict(X_test)  # ← AI tahmin yapıyor!
```

**Açıklama:**
- `LinearRegression()`: Yapay zeka **algoritması** (beyin)
- `fit()`: Model **öğreniyor** (eğitim)
- `predict()`: Model **tahmin yapıyor** (kullanım)

---

## 📚 Lab 5'in Hikayesi (Adım Adım)

### Senaryo: Hava Kalitesi Tahmini

Hayal edin: İtalya'da bir çevre mühendisi çalışıyorsunuz.

#### **Bölüm 1: Veri Toplama** 📊

```python
# İtalyan istasyonundan veri geldi!
df = pd.read_csv('AirQualityUCI.csv')
# 9,471 saatlik ölçüm!
```

**Veri nasıl görünüyor?**
```
Tarih        Saat      T     RH    AH    CO(GT)
10/03/2004  18:00:00  13.6  48.9  0.76   2.6
10/03/2004  19:00:00  13.3  47.7  0.73   2.0
10/03/2004  20:00:00  11.9  54.0  0.75   2.2
...
```

Her satır = 1 saatlik ölçüm
- T = Sıcaklık (°C)
- RH = Bağıl nem (%)
- AH = Mutlak nem (g/m³)
- CO(GT) = Gerçek CO değeri (mg/m³)

#### **Bölüm 2: Veri Temizleme** 🧹

```python
# Sorunlu değerleri temizle
df_clean = df.replace(-200.0, np.nan)  # -200 = kayıp veri
data_cleaned = data.dropna()  # Kayıp olanları çıkar
```

**Sonuç:** 7,344 temiz veri satırı kaldı!

**Neden temizleme?**
- Sensör bazen arızalanıyor (-200 gönderiyor)
- AI, bozuk veriyle öğrenemez!

#### **Bölüm 3: Veriyi Ayırma** ✂️

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

**Ne oluyor?**
```
Toplam: 7,344 veri
    ↓
    ├─ Eğitim (70%): 5,140 veri  ← Model bunlarla öğrenecek
    └─ Test (30%):   2,204 veri  ← Model bunları GÖRMEYECEK!
```

**Analoji - Sınav Hazırlığı:**

**Eğitim Verisi = Ders Çalışma**
```
Öğrenci: "Şu örnekleri çöz, öğren"
         (5,140 örnek soru)
```

**Test Verisi = Gerçek Sınav**
```
Öğretmen: "Hiç görmediğin soruları çöz!"
          (2,204 yeni soru)
```

**Neden ayırıyoruz?**
- Ezberle

me vs Öğrenme ayrımı!
- Modelin **yeni verilerde** ne kadar iyi olduğunu görmek için!

#### **Bölüm 4: Model Eğitimi** 🎓

##### Degree 1 (Linear - En Basit)

```python
# Degree 1: Düz çizgi ilişkisi
poly = PolynomialFeatures(degree=1)
X_train_poly = poly.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_poly, y_train)  # ← ÖĞRENME BURASI!
```

**Model ne öğreniyor?**
```
CO = β₀ + β₁×T + β₂×RH + β₃×AH

Örnek:
CO = 0.5 + 0.03×T + 0.01×RH + 0.8×AH
```

Model, **en iyi β değerlerini** buluyor!

**Nasıl öğreniyor? (Basitleştirilmiş)**
```
1. Random β değerleriyle başla
   CO_tahmin = 0.2 + 0.05×T + 0.02×RH + 0.5×AH

2. Tahminleri yap
   Gerçek CO: 2.6  →  Tahmin: 2.1  →  Hata: 0.5
   Gerçek CO: 2.0  →  Tahmin: 1.8  →  Hata: 0.2
   ...

3. Toplam hatayı hesapla
   MSE = (0.5² + 0.2² + ...) / 5140 = 2.04

4. β'ları değiştir, hatayı azalt!
   (Gradient Descent algoritması)

5. En düşük hata bulunana kadar tekrarla!
```

**Sonuç:** Model, **optimal β değerlerini** buldu!

##### Degree 2 (Quadratic - Daha Karmaşık)

```python
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
```

**Şimdi ne öğreniyor?**
```
CO = β₀ + β₁×T + β₂×RH + β₃×AH
     + β₄×T² + β₅×T×RH + β₆×T×AH
     + β₇×RH² + β₈×RH×AH + β₉×AH²
```

**Fark:**
- Degree 1: 3 terim (T, RH, AH)
- Degree 2: 9 terim (artı kareler ve çarpımlar!)
- Daha karmaşık ilişkileri yakalayabilir!

##### Degree 3, 4, 5... 10

Her degree'de daha da karmaşık!
- Degree 3: 19 terim
- Degree 10: 364 terim!

**Soru:** Hangisi en iyi?
**Cevap:** Lab 5'in ana konusu! (bias-variance tradeoff)

#### **Bölüm 5: Tahmin Yapma** 🔮

```python
# Test verisinde tahmin yap
y_test_pred = model.predict(X_test_poly)
```

**Ne oluyor?**
```
Test Verisi (model HİÇ görmedi!):
   T=15°C, RH=52%, AH=0.9 g/m³

Model: "Hmm... öğrendiğim formülle hesaplayayım..."
   CO = 0.5 + 0.03×15 + 0.01×52 + 0.8×0.9
   CO = 0.5 + 0.45 + 0.52 + 0.72
   CO = 2.19 mg/m³

Gerçek değer: 2.3 mg/m³
Hata: |2.19 - 2.3| = 0.11 mg/m³
```

Model, 2,204 test örneği için bunu yapıyor!

#### **Bölüm 6: Değerlendirme** 📊

```python
# Hatayı hesapla
test_mse = mean_squared_error(y_test, y_test_pred)
```

**MSE (Mean Squared Error) nedir?**

```
MSE = (hata₁² + hata₂² + ... + hata₂₂₀₄²) / 2204

Örnek:
Hata₁ = 0.11  →  0.11² = 0.0121
Hata₂ = -0.25 →  0.25² = 0.0625
Hata₃ = 0.08  →  0.08² = 0.0064
...

MSE = (0.0121 + 0.0625 + 0.0064 + ...) / 2204 = 1.98
```

**Düşük MSE = İyi model!**

---

## 🔄 Training vs Testing - Derinlemesine

### Neden İkiye Ayırıyoruz?

#### Senaryo 1: Tüm Veriyle Eğitsek Ne Olur? ❌

```python
# YANLIŞ: Tüm veriyi eğitimde kullanalım
model.fit(tüm_veri, tüm_CO_değerleri)

# Test edelim
hata = model.predict(tüm_veri)
print(f"Hata: {hata}")  # Çok düşük!
```

**Sorun:** Model **gördüğü veride** test ediliyor!
- Ezberlemis olabilir!
- **Yeni verilerle** ne olacak bilmiyoruz!

**Analoji:**
```
Öğrenci: Sınav sorularını önceden görmüş
Sınavda: %100 yapıyor (ama ezberlemiş!)
Yeni konuda: Hiçbir şey bilmiyor!
```

#### Senaryo 2: Train-Test Split ✅

```python
# DOĞRU: Veriyi ayır
Eğitim: 5,140 veri  # Model bunlarla öğrenir
Test:   2,204 veri  # Model HİÇ görmez!

model.fit(eğitim_verisi)
hata = model.predict(test_verisi)
```

**Avantaj:** **Gerçek performans** görüyoruz!
- Model test verisini HİÇ görmedi
- Yeni verilerdeki başarıyı simüle ediyor!

**Analoji:**
```
Öğrenci: Sadece ders kitabındaki örnekleri çalıştı
Sınavda: Farklı sorular (ama aynı konudan)
Başarı: Gerçek öğrenmeyi gösteriyor!
```

---

## 📈 Lab 5'te Ne Buluyoruz?

### 10 Farklı Model

```python
for degree in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    # Her degree için model eğit
    model = train_polynomial_model(degree)

    # Eğitim hatası
    train_error = hesapla(eğitim_verisi)

    # Test hatası
    test_error = hesapla(test_verisi)
```

### Sonuçlar

| Degree | Eğitim Hatası | Test Hatası | Yorum |
|--------|---------------|-------------|-------|
| 1      | 2.04          | 2.06        | İkisi de yüksek (çok basit!) |
| 2      | 2.02          | 2.03        | Biraz düştü |
| ...    | ...           | ...         | ... |
| 9      | 1.95          | 1.98        | Test en düşük! ⭐ |
| 10     | 1.94          | 1.99        | Eğitim düşük ama test arttı! |

**Bulgu:**
- **Degree 9**: En iyi genelleme!
- **Degree 10**: Eğitim çok iyi, test kötü → Overfitting!

---

## 🎯 Yapay Zeka İş Başında: Gerçek Örnek

### Adım Adım Bir Tahmin

```python
# 1. YENİ BİR VERİ GELDİ
yeni_olcum = {
    'T': 22.5,   # Sıcaklık
    'RH': 58.0,  # Bağıl nem
    'AH': 1.15   # Mutlak nem
}

# 2. MODELİ HAZIRLA (Degree 9 - en iyisi!)
poly = PolynomialFeatures(degree=9)
X_new_poly = poly.transform([[22.5, 58.0, 1.15]])
# [22.5, 58.0, 1.15, 22.5², 22.5×58, ..., 1.15⁹]

# 3. TAHMİN YAP
CO_tahmini = model.predict(X_new_poly)
print(f"Tahmini CO: {CO_tahmini[0]:.2f} mg/m³")
# Çıktı: Tahmini CO: 2.34 mg/m³
```

**Model ne yaptı?**
```
1. Öğrendiği formülü kullandı:
   CO = β₀ + β₁×22.5 + β₂×58.0 + β₃×1.15 + ...
        + β₃₆₄×(1.15⁹)

2. 364 terimi hesapladı

3. Sonuç: 2.34 mg/m³
```

**Bu bir AI tahmini!**
- Model, 5,140 örnekten **öğrendi**
- Hiç görmediği yeni veriye **uyguladı**
- İnsan müdahalesi olmadan **karar verdi**!

---

## 🧠 Model Nasıl Öğreniyor? (Detaylı)

### Linear Regression'ın Matematiği

**Amaç:** En iyi β değerlerini bul!

```
Minimize et: MSE = Σ(gerçek - tahmin)² / n

Tahmin = β₀ + β₁×T + β₂×RH + β₃×AH
```

**Çözüm Yöntemi: Normal Equation**
```
β = (XᵀX)⁻¹Xᵀy
```

Sklearn bunu sizin için yapıyor!

### Gradient Descent (Alternatif)

**Benzetme:** Karanlıkta dağdan inme

```
1. Rastgele bir noktadan başla (random β)
2. Hangi yön aşağı? (gradient hesapla)
3. O yöne küçük adım at
4. Tekrarla → En alçak noktayı bul!
```

**Lab 5'te:**
- Sklearn otomatik yapıyor
- Normal Equation kullanıyor (küçük veriler için hızlı)

---

## 📊 Görselleştirme

### Model Öğrenmeden Önce

```
CO ↑ Gerçek veriler
   |    ×
   |      ×  ×
   |  ×     ×   ×
   |   ×       ×
   |________________→ Sıcaklık

Model: "Hmm... ne yapacağımı bilmiyorum"
```

### Model Öğrendikten Sonra (Degree 1)

```
CO ↑
   |    ×
   |      ×/ ×  ← Tahmin çizgisi
   |  × /   ×   ×
   |   /×      ×
   |/_____________→ Sıcaklık

Model: "Düz bir çizgi çizdim, yaklaşık doğru!"
```

### Degree 9 (En İyi)

```
CO ↑
   |    ×
   |   ~~~×~~×  ← Daha iyi uyum
   |  × ~~  × ~~×
   |   ~~×    ~~×
   |/_____________→ Sıcaklık

Model: "Eğriyi takip ediyorum, daha iyi!"
```

### Degree 10 (Overfitting)

```
CO ↑
   |    ×
   | /\/\×/\×  ← Çok karmaşık!
   |×/\/  × /\×
   | \/×  \/  ×
   |/_____________→ Sıcaklık

Model: "Her noktaya dokun... ama test verisinde kötü!"
```

---

## 🎯 Özet: AI'nın Lab 5'teki Rolü

### 1. **Öğrenme** (Training)
```python
model.fit(X_train, y_train)
# 5,140 örnekten ilişkileri öğren!
```

### 2. **Tahmin** (Prediction)
```python
predictions = model.predict(X_test)
# Yeni 2,204 örneği tahmin et!
```

### 3. **Değerlendirme** (Evaluation)
```python
mse = mean_squared_error(y_test, predictions)
# Ne kadar iyiyim?
```

### 4. **Optimizasyon** (Model Selection)
```
Degree 1, 2, 3... 10 dene
En iyi test performansını bul!
```

---

## 💡 Neden Bu AI/ML Sayılıyor?

### Klasik Programlama

```python
def predict_CO(T, RH, AH):
    if T > 25 and RH < 50:
        return 2.5
    elif T < 15 and RH > 70:
        return 1.8
    # ... 1000 tane kural!
```

❌ **Sorun:** Kuralları elle yazmanız gerekiyor!

### Machine Learning (Lab 5)

```python
model.fit(veriler, CO_değerleri)
prediction = model.predict(yeni_veri)
```

✅ **Çözüm:** Model **kendisi öğreniyor**!

**Fark:**
- **Klasik:** Siz kuralları yazıyorsunuz
- **ML:** Model kuralları **kendisi keşfediyor**!

---

## 🚀 Sonraki Adım

Şimdi **"Kod Açıklamaları"** belgesine geçin ve her satırın **tam olarak ne yaptığını** görün!

---

**Hazırlayan:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025
