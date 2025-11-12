# Sık Sorulan Sorular - Lab 5

**Öğrenci:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025

---

## 🤔 TEMEL SORULAR

### S1: Sklearn tam olarak nedir?

**C:** Sklearn (Scikit-learn), Python'da machine learning yapmak için kullanılan bir kütüphanedir. Hazır algoritmalar, veri işleme araçları ve model değerlendirme fonksiyonları sunar.

**Benzetme:** Excel gibi! Excel'de hazır formüller var (SUM, AVERAGE). Sklearn'de de hazır ML algoritmaları var (LinearRegression, PolynomialFeatures).

---

### S2: Model nasıl "öğreniyor"?

**C:** Model, **en iyi parametreleri (β katsayılarını) bulmaya** çalışıyor.

**Adımlar:**
1. Random β değerleriyle başla
2. Tahminler yap
3. Hataları hesapla
4. β'ları değiştir, hatayı azalt
5. En düşük hata bulunana kadar tekrarla!

Bu süreç `model.fit()` içinde otomatik oluyor!

---

### S3: Training ve testing neden ayrı?

**C:** **Ezberle vs. Öğrenme** ayrımı için!

**Analoji:**
- Training = Ders çalışma
- Testing = Gerçek sınav (hiç görmedin!)

Eğer tüm veriyi eğitimde kullansak, model ezberlemiş olabilir. Test verisi, **yeni verilerde ne kadar iyi olduğunu** gösterir.

---

### S4: Polynomial nedir? Neden kullanıyoruz?

**C:** Polynomial, **eğrisel ilişkileri** yakalamamızı sağlıyor.

**Linear (degree 1):**
```
CO = β₀ + β₁×T
```
Sadece düz çizgi!

**Polynomial (degree 2):**
```
CO = β₀ + β₁×T + β₂×T²
```
Eğri çizebiliyor!

**Neden?** Gerçek hayatta ilişkiler genelde düz değil, eğrisel!

---

## 📊 KAVRAMSAL SORULAR

### S5: Bias nedir?

**C:** Model'in **sistematik hataları**.

**Örnek:**
- Gerçek ilişki eğrisel
- Model düz çizgi çiziyor
- Sonuç: Her zaman biraz yanılıyor → Yüksek bias!

**Düşük bias:** Model gerçek ilişkiye yakın
**Yüksek bias:** Model çok basit, ilişkiyi yakalayamıyor

---

### S6: Variance nedir?

**C:** Model'in **farklı verilerle ne kadar değiştiği**.

**Örnek:**
- Eğitim seti 1: Model A öğreniyor
- Eğitim seti 2: Tamamen farklı Model B öğreniyor
- Sonuç: Çok değişken → Yüksek variance!

**Düşük variance:** Model tutarlı
**Yüksek variance:** Model çok hassas, her veriye farklı uyuyor

---

### S7: Underfitting nedir?

**C:** Model **çok basit**, önemli ilişkileri yakalayamıyor.

**Belirtiler:**
- Hem eğitim hem test hatası **yüksek**
- Model deseni görmüyor
- Degree çok düşük

**Çözüm:** Model kompleksitesini artır!

---

### S8: Overfitting nedir?

**C:** Model **çok karmaşık**, eğitim verisini ezberliyor.

**Belirtiler:**
- Eğitim hatası çok **düşük**
- Test hatası **yüksek**
- **Büyük gap** var!
- Degree çok yüksek

**Çözüm:** Model kompleksitesini azalt veya daha fazla veri topla!

---

### S9: Bias-variance tradeoff nedir?

**C:** Bias ve variance arasında **ödünleşim** var!

```
Bias ↓ (azalır) → Variance ↑ (artar)
Variance ↓ (azalır) → Bias ↑ (artar)
```

İkisini de aynı anda azaltamazsınız! **Optimal nokta**, ikisinin de dengeli olduğu yerdir.

---

## 💻 KOD SORULARI

### S10: `model.fit()` ne yapıyor?

**C:** Model'i **eğitiyor** (training).

```python
model.fit(X_train, y_train)
```

**Ne oluyor:**
1. X_train'deki özelliklere bakıyor
2. y_train'deki çıktılara bakıyor
3. En iyi β katsayılarını buluyor
4. Formülü öğreniyor: `CO = β₀ + β₁×T + ...`

**Sonuç:** Model artık tahmin yapabilir!

---

### S11: `model.predict()` ne yapıyor?

**C:** Model öğrendiği formülle **tahmin yapıyor**.

```python
predictions = model.predict(X_test)
```

**Ne oluyor:**
1. X_test'teki her satır için
2. Öğrendiği formülü uygula
3. CO değerini hesapla
4. Tahminleri döndür

**Örnek:**
```
Input: T=20, RH=50, AH=1.0
Model: CO = 0.5 + 0.03×20 + 0.01×50 + 0.8×1.0
Output: CO = 2.4 mg/m³
```

---

### S12: `PolynomialFeatures()` ne yapıyor?

**C:** Özellikleri **polynomial terimlere dönüştürüyor**.

**Degree 1:**
```python
Input:  [T, RH, AH]
Output: [T, RH, AH]
# Değişiklik yok
```

**Degree 2:**
```python
Input:  [T, RH, AH]
Output: [T, RH, AH, T², T×RH, T×AH, RH², RH×AH, AH²]
# 3 terim → 9 terim!
```

**Neden?** Linear regression'a eğrileri öğretmek için!

---

### S13: `mean_squared_error()` ne yapıyor?

**C:** Tahminlerin **ne kadar yanlış olduğunu** ölçüyor.

**Formül:**
```
MSE = (hata₁² + hata₂² + ... + hataₙ²) / n

hata = gerçek - tahmin
```

**Örnek:**
```
Gerçek: [2.0, 3.0, 2.5]
Tahmin: [2.1, 2.8, 2.6]
Hata:   [0.1, -0.2, 0.1]
MSE = (0.01 + 0.04 + 0.01) / 3 = 0.02
```

**Düşük MSE = İyi model!**

---

## 📈 GRAFİK SORULARI

### S14: Validation curve nasıl okunur?

**C:** İki çizgiye bakın:

**Yeşil (Training Error):**
- Sürekli azalıyor
- ❌ Buna bakmayın! Yanıltıcı!

**Kırmızı (Test Error):**
- U-şekli yapıyor
- ✅ Buna bakın! Bu gerçek performans!
- **En düşük nokta = Optimal model!**

---

### S15: U-şekli neden oluşuyor?

**C:** Bias-variance tradeoff!

```
Sol taraf (Degree 1-2):
  Yüksek bias → Hata yüksek → Underfitting

Orta (Degree 5-9):
  Dengeli → Hata düşük → Optimal!

Sağ taraf (Degree 10):
  Yüksek variance → Hata artıyor → Overfitting
```

---

### S16: Gap nedir?

**C:** Test hatası ile eğitim hatası arasındaki fark.

```python
gap = test_error - train_error
```

**Küçük gap (< 0.05):** Good!
**Orta gap (0.05-0.10):** Dikkat!
**Büyük gap (> 0.10):** Overfitting!

---

## 🎯 LAB 5 SPESİFİK SORULAR

### S17: Neden 3 özellik kullanıyoruz?

**C:** Lab'ın amacına uygun!

**Kullandığımız:**
- T (Sıcaklık)
- RH (Bağıl nem)
- AH (Mutlak nem)

**Neden sadece bunlar?**
- Lab, **bias-variance** öğretmeye odaklanıyor
- Çok özellik olsa, çok karmaşık olurdu
- Basit tutmak pedagojik amaçlı!

**Gerçek hayatta:** Daha çok özellik kullanılır (rüzgar, trafik, saat, vs.)

---

### S18: R² neden çok düşük (%4)?

**C:** Çünkü **sadece 3 meteorolojik özellik** kullanıyoruz!

**R² = 0.04 demek:**
- Model, CO varyansının sadece %4'ünü açıklayabiliyor
- %96 başka faktörlerden kaynaklanıyor

**Hangi faktörler eksik?**
- Trafik yoğunluğu (en önemli!)
- Rüzgar hızı ve yönü
- Saat (sabah trafiği)
- Mevsim
- Emisyon kaynakları

**Lab'ın amacı:** Yüksek R² değil, bias-variance'ı anlamak!

---

### S19: Neden degree 9 optimal, 10 değil?

**C:** Degree 10'da **overfitting başlıyor**.

**Kanıt:**
```
Degree 9:
  Train: 1.95
  Test:  1.98
  Gap:   0.03

Degree 10:
  Train: 1.94 (daha düşük!)
  Test:  1.99 (arttı!)
  Gap:   0.05 (büyüdü!)
```

Degree 10, eğitim verisine daha iyi uyuyor ama test'te kötüleşiyor → Overfitting!

---

### S20: Cross-validation neden degree 1 öneriyor?

**C:** Çünkü CV **daha güvenilir ve stabil** modeli seçiyor!

**Single split:**
- Tek bir rastgele ayırma
- Şanslı/şanssız olabilir
- Degree 9 en düşük test hatası

**Cross-validation:**
- 5 farklı ayırmayı test ediyor
- Daha güvenilir
- Degree 1 en **tutarlı** (standart sapma düşük)

**Hangisi doğru?**
- Degree 9: En düşük hata, ama varyans yüksek
- Degree 1: Biraz daha yüksek hata, ama çok tutarlı

**Gerçek projede:** Degree 1 daha güvenli seçim olurdu!

---

## 🐛 PROBLEM ÇÖZME

### S21: "ModuleNotFoundError: No module named 'sklearn'"

**C:** Python versiyonu yanlış veya sklearn kurulu değil!

**Çözüm 1:** Doğru Python kullan
```bash
/Users/alikaratas/miniconda3/bin/python3 code/run_lab5.py
```

**Çözüm 2:** Conda aktive et
```bash
conda activate base
python3 code/run_lab5.py
```

**Çözüm 3:** Sklearn kur
```bash
pip install scikit-learn
```

---

### S22: "FileNotFoundError: AirQualityUCI.csv"

**C:** CSV dosyası doğru yerde değil veya yanlış dizindesiniz!

**Çözüm:** Doğru dizine gidin
```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/run_lab5.py
```

Veya script'i güncelleyin:
```python
df = pd.read_csv('dataset/AirQualityUCI.csv', ...)
```

---

### S23: Jupyter notebook çalışmıyor!

**C:** Jupyter kurulu değil veya Python versiyonu yanlış!

**Çözüm:**
```bash
# Jupyter kur
pip install jupyter

# Notebook aç
jupyter notebook Lab5_BiasVariance.ipynb

# Sonra: Kernel → Restart & Run All
```

---

## 💡 İLERİ SEVIYE SORULAR

### S24: Gradient Descent nedir?

**C:** Modelin parametreleri (β) **optimize etme yöntemi**.

**Benzetme:** Karanlıkta dağdan inme
1. Rastgele bir yerden başla
2. Hangi yön aşağı? (Gradient hesapla)
3. O yöne küçük adım at
4. Tekrarla → En alçak noktayı bul!

**Lab 5'te:** Sklearn otomatik yapıyor (Normal Equation kullanıyor).

---

### S25: Regularization nedir?

**C:** Overfitting'i önlemek için **model kompleksitesine ceza** verme!

**Ridge Regression:**
```
Minimize et: MSE + λ×(β₁² + β₂² + ... + βₙ²)
```

λ büyük → Katsayılar küçük → Basit model → Az overfitting

**Lab 5'te:** Kullanmıyoruz, ama bonus olarak eklenebilir!

---

### S26: Feature scaling gerekli mi?

**C:** Bazı algoritmalarda evet, Linear Regression'da **gerekli değil**!

**Gerekli olan algoritmalar:**
- K-Nearest Neighbors
- Support Vector Machines
- Neural Networks

**Gerekli olmayan:**
- Linear Regression
- Decision Trees

**Lab 5'te:** Yapmıyoruz çünkü gerekli değil!

---

## 📚 KAVRAM KARŞILAŞTIRMA

### S27: MSE vs RMSE vs R² - Hangisi daha iyi?

**C:** Hepsi farklı bilgi verir!

**MSE (Mean Squared Error):**
- Matematiksel olarak işlem yapmaya uygun
- Birim: (mg/m³)²
- Yorumlaması zor

**RMSE (Root MSE):**
- MSE'nin karekökü
- Birim: mg/m³ (hedefle aynı!)
- Yorumlaması kolay: "Ortalama hata 1.4 mg/m³"

**R² (Coefficient of Determination):**
- Açıklanan varyans yüzdesi
- 0-1 arası (1 = mükemmel)
- Model'in genel başarısını gösterir

**Hangisini kullanmalı:**
- Optimizasyon için: MSE
- Yorumlama için: RMSE
- Genel başarı için: R²

---

### S28: Training error vs Validation error vs Test error?

**C:** Farklı aşamalarda kullanılıyor!

**Training Error:**
- Model'in eğitim verisindeki hatası
- Her zaman yanıltıcı!
- Düşük olması iyi model anlamına gelmez

**Validation Error:**
- Model seçimi için kullanılan hata
- Cross-validation'da hesaplanır
- En iyi hiperparametreyi bulmak için

**Test Error:**
- Final performans ölçümü
- En son, tek sefer hesaplanır
- Gerçek performansı gösterir

**Lab 5'te:**
- Training error hesaplıyoruz (karşılaştırma için)
- Test error kullanıyoruz (model seçimi için)
- Validation: Cross-validation bonus'ta var

---

## 🎓 SINAV/SUNUM SORULARI

### S29: Bias-variance tradeoff'u nasıl açıklarım?

**C:** 3 adımda:

**1. Sorun:**
"Model ne kadar karmaşık olmalı?"

**2. İki uç:**
- Çok basit → Yüksek bias → Underfitting
- Çok karmaşık → Yüksek variance → Overfitting

**3. Çözüm:**
"Optimal model, test hatasının en düşük olduğu dengede!"

**Grafik göster:** U-shaped curve!

---

### S30: Lab 5'in ana mesajı nedir?

**C:**

> **"Daha karmaşık her zaman daha iyi değildir!
> En iyi model, yeni verilerde en iyi performans gösterendir!"**

**Kanıt:** Lab 5'te:
- Degree 10: En düşük eğitim hatası
- Degree 9: En düşük test hatası → Bu daha önemli!

---

## 🚀 SON TAVSİYELER

### Sınav İçin
1. Bias-variance kavramlarını **örneklerle** açıklayabilin
2. U-shaped curve'ü **çizebilir ve yorumlayabilin**
3. Training vs test error **farkını** anlayın
4. Overfitting'in **nasıl tespit edileceğini** bilin

### Proje İçin
1. Kodu **satır satır** anlayın
2. Grafikleri **yorumlayabilin**
3. Sonuçları **açıklayabilin**
4. Alternatif çözümler **önerebilir**

### Kariyer İçin
1. Bu konseptler **her ML projesinde** kullanılır!
2. Interview'larda **sıkça sorulur**
3. Gerçek problemlerde **kritik önem** taşır!

---

**Hazırlayan:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025
**Durum:** ✅ 30 Soru Cevaplandı!
