# Bias-Variance Tradeoff (Ödünleşim) - Lab 5'in Ana Konusu

**Öğrenci:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025

---

## 🎯 Bu Lab'ın ASIL Konusu Budur!

Bias-Variance Tradeoff, **makine öğrenmesinin en temel konseptlerinden biri**dir. Lab 5'in tüm amacı bunu anlamaktır!

---

## 🤔 Basit Soru: Model Ne Kadar Karmaşık Olmalı?

### Seçenekler

**Seçenek 1:** Çok basit model (degree 1)
**Seçenek 2:** Orta karmaşıklık (degree 5)
**Seçenek 3:** Çok karmaşık (degree 10)

**Hangisi en iyi?** İşte bu sorunun cevabı **Bias-Variance Tradeoff**!

---

## 📚 Üç Durum: Goldilocks Hikayesi

### 1. Underfitting (Çok Basit) - "Çok Soğuk" 🥶

**Ne oluyor?**
- Model **çok basit**
- Verilerdeki ilişkileri **yakalayamıyor**
- Hem eğitimde hem testte **başarısız**

**Lab 5'te:**
```
Degree 1 (Linear):
CO = β₀ + β₁×T + β₂×RH + β₃×AH

Eğitim Hatası: 2.04 (yüksek!)
Test Hatası:    2.06 (yüksek!)
```

**Analoji - Sınav:**
```
Öğrenci: Konuyu hiç anlamadı
Ders çalışırken: %60 doğru
Sınavda: %58 doğru

Her iki skorda da kötü → Underfitting!
```

**Görsel:**
```
CO ↑
   |    ×
   |      ×  ×    ← Noktalar (gerçek veri)
   |  × /    ×  ×
   |   /×      ×   ← Düz çizgi (model)
   |/_____________→ T

Model çok basit! Eğriyi yakalayamıyor!
```

### 2. Just Right (Tam Olmalı) - "Tam Kararında" ✨

**Ne oluyor?**
- Model **dengeli**
- Önemli ilişkileri **yakalıyor**
- Gürültüye takılmıyor
- **En iyi genelleme**!

**Lab 5'te:**
```
Degree 9 (Optimal):
Daha karmaşık formül...

Eğitim Hatası: 1.95 (düşük!)
Test Hatası:    1.98 (en düşük!)
Gap:            0.03 (küçük!)
```

**Analoji - Sınav:**
```
Öğrenci: Konuyu iyi öğrendi
Ders çalışırken: %85 doğru
Sınavda: %82 doğru

Her ikisi de iyi, yakın skor → Good fit!
```

**Görsel:**
```
CO ↑
   |    ×
   |   ~~~×~~    ← Noktalar
   |  ×~~  ×~~×
   |  ~~×   ~~×  ← Eğri (model)
   |/_____________→ T

Model deseni yakalıyor, gürültüyü değil!
```

### 3. Overfitting (Çok Karmaşık) - "Çok Sıcak" 🔥

**Ne oluyor?**
- Model **çok karmaşık**
- Eğitim verisini **ezberliyor**
- Gürültüyü bile öğreniyor!
- Yeni verilerle **başarısız**

**Lab 5'te:**
```
Degree 10 (Overfit):
CO = β₀ + ... + 364 terim!

Eğitim Hatası: 1.94 (çok düşük!)
Test Hatası:    1.99 (arttı!)
Gap:            0.05 (büyüyor!)
```

**Analoji - Sınav:**
```
Öğrenci: Soruları ezberledi
Ders çalışırken: %100 doğru (ezber!)
Sınavda: %65 doğru (farklı sorularda kötü!)

Eğitim mükemmel, test kötü → Overfitting!
```

**Görsel:**
```
CO ↑
   |    ×
   | /\/\×/\    ← Noktalar
   |×/\/ ×/\×
   |\/×  \/  ×  ← Aşırı karmaşık eğri
   |/_____________→ T

Model her noktayı geçiyor!
Ama gürültüyü de öğrendi, genelleyemiyor!
```

---

## 📊 Lab 5'in Ana Grafiği: U-Shaped Curve

### Validation Curve

```
Hata ↑
    |
    |  Test Hatası (Kırmızı)
    |    \___________/
    |
    |     \___________  Eğitim Hatası (Yeşil)
    |
    +_____________________→ Karmaşıklık (Degree)
      1  2  3  4  5  6  7  8  9  10

    Underfitting  Optimal  Overfitting
```

### Ne Görüyoruz?

**Eğitim Hatası (Yeşil):**
- Sürekli azalıyor ↓
- Degree arttıkça model eğitim verisine daha iyi uyuyor
- ❌ **Yanıltıcı!** Bu iyi model anlamına gelmez!

**Test Hatası (Kırmızı):**
- U-şekli yapıyor!
- Önce azalıyor ↓ (iyileşiyor)
- Sonra artıyor ↑ (overfitting başlıyor)
- ✅ **Doğru metric!** Buna bakmamız gerekiyor!

**Optimal Nokta:**
- Test hatasının **en düşük** olduğu yer
- Lab 5'te: **Degree 9**

---

## 🧠 Bias ve Variance Nedir?

### Bias (Önyargı/Yanlılık)

**Tanım:** Modelin **sistematik hataları**

**Yüksek Bias:**
- Model çok basit
- Gerçek ilişkiyi yakalayamıyor
- **Underfitting**

**Düşük Bias:**
- Model yeterince karmaşık
- İlişkileri yakalayabiliyor

**Örnek:**
```
Gerçek ilişki: CO eğrisel olarak artıyor
Model (degree 1): Düz çizgi
Sonuç: Her zaman biraz yanılıyor → Yüksek bias
```

### Variance (Varyans/Değişkenlik)

**Tanım:** Modelin **farklı verilerle ne kadar değiştiği**

**Yüksek Variance:**
- Model çok karmaşık
- Eğitim verisine çok bağımlı
- Farklı veride çok değişiyor
- **Overfitting**

**Düşük Variance:**
- Model kararlı
- Farklı verilerde tutarlı

**Örnek:**
```
Degree 10 model:
Eğitim seti 1: CO = ...çok karmaşık formül 1...
Eğitim seti 2: CO = ...tamamen farklı formül 2...
Sonuç: Çok değişken → Yüksek variance
```

---

## ⚖️ Tradeoff (Ödünleşim) Nedir?

### Temel İlke

**Bias ↓ (azalttıkça) → Variance ↑ (artıyor)**
**Variance ↓ (azalttıkça) → Bias ↑ (artıyor)**

İkisini de aynı anda azaltamazsınız!

### Matematiksel

```
Toplam Hata = Bias² + Variance + Gürültü

Amaç: Bias² + Variance'ı minimize et!
```

### Görsel

```
           ↑ Bias & Variance
           |
Yüksek     |     Bias
           |      \
           |       \___
Bias/Var   |           \___      Optimal!
           |               \___×
           |                   \
           |        Variance    \___
Düşük      |_________________________→
             Basit            Karmaşık
                Model Kompleksitesi
```

---

## 🎯 Lab 5'te Bias-Variance

### Degree 1 (Linear)

```
Bias:     Yüksek ↑
Variance: Düşük ↓
Sonuç:    Underfitting

Neden?
- Çok basit formül
- Gerçek ilişkiyi yakalayamıyor (bias)
- Ama kararlı, tutarlı (düşük variance)
```

### Degree 9 (Optimal)

```
Bias:     Orta →
Variance: Orta →
Sonuç:    Dengeli!

Neden?
- Yeterince karmaşık (bias düşük)
- Ama çok da karmaşık değil (variance kontrol altında)
- **En iyi genelleme!**
```

### Degree 10 (Overfit)

```
Bias:     Düşük ↓
Variance: Yüksek ↑
Sonuç:    Overfitting

Neden?
- Çok karmaşık formül
- Eğitim verisine mükemmel uyuyor (bias düşük)
- Ama gürültüyü de öğrendi (yüksek variance)
```

---

## 📈 Pratik Nasıl Anlarsınız?

### Test 1: Eğitim vs Test Hatası

```
Her ikisi de yüksek:
  → Underfitting (degree çok düşük)

Test < Eğitim (yakın):
  → İyi fit (optimal degree)

Test >> Eğitim (büyük fark):
  → Overfitting (degree çok yüksek)
```

### Test 2: Error Gap

```python
gap = test_error - train_error

gap küçük (< 0.05):  Good!
gap orta (0.05-0.10): Dikkat
gap büyük (> 0.10):   Overfitting!
```

### Test 3: Cross-Validation

```
CV standard deviation yüksek:
  → Model tutarsız (overfitting riski)

CV standard deviation düşük:
  → Model kararlı (güvenilir)
```

---

## 💡 Gerçek Hayat Örnekleri

### Örnek 1: Hava Durumu Tahmini

**Underfitting:**
```
Model: "Her gün 20°C olacak"
Sonuç: Hiçbir zaman doğru değil
```

**Good Fit:**
```
Model: "Mevsimi, bölgeyi, tarihi dikkate al"
Sonuç: Genelde doğru tahminler
```

**Overfitting:**
```
Model: "Geçen yıl bugün 23.4°C'ydi, bu yıl da öyle olacak"
Sonuç: Çok spesifik, genelleyemiyor
```

### Örnek 2: Sınav Stratejisi

**Underfitting (Hiç çalışmama):**
- Konuyu anlamıyorsunuz
- Her sınavda kötü

**Good Fit (İyi çalışma):**
- Konuyu anlıyorsunuz
- Farklı sorulara adapte olabiliyorsunuz

**Overfitting (Sadece ezber):**
- Çözdüğünüz soruları mükemmel biliyorsunuz
- Ama yeni soru tiplerinde başarısız oluyorsunuz

---

## 🔧 Nasıl Düzeltiriz?

### Underfitting'i Çözmek

1. **Model kompleksitesini artır**
   ```python
   degree = 1 → degree = 3
   ```

2. **Daha fazla özellik ekle**
   ```python
   Sadece T → T, RH, AH, Rüzgar, Saat...
   ```

3. **Daha karmaşık model kullan**
   ```python
   Linear → Polynomial
   ```

### Overfitting'i Çözmek

1. **Model kompleksitesini azalt**
   ```python
   degree = 10 → degree = 5
   ```

2. **Daha fazla veri topla**
   ```
   5,000 örnek → 50,000 örnek
   ```

3. **Regularization kullan**
   ```python
   Ridge, Lasso regression
   ```

4. **Cross-validation yap**
   ```python
   Tek split yerine 5-fold CV
   ```

---

## 🎓 Lab 5'in Sonuçları

### Bulgularımız

| Degree | Train MSE | Test MSE | Gap | Durum |
|--------|-----------|----------|-----|-------|
| 1      | 2.04      | 2.06     | 0.01 | Underfitting |
| 5      | 1.98      | 2.01     | 0.03 | İyi |
| **9**  | **1.95**  | **1.98** | **0.03** | **Optimal!** |
| 10     | 1.94      | 1.99     | 0.05 | Başlangıç overfitting |

### Sonuç

- **Degree 9**: En iyi genelleme
- **Neden?** Bias ve variance dengesi!
- **Cross-validation:** Degree 1 daha stabil (ama daha yüksek hata)

---

## 💭 Önemli Kavramlar

### 1. Training Error Aldatıcıdır!

```
❌ "Eğitim hatam 0.001, süper model!"
✅ "Test hatam 0.05, iyi model!"
```

### 2. Test Error Gerçeği Gösterir!

```
Test hatası = Gerçek performans
```

### 3. Gap Overfitting Göstergesi!

```
Büyük gap = Ezber yapıyor = Overfitting!
```

### 4. Optimal ≠ Eğitim Hatası En Düşük!

```
Optimal = Test Hatası En Düşük!
```

---

## 🚀 Özet

### Bias-Variance Tradeoff 3 Cümlede

1. **Çok basit model:** Yüksek bias → Underfitting
2. **Çok karmaşık model:** Yüksek variance → Overfitting
3. **Optimal model:** Bias ve variance dengesi → Best generalization!

### Lab 5'in Mesajı

**"Daha karmaşık her zaman daha iyi değildir!"**

En iyi model, **test verisinde en iyi performans gösterendir**!

---

**Hazırlayan:** Muhammed Ali Karataş (2021403030)
**Tarih:** 12 Kasım 2025
