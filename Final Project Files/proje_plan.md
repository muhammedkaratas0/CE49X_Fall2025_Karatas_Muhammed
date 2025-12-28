# Proje Yol Haritası: Civil Engineering & AI Trends Analyzer

**Ders:** CE49X - Introduction to Data Science for Civil Engineering
**Dönem:** Fall 2025
**Yaklaşım:** RSS Tabanlı Veri Toplama + LLM Destekli Analiz + Supabase Veritabanı

---

## 1. Proje Özeti ve Mimari

Bu proje, İnşaat Mühendisliği alanındaki yapay zeka (AI) kullanım trendlerini belirlemek amacıyla geliştirilmiştir. Klasik kelime sayma yöntemleri yerine, **Büyük Dil Modelleri (LLM)** kullanılarak makaleler anlamsal olarak analiz edilir, özetlenir ve etiketlenir.

### Veri Akış Şeması (Pipeline)

1. **Kaynak (Ingestion):** RSS Feed'leri (ScienceDaily, ENR, TechCrunch vb.) üzerinden otomatik veri çekimi.
2. **İşleme (Processing):** Her makale Python scripti ile çekilir ve bir LLM'e (GPT-4o-mini veya Llama) gönderilir.
3. **Zeka (Intelligence):** LLM, makaleyi okur, özetler ve önceden belirlenen etiketlere (Kategori, AI Teknolojisi) göre sınıflandırır.
4. **Depolama (Storage):** İşlenen yapısal veri **Supabase** (PostgreSQL) veritabanına kaydedilir.
5. **Görselleştirme (Visualization):** Veritabanından çekilen veriler ile Python (Matplotlib/Seaborn) kullanılarak trend grafikleri oluşturulur.

---

## 2. Teknoloji Yığını (Tech Stack)

| Bileşen                   | Araç                | Nedeni                                                                                                             |
| :------------------------- | :------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **Dil**              | Python 3.8+          | Veri bilimi için standart .                                                                                       |
| **Database**         | **Supabase**   | Ücretsiz, kurulum gerektirmez, Python ile kolay entegre olur. (500MB text verisi için sınır sorunu yaşanmaz). |
| **Veri Çekme**      | `feedparser`       | RSS akışlarını okumak için en stabil kütüphane.                                                             |
| **AI/LLM**           | OpenAI API / Ollama  | Metni anlayıp JSON formatında etiketlemek için.                                                                 |
| **Görselleştirme** | Matplotlib / Seaborn | Isı haritaları ve grafikler için .                                                                              |

---

## 3. Veritabanı Tasarımı (Supabase)

Supabase üzerinde `articles` adında bir tablo oluşturulmalıdır. Bu yapı, hocanın "Structured Format" isteğini fazlasıyla karşılar.

**Tablo Adı:** `articles`

| Sütun Adı      | Veri Tipi | Açıklama                                                        |
| :--------------- | :-------- | :---------------------------------------------------------------- |
| `id`           | uuid      | Benzersiz kayıt ID'si (Primary Key).                             |
| `title`        | text      | Makale başlığı .                                              |
| `url`          | text      | Makalenin linki.                                                  |
| `published_at` | timestamp | Yayınlanma tarihi .                                              |
| `source_name`  | text      | Kaynak adı (örn: ENR, TechCrunch) .                             |
| `summary`      | text      | **LLM Çıktısı:** Makalenin 3 cümlelik Türkçe özeti. |
| `category`     | text      | **LLM Çıktısı:** (Structural, Geotechnical, etc.) .     |
| `ai_tech`      | text      | **LLM Çıktısı:** (Computer Vision, ML, etc.) .          |
| `sentiment`    | text      | **LLM Çıktısı:** (Positive, Neutral, Negative).         |

---

## 4. LLM Entegrasyonu ve Prompt Tasarımı

Python kodun, makale metnini LLM'e gönderirken aşağıdaki "System Prompt"u kullanmalıdır. Bu, verinin veritabanına temiz girmesini sağlar.

**System Prompt:**

> "Sen uzman bir inşaat mühendisliği ve veri analistisin. Sana verilen makale başlığını ve özetini analiz et. Cevabını SADECE aşağıdaki JSON formatında ver:
>
> {
> "summary": "Makalenin 2-3 cümlelik Türkçe özeti",
> "category": "Makalenin odaklandığı alan (Sadece şunlardan biri: Structural, Geotechnical, Transportation, Construction Management, Environmental, Other)",
> "ai_tech": "Kullanılan ana teknoloji (Sadece şunlardan biri: Computer Vision, Machine Learning, Robotics, Generative AI, IoT, None)",
> "sentiment": "Makalenin tonu (Positive, Neutral, Negative)"
> }
>
> Başka hiçbir metin ekleme, sadece JSON döndür."

---

## 5. Veri Kaynakları (RSS Listesi)

Hocanın önerdiği RSS yöntemi için kullanılacak yüksek kaliteli kaynaklar:

* **ScienceDaily (Civil Eng):** `https://www.sciencedaily.com/rss/matter_energy/civil_engineering.xml`
* **Construction Dive:** `https://www.constructiondive.com/feeds/news/`
* **MIT Tech Review:** `https://www.technologyreview.com/feed/`
* **BIMplus:** `https://www.bimplus.co.uk/feed/`
* **TechCrunch (Tag: AI):** `https://techcrunch.com/category/artificial-intelligence/feed/`

---

## 6. Uygulama Adımları (To-Do)

### Aşama 1: Kurulum

- [ ] Supabase hesabı aç ve yeni proje oluştur.
- [ ] SQL Editor kısmından `articles` tablosunu oluştur.
- [ ] Python ortamını kur: `pip install feedparser supabase openai pandas`

### Aşama 2: Python Script (Extract & Transform)

- [ ] `feedparser` ile RSS linklerinden son haberleri çeken döngüyü yaz.
- [ ] Gelen veriyi (Title + Description) alıp OpenAI API'ye (Prompt ile) gönder.
- [ ] OpenAI'dan gelen JSON cevabını parse et (`json.loads`).

### Aşama 3: Yükleme (Load)

- [ ] Parse edilen veriyi (`supabase.table('articles').insert({...})`) komutuyla veritabanına yaz.

### Aşama 4: Analiz ve Raporlama

- [ ] Veritabanındaki veriyi Pandas DataFrame'e çek.
- [ ] **Soru 1:** Hangi alanda en çok makale var? (Bar Chart)
- [ ] **Soru 2:** Hangi kategori hangi AI teknolojisini kullanıyor? (Heatmap)
- [ ] **Rapor:** Final raporuna LLM tarafından oluşturulan özetlerden örnekler ekle.

---

## 7. Teslim Edilecekler Kontrol Listesi

* [ ] **Code Repository:** GitHub linki (Scriptler ve Requirements.txt) .
* [ ] **Database Export:** Supabase'den alınan örnek veri seti (CSV/JSON).
* [ ] **Final Rapor:** Metodolojide "LLM tabanlı semantik analiz" kullandığını vurgula .
* [ ] **Sunum:** Canlı demoda Supabase dashboard'unu göstererek verilerin nasıl aktığını anlat .
