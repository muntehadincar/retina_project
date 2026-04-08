# 🔬 RetinAI — Retinal Damar Segmentasyon Sistemi

> **Attention U-Net** tabanlı, uçtan uca retinal damar segmentasyonu ve klinik özellik çıkarımı yapan web uygulaması.

### 💻 Arayüzden Görüntüler
![RetinAI Web Arayüzü - Başlangıç](assets/web_interface.png)
![RetinAI Analiz Sonucu](assets/web_interface1.png)

---


## 📌 Proje Hakkında

Bu proje, göz fundus görüntülerindeki retinal damarları otomatik olarak segmente eden ve klinik özellikler çıkaran bir yapay zeka sistemidir. Kullanıcı bir fundus görüntüsü yüklediğinde sistem:

1. Görüntüyü **Attention U-Net** modeliyle işler
2. Damar segmentasyon maskesini üretir
3. **Damar piksel sayısı**, **alan oranı** ve **damar yoğunluğu** gibi klinik özellikleri hesaplar
4. Sonuçları web arayüzünde görsel olarak sunar

---

## 🏗️ Proje Yapısı

```
retina_project/
├── src/                          # Model eğitim ve değerlendirme kodları
│   ├── model.py                  # UNet ve Attention U-Net mimarileri (PyTorch)
│   ├── train.py                  # UNet eğitim döngüsü
│   ├── train_attention.py        # Attention U-Net eğitim döngüsü
│   ├── train_compare.py          # İki modeli aynı anda karşılaştırmalı eğitme
│   ├── evaluate.py               # Test seti değerlendirmesi (Dice, IoU, vb.)
│   ├── clinical_features.py      # Klinik özellik çıkarımı ve CSV üretimi
│   ├── visualize.py              # Tahmin görselleştirme
│   ├── dataset.py                # Dataset sınıfı
│   ├── utils.py                  # Loss fonksiyonları, metrikler, yardımcı araçlar
│   └── split_test.py             # Test/train/val bölme scripti
│
├── retina_system/                # Web uygulaması (Full-Stack)
│   ├── backend/                  # FastAPI sunucu
│   │   ├── main.py               # Uygulama giriş noktası
│   │   ├── routers/predict.py    # /api/predict endpointi
│   │   ├── services/model_service.py  # Inference servisi
│   │   └── core/config.py        # Konfigürasyon (model yolu, eşik vb.)
│   ├── frontend/public/          # Statik web arayüzü
│   │   ├── index.html            # Ana sayfa (RetinAI web UI)
│   │   ├── style.css             # Glassmorphism tasarım sistemi
│   │   └── script.js             # Görüntü yükleme ve analiz akışı
│   ├── models/                   # Eğitilmiş model ağırlıkları (.pth)
│   └── run.bat                   # Tek tıkla başlatma scripti (Windows)
│
├── data/                         # Veri seti (paylaşılmaz, .gitignore)
└── results/                      # Değerlendirme sonuçları, CSV çıktıları
```

---

## 🤖 Model Mimarisi

### Attention U-Net
Oktay et al. (2018) makalesine dayanan bu mimari, standart U-Net'e **Attention Gate** ekler. Decoder'daki her skip connection, encoder'dan gelen özellik haritasını dikkat mekanizmasıyla ağırlıklandırarak damar gibi ince yapılara odaklanmayı iyileştirir.

```
Encoder (Down) → Bottleneck → Decoder (Up)
                                  ↑
                           AttentionGate
                          (skip × alpha)
```

### Loss Fonksiyonu
`BCEWithLogitsLoss` + `DiceLoss` kombinasyonu kullanılmıştır. Damar pikselleri az olduğundan sınıf dengesizliğini gidermek için `pos_weight=5.0` uygulanmıştır.

### Değerlendirme Metrikleri
| Metrik | Açıklama |
|---|---|
| **Dice** | Ana segmentasyon başarım ölçütü |
| **IoU** | Kesişim / Birleşim oranı |
| **Precision / Recall** | Kesinlik ve duyarlılık |
| **Sensitivity / Specificity** | Klinik doğruluk metrikleri |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
```bash
pip install -r requirements.txt
```

Ana bağımlılıklar: `torch`, `torchvision`, `fastapi`, `uvicorn`, `opencv-python`, `Pillow`, `numpy`

### Web Uygulamasını Başlatma (Windows)

```bash
cd retina_system
run.bat
```

veya manuel olarak:

```bash
cd retina_system/backend
uvicorn main:app --reload --port 8000
```

Tarayıcıda aç: [http://localhost:8000](http://localhost:8000)

> ⚠️ Çalıştırmadan önce `retina_system/models/` klasörüne eğitilmiş model ağırlığını (`attention_unet_best.pth`) yerleştirin.

### Model Eğitimi

```bash
# Attention U-Net eğitimi
python src/train_attention.py

# Her iki modeli karşılaştırmalı eğitme
python src/train_compare.py
```

### Değerlendirme ve Klinik Özellik Çıkarımı

```bash
# Test seti değerlendirmesi
python src/evaluate.py

# Klinik özellik CSV üretimi
$env:PYTHONUTF8=1; python src/clinical_features.py
```

---

## 🌐 API Kullanımı

`POST /api/predict` — Görüntü yükle, segmentasyon al

```json
// İstek: multipart/form-data — "file" alanı
// Yanıt:
{
  "mask_base64": "<base64 PNG>",
  "vessel_pixel_count": 4821,
  "vessel_area_ratio": 0.0736,
  "vessel_density": 0.0736
}
```

---

## 📊 Veri Seti


- 40 renkli fundus görüntüsü (20 eğitim, 20 test)
- 565×584 piksel çözünürlük
- Manuel damar segmentasyon maskeleri

Görüntüler eğitim öncesinde **CLAHE** (Contrast Limited Adaptive Histogram Equalization) ile önişlemden geçirilmiştir.

---

## 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|---|---|
| **Derin Öğrenme** | PyTorch |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML / CSS / JavaScript |
| **Görüntü İşleme** | OpenCV, Pillow |
| **Veri Analizi** | NumPy, CSV |

---

## 👩‍💻 Geliştirici

**Münteha Dincar**
retinai Fundus Görüntülerinde Damar Segmentasyonu — Tıp Mühendisliği Bitirme Projesi 2026
