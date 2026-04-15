# 🔬 retinai — retinai Damar Segmentasyon Sistemi

> **ResUNet** tabanlı, uçtan uca retinal damar segmentasyonu ve klinik özellik çıkarımı yapan web uygulaması.

<p align="center">
  <img src="assets/web_interface.png" alt="RetinAI Web Arayüzü" width="100%">
  <img src="assets/web_interface1.png" alt="RetinAI Analiz Sonucu" width="100%">
</p>

---

## 📌 Proje Hakkında

Bu proje, göz fundus görüntülerindeki retinal damarları otomatik olarak segmente eden ve klinik özellikler çıkaran yapay zeka destekli bir web sistemidir.

Kullanıcı bir fundus görüntüsü yüklediğinde sistem:

1. Görüntüyü **ResUNet** modeliyle işler
2. Damar segmentasyon maskesini üretir
3. **Damar piksel sayısı**, **alan oranı** ve **damar yoğunluğu** gibi klinik özellikleri hesaplar
4. Sonuçları web arayüzünde görsel olarak sunar

---

## 🏗️ Proje Yapısı

```
retina_project/
├── src/                               # Model eğitim ve değerlendirme kodları
│   ├── model.py                       # UNet, AttentionUNet, ResUNet, SegFormerLite, SwinUNet
│   ├── train.py                       # UNet eğitim döngüsü
│   ├── train_attention.py             # Attention U-Net eğitimi
│   ├── train_resunet.py               # ResUNet eğitimi
│   ├── train_segformer.py             # SegFormer eğitimi
│   ├── train_swinunet.py              # Swin-UNet eğitimi
│   ├── train_compare.py               # UNet vs AttentionUNet karşılaştırmalı eğitim
│   ├── evaluate.py                    # Test seti değerlendirmesi (5 model, Dice, IoU vb.)
│   ├── plot_multi_compare.py          # 5 model karşılaştırma grafikleri
│   ├── clinical_features.py           # Klinik özellik çıkarımı ve CSV üretimi
│   ├── visualize.py                   # Tahmin görselleştirme
│   ├── dataset.py                     # Dataset sınıfı
│   ├── utils.py                       # Loss fonksiyonları, metrikler, yardımcı araçlar
│   └── split_test.py                  # Test/train/val bölme scripti
│
├── retina_system/                     # Web uygulaması (Full-Stack)
│   ├── backend/                       # FastAPI sunucu
│   │   ├── main.py                    # Uygulama giriş noktası
│   │   ├── routers/predict.py         # /api/predict endpointi
│   │   ├── services/model_service.py  # ResUNet inference servisi
│   │   └── core/config.py             # Konfigürasyon (model yolu, eşik vb.)
│   ├── frontend/public/               # Statik web arayüzü
│   │   ├── index.html                 # Ana sayfa (RetinAI web UI)
│   │   ├── style.css                  # Glassmorphism tasarım sistemi
│   │   └── script.js                  # Görüntü yükleme ve analiz akışı
│   ├── models/                        # Eğitilmiş model ağırlıkları (.pth)
│   └── run.bat                        # Tek tıkla başlatma scripti (Windows)
│
├── data/                              # Veri seti (paylaşılmaz, .gitignore)
├── results/                           # Değerlendirme sonuçları, CSV ve grafikler
│   ├── logs/                          # Epoch bazlı eğitim logları (CSV)
│   ├── evaluation/                    # Test seti metrik çıktıları (CSV)
│   ├── plots/                         # Karşılaştırmalı grafik görselleri
│   └── models/                        # Kayıtlı model ağırlıkları
└── assets/                            # README görselleri
```

---

## 🤖 Model Mimarileri

Projede **5 farklı segmentasyon mimarisi** karşılaştırmalı olarak eğitilmiş ve değerlendirilmiştir:

| Model | Tür | Parametre | Açıklama |
|---|---|---|---|
| **UNet** | CNN | 31.0M | Klasik encoder-decoder mimarisi |
| **Attention U-Net** | CNN | 34.9M | UNet + Attention Gate ile ince damar odağı |
| **ResUNet** ⭐ | CNN | 32.4M | Residual (artık) bağlantılı U-Net — **En Başarılı** |
| **SegFormer Lite** | Transformer | 1.3M | MiT-inspired DWSConv + MLP decoder |
| **Swin-UNet** | Transformer | 34.5M | Swin Transformer encoder + CNN decoder |

### 🏆 ResUNet — Üretim Modeli

ResUNet, standart U-Net'in encoder bloklarını **Residual Block** yapısıyla güçlendirir. Her blokta giriş ve çıkış arasında kısayol (shortcut) bağlantı eklenir; bu sayede gradyan akışı iyileşir ve derin ağlarda eğitim kararlılığı artar.

```
[Giriş] ──► ResidualBlock ──► MaxPool ──► ... ──► Bottleneck
                │                                       │
                └────── skip connection ────────────────┘
                                                        │
                              ConvTranspose ◄──────────-┘
                                    │
                              ResidualBlock
                                    │
                                [Çıkış Maskesi]
```

**ResidualBlock yapısı:**
```
x ──► Conv3×3 ──► BN ──► ReLU ──► Conv3×3 ──► BN ──► (+) ──► ReLU
│                                                       ▲
└─────────────────── shortcut (Conv1×1 veya Identity) ──┘
```

---

## 📊 Model Performans Karşılaştırması

5 model, **119 test görüntüsü** üzerinde değerlendirilmiştir (sample 35–36 hariç: boş maske):

| Model | Dice ↑ | IoU ↑ | Accuracy ↑ | Sensitivity ↑ | Specificity ↑ | Precision ↑ |
|---|---|---|---|---|---|---|
| **ResUNet** 🥇 | **0.7564** | **0.6243** | **0.9601** | 0.9133 | **0.9631** | **0.6631** |
| Attention U-Net 🥈 | 0.7329 | 0.5882 | 0.9495 | **0.9198** | 0.9505 | 0.6202 |
| UNet 🥉 | 0.7176 | 0.5704 | 0.9506 | 0.8378 | 0.9588 | 0.6440 |
| SwinUNet | 0.6458 | 0.4842 | 0.9278 | 0.8605 | 0.9319 | 0.5266 |
| SegFormer Lite | 0.5259 | 0.3645 | 0.9040 | 0.7539 | 0.9146 | 0.4194 |

> **Dice skoru**, segmentasyonda tahmin edilen ve gerçek mask piksellerinin örtüşme oranını ölçer. En kritik başarım metriğidir.

**Sonuç:** ResUNet, residual bağlantılar sayesinde ince damar yapılarını yakalamada CNN tabanlı modeller arasında en yüksek başarımı göstermiştir. Transformer tabanlı modeller (SwinUNet, SegFormer) sınırlı veri setinde CNN modellerinin gerisinde kalmıştır.

---

## ⚙️ Loss Fonksiyonu ve Eğitim

```python
Loss = BCEWithLogitsLoss(pos_weight=5.0) + DiceLoss
```

- **BCEWithLogitsLoss** — Piksel bazlı ikili sınıflandırma
- **DiceLoss** — Segmentasyon örtüşme odaklı düzeltici
- **pos_weight=5.0** — Damar piksellerinin azlığından kaynaklanan sınıf dengesizliğini giderir

### Değerlendirme Metrikleri

| Metrik | Açıklama |
|---|---|
| **Dice** | 2×TP / (2×TP + FP + FN) — Ana başarım ölçütü |
| **IoU** | TP / (TP + FP + FN) — Kesişim / Birleşim oranı |
| **Precision** | TP / (TP + FP) — Kesinlik |
| **Sensitivity (Recall)** | TP / (TP + FN) — Duyarlılık |
| **Specificity** | TN / (TN + FP) — Özgüllük |
| **Accuracy** | (TP + TN) / Toplam piksel sayısı |

---

## 📂 Veri Seti

Bu çalışmada kullanılan retina görüntüleri, 4–83 yaş aralığındaki bireyleri kapsayan açık erişimli veri setlerinden elde edilmiştir:

| Veri Seti | Kaynak |
|---|---|
| **DRIVE** | Digital Retinal Images for Vessel Extraction |
| **STARE** | Structured Analysis of the Retina |
| **CHASEDB1** | Child Heart and Health Study in England Database |
| **FIVES** | Fundus Image Vessel Segmentation |

### Görüntü Özellikleri

- **600 renkli retina fundus görüntüsü** + **600 manuel damar segmentasyon maskesi**
- 565×584 piksel çözünürlük
- Eğitim öncesinde **CLAHE** (Contrast Limited Adaptive Histogram Equalization) ile kontrast artırma uygulanmıştır

### Veri Bölünmesi

| Set | Görüntü Sayısı | Kullanım |
|---|---|---|
| **Eğitim (Train)** | ~380 | Model ağırlıklarının öğrenilmesi |
| **Doğrulama (Validation)** | ~100 | Overfitting kontrolü |
| **Test** | ~120 | Nihai performans değerlendirmesi |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install -r requirements.txt
```

Ana bağımlılıklar: `torch`, `torchvision`, `fastapi`, `uvicorn`, `opencv-python`, `Pillow`, `numpy`

---

### 🖥️ Web Uygulamasını Başlatma (Windows)

```bash
cd retina_system
run.bat
```

veya manuel olarak:

```bash
cd retina_system/backend
python -m uvicorn main:app --reload --port 8000
```

Tarayıcıda aç: [http://localhost:8000](http://localhost:8000)

> ⚠️ Çalıştırmadan önce `retina_system/models/` klasörüne eğitilmiş model ağırlığını (`resunet_best.pth`) yerleştirin.

---

### 🧠 Model Eğitimi

```bash
# ResUNet eğitimi
python src/train_resunet.py

# Attention U-Net eğitimi
python src/train_attention.py

# Tüm modelleri karşılaştırmalı değerlendirme
python src/evaluate.py
```

---

### 📈 Değerlendirme ve Görselleştirme

```bash
# Test seti değerlendirmesi
python src/evaluate.py

# Karşılaştırmalı grafik üretimi
python src/plot_multi_compare.py

# Klinik özellik CSV üretimi
$env:PYTHONUTF8=1; python src/clinical_features.py
```

---

## 🌐 API Kullanımı

### `POST /api/predict`

Görüntü yükle, segmentasyon maskesi ve klinik özellikler al.

**İstek:** `multipart/form-data` — `file` alanı (PNG/JPEG)

**Yanıt:**
```json
{
  "mask_base64": "<base64 PNG>",
  "vessel_pixel_count": 4821,
  "vessel_area_ratio": 0.0736,
  "vessel_density": 0.0736
}
```

---

## 📊 Veri Seti

Bu çalışmada kullanılan retina görüntüleri, 4–83 yaş aralığındaki bireyleri kapsayan ve retina damar segmentasyonu alanında yaygın olarak kullanılan açık erişimli veri setlerinden elde edilmiştir. Veri seti oluşturulurken aşağıdaki veri tabanlarından yararlanılmıştır:

- **DRIVE** (Digital Retinal Images for Vessel Extraction)
- **STARE** (Structured Analysis of the Retina)
- **CHASEDB1** (Child Heart and Health Study in England Database)
- **FIVES** (Fundus Image Vessel Segmentation)

Bu veri setleri, retina damarlarının otomatik tespiti ve segmentasyonu üzerine gerçekleştirilen akademik çalışmalarda sıklıkla kullanılan ve referans niteliği taşıyan veri kaynaklarıdır.

### Görüntü Özellikleri
- **600 renkli retina fundus görüntüsü** + **600 manuel damar segmentasyon maskesi**
- 565×584 piksel çözünürlük
- Eğitim öncesinde **CLAHE** (Contrast Limited Adaptive Histogram Equalization) ile kontras artırma uygulanmıştır

### Veri Bölünmesi

| Set | Görüntü Sayısı | Maske Sayısı | Kullanım |
|---|---|---|---|
| **Eğitim (Train)** | ~380 | ~380 | Model ağırlıklarının öğrenilmesi |
| **Doğrulama (Validation)** | ~100 | ~100 | Eğitim sırasında overfitting kontrolü |
| **Test** | ~120 | ~120 | Nihai model performans değerlendirmesi |

**Örnek cURL:**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@retina_image.png"
```

Swagger dokümantasyonu: [http://localhost:8000/docs](http://localhost:8000/docs)
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
