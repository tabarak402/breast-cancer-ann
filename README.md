#  Meme Kanseri Riski Tahmini — Yapay Sinir Ağı (ANN)

> Wisconsin Breast Cancer Dataset kullanılarak iyi huylu / kötü huylu tümör sınıflandırması

---

##  Proje Hakkında

Bu proje, hücre çekirdeği özelliklerinden yola çıkarak meme kanseri vakalarının
**Benign (İyi Huylu)** veya **Malignant (Kötü Huylu)** olarak sınıflandırılmasını amaçlamaktadır.

Klinik değerlendirmeleri destekleyici bir yapay zeka aracı olarak tasarlanmıştır.
Erken tanıya katkı sağlamak ve kullanıcıyı tarama/kontrole yönlendirmek hedeflenmektedir.

---

##  Model Performansı

| Metrik | Değer |
|---|---|
| Doğruluk (Accuracy) | **%98.25** |
| AUC-ROC | **0.9962** |
| Precision (Malignant) | %97 |
| Recall (Malignant) | %98 |
| F1-Score | %98 |

---

##  Model Mimarisi

Giriş (30 özellik)
↓
Gizli Katman 1 — 32 nöron, tanh aktivasyon
↓
Gizli Katman 2 — 16 nöron, sigmoid aktivasyon
↓
Çıkış — 1 nöron, sigmoid (0=Benign, 1=Malignant)


- **Optimizer:** Adam (lr=0.03)
- **Loss:** Binary Crossentropy
- **Early Stopping:** val_loss izlenerek en iyi ağırlıklar korundu
- **Veri Bölümü:** %70 eğitim, %30 test

---

##  Veri Seti

- **Kaynak:** [Kaggle — Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Boyut:** 569 hasta, 30 özellik
- **Hedef:** diagnosis — M (Malignant=1) / B (Benign=0)

>  `data/data.csv` repoya dahil değildir. Kaggle'dan indirip `data/` klasörüne koyunuz.

---

##  Kurulum ve Çalıştırma

```bash
# 1. Repoyu klonla
git clone https://github.com/tabarak402/breast-cancer-ann.git
cd breast-cancer-ann

# 2. Sanal ortam oluştur
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Mac/Linux

# 3. Kütüphaneleri yükle
pip install -r requirements.txt

# 4. Modeli çalıştır
python main.py
```

---

##  Klasör Yapısı

breast-cancer-ann/
├── data/
│   └── data.csv          # Kaggle'dan indirilecek
├── src/
│   ├── preprocess.py     # Veri ön işleme
│   ├── model.py          # ANN mimarisi + eğitim
│   └── evaluate.py       # Metrikler + grafikler
├── outputs/              # Grafikler ve model
├── main.py               # Ana çalıştırma dosyası
├── requirements.txt
└── README.md

---

##  Çıktı Grafikleri

| Dosya | Açıklama |
|---|---|
| `1_training_history.png` | Eğitim/doğrulama doğruluğu ve kaybı |
| `2_confusion_matrix.png` | Gerçek vs tahmin karşılaştırması |
| `3_roc_curve.png` | ROC eğrisi ve AUC skoru |
| `4_prediction_distribution.png` | Tahmin olasılığı dağılımı |
| `ann_model.keras` | Eğitilmiş ANN modeli |
| `scaler.pkl` | Normalizasyon için kaydedilmiş scaler |

---

##  API Kullanımı

API'yi başlatmak için:
```bash
uvicorn api:app --reload
```

Tahmin yapmak için POST isteği:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [17.99, 10.38, ...]}'
```

Yanıt:
```json
{
  "tahmin": "Malignant (Kötü Huylu)",
  "olasilik": 0.9978,
  "risk": "YÜKSEK"
}
```