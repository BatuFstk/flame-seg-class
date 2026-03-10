# FLAME - Drone Goruntuleriyle Yangin Tespiti

Drone/UAV hava goruntuleri uzerinde yangin tespiti yapan derin ogrenme projesi. Proje iki temel gorev icerir:

- **Siniflandirma**: Goruntude yangin var mi yok mu? (ResNet18 - Transfer Learning)
- **Segmentasyon**: Yangin bolgesinin piksel bazli tespiti (U-Net)

## Veri Seti

Bu projede **FLAME (Fire Luminosity Airborne-based Machine learning Evaluation)** veri seti kullanilmistir.

- **Kaynak**: [IEEE DataPort - FLAME Dataset](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)
- **Aciklama**: Drone'lar ile cekilen hava goruntuleri uzerinde yangin tespiti icin olusturulmus acik erisimli bir veri setidir.
- **Siniflandirma verisi**: Fire / No_Fire etiketli 254x254 piksel goruntular
- **Segmentasyon verisi**: 3840x2160 piksel goruntular + binary yangin maskeleri

### Veri Seti Dosyalari

Asagidaki dosyalari indirip `data/` klasorune yerlestiriniz:

| Dosya | Aciklama |
|-------|----------|
| `Training.zip` | Siniflandirma egitim verisi (Fire / No_Fire) |
| `Test.zip` | Siniflandirma test verisi |
| `Images.zip` | Segmentasyon RGB goruntuleri |
| `Masks.zip` | Segmentasyon binary maskeleri |

## Proje Yapisi

```
flame-seg-class/
├── app.py                 # Gradio web arayuzu
├── requirements.txt       # Bagimliliklar
├── src/
│   ├── config.py          # Tum ayarlar ve hiperparametreler
│   ├── dataset.py         # Veri yukleyiciler
│   ├── model.py           # Model mimarileri (ResNet18, U-Net)
│   ├── train.py           # Egitim scripti
│   ├── evaluate.py        # Degerlendirme scripti
│   ├── predict.py         # Tek gorsel tahmin (CLI)
│   └── utils.py           # Metrikler, checkpoint, gorselestirme
├── checkpoints/           # Egitilmis model dosyalari (.pth)
├── data/                  # Veri seti zip dosyalari
├── test-image/            # Ornek test gorselleri
└── runs/                  # TensorBoard loglari
```

## Kurulum

```bash
git clone https://github.com/KULLANICI_ADIN/flame-seg-class.git
cd flame-seg-class
pip install -r requirements.txt
```

## Kullanim

### Web Arayuzu (Gradio)

Gorseli surukle-birak ile yukleyip sonucu gorun:

```bash
python app.py
```

Tarayicida `http://localhost:7860` adresini acin.

### Komut Satiri ile Tahmin

```bash
# Siniflandirma
python src/predict.py --task classification --image test-image/orman-yangini.jpg

# Segmentasyon
python src/predict.py --task segmentation --image test-image/orman-yangini.jpg

# Segmentasyon sonucunu dosyaya kaydet
python src/predict.py --task segmentation --image test-image/orman-yangini.jpg --output sonuc.png
```

### Egitim

Veri setini `data/` klasorune yerlestirdikten sonra:

```bash
# Sadece siniflandirma
python src/train.py --task classification

# Sadece segmentasyon
python src/train.py --task segmentation

# Her ikisi
python src/train.py --task both
```

### Degerlendirme

```bash
python src/evaluate.py --task classification
python src/evaluate.py --task segmentation
```

## Egitilmis Modeller

Onceden egitilmis model dosyalari [Releases](../../releases) sayfasindan indirilebilir.

Indirdikten sonra `checkpoints/` klasorune yerlestiriniz:

```
checkpoints/
├── best_classifier.pth
└── best_segmentation.pth
```

## Model Mimarileri

### Siniflandirma - FlameClassifier
- **Temel**: ResNet18 (ImageNet pretrained)
- **Transfer Learning**: Son FC katmani 2 sinifa (Fire/No_Fire) uyarlanmis
- **Giris**: 254x254 RGB goruntu
- **Cikis**: 2 sinif (Fire, No_Fire)

### Segmentasyon - FlameUNet
- **Mimari**: U-Net (4 seviye encoder-decoder + skip connections)
- **Giris**: 256x256 RGB goruntu
- **Cikis**: 1 kanal binary maske (yangin var/yok)
- **Metrikler**: IoU, Dice, Pixel Accuracy

## Hiperparametreler

| Parametre | Siniflandirma | Segmentasyon |
|-----------|:------------:|:------------:|
| Image Size | 254x254 | 256x256 |
| Batch Size | 32 | 8 |
| Learning Rate | 1e-4 | 1e-4 |
| Optimizer | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |

## Teknolojiler

- Python 3.12
- PyTorch 2.10
- Gradio (Web arayuzu)
- TensorBoard (Egitim takibi)

## Referans

```
Shamsoshoara, A., Afghah, F., Razi, A., Zheng, L., Ful e, P., & Blasch, E. (2021).
Aerial Imagery Pile burn detection using Deep Learning: the FLAME dataset.
Computer Networks, 193, 108001.
```
