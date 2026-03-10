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

### Siniflandirma Verisi

| Set | Fire | No_Fire | Toplam | Boyut |
|-----|:----:|:-------:|:------:|:-----:|
| **Training** | 25,018 | 14,357 | 39,375 | 1.2 GB |
| **Test** | 5,137 | 3,480 | 8,617 | 287 MB |

- Gorsel boyutu: **254x254** piksel (JPG)
- Sinif dagilimi: Fire %63 / No_Fire %37

### Segmentasyon Verisi

| Veri | Adet | Boyut |
|------|:----:|:-----:|
| **RGB Goruntuleri** | 2,003 | 5.1 GB |
| **Binary Maskeler** | 2,003 | 9 MB |

- Gorsel boyutu: **3840x2160** piksel (4K), egitimde 256x256'ya resize edilir
- Goruntu format: JPG / Maske format: PNG
- Maske degerleri: 0 = arka plan, 1 = yangin

### Veri Seti Dosyalari

Asagidaki dosyalari [IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs) adresinden indirip `data/` klasorune yerlestiriniz:

| Dosya | Aciklama | Boyut |
|-------|----------|:-----:|
| `Training.zip` | Siniflandirma egitim verisi (Fire / No_Fire) | 1.2 GB |
| `Test.zip` | Siniflandirma test verisi | 287 MB |
| `Images.zip` | Segmentasyon RGB goruntuleri | 5.1 GB |
| `Masks.zip` | Segmentasyon binary maskeleri | 9 MB |

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
git clone https://github.com/BatuFstk/flame-seg-class.git
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

## Değerlendirme Sonucu metrikler

<img width="1877" height="830" alt="segmen" src="https://github.com/user-attachments/assets/9f9e411a-3b1f-4a9a-a1fd-d72e258d385c" />

<img width="1911" height="829" alt="segmen2" src="https://github.com/user-attachments/assets/675fb6e4-cc4d-46db-9ad8-87a57fa29ea1" />

<img width="1900" height="768" alt="segmen3" src="https://github.com/user-attachments/assets/64f9cfe8-1663-4667-a0d3-4d9963484b2a" />

<img width="757" height="512" alt="Accuracy" src="https://github.com/user-attachments/assets/65c294d1-b563-4fce-97b9-744993f7adcd" />

<img width="731" height="504" alt="dice" src="https://github.com/user-attachments/assets/6e9435ce-4df8-42bb-834d-94ac567f42dd" />

<img width="737" height="508" alt="IoU" src="https://github.com/user-attachments/assets/009131ad-6817-4951-b974-0c7e5c2fd1e7" />

<img width="777" height="567" alt="Loss" src="https://github.com/user-attachments/assets/e714dcde-2570-416d-b90a-ebcd44add350" />

<img width="736" height="32" alt="Epoch" src="https://github.com/user-attachments/assets/30a6a51c-ccfc-492e-9276-465753dfe0a1" />

<img width="1681" height="901" alt="gradioarayüz" src="https://github.com/user-attachments/assets/b80a97aa-fe81-4b56-9ac3-b276cfd1de35" />


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
