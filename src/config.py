"""
FLAME Projesi - Merkezi Konfigürasyon Dosyası
Tüm yollar, hiperparametreler ve ayarlar burada tanımlanır.
"""

import os

# =============================================================================
# PROJE KÖK DİZİNİ
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# VERİ YOLLARI
# =============================================================================
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Sınıflandırma verileri (zip içinden okunacak)
CLASSIFICATION_TRAIN_ZIP = os.path.join(DATA_DIR, "Training.zip")
CLASSIFICATION_TEST_ZIP = os.path.join(DATA_DIR, "Test.zip")

# Segmentasyon verileri (zip içinden okunacak)
SEGMENTATION_IMAGES_ZIP = os.path.join(DATA_DIR, "Images.zip")
SEGMENTATION_MASKS_ZIP = os.path.join(DATA_DIR, "Masks.zip")

# =============================================================================
# ÇIKTI DİZİNLERİ
# =============================================================================
TENSORBOARD_DIR = os.path.join(PROJECT_ROOT, "runs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# =============================================================================
# VERİ AYARLARI
# =============================================================================
# Tüm verinin kaçta kaçı kullanılsın (1.0 = tamamı, 0.2 = beşte biri)
DATA_FRACTION = 0.5

# Segmentasyon verisi train/val bölme oranı
SEG_VAL_SPLIT = 0.2

# =============================================================================
# SINIFLANDIRMA HİPERPARAMETRELERİ
# =============================================================================
CLF_IMAGE_SIZE = 254            # Training görselleri zaten 254x254
CLF_BATCH_SIZE = 32
CLF_NUM_EPOCHS = 5
CLF_LEARNING_RATE = 1e-4
CLF_NUM_CLASSES = 2             # Fire, No_Fire
CLF_NUM_WORKERS = 0             # Windows için 0 güvenli

# =============================================================================
# SEGMENTASYON HİPERPARAMETRELERİ
# =============================================================================
SEG_IMAGE_SIZE = 256            # 3840x2160 → 256x256'ya resize edilecek
SEG_BATCH_SIZE = 8
SEG_NUM_EPOCHS = 25
SEG_LEARNING_RATE = 1e-4
SEG_NUM_CLASSES = 1             # Binary segmentasyon (yangın var/yok)
SEG_NUM_WORKERS = 0

# =============================================================================
# TERMAL GÖRÜNTÜ AYARLARI (İLERİ AŞAMA - ŞİMDİLİK KAPALI)
# =============================================================================
USE_THERMAL = False             # True yapıldığında termal kanal aktif olur
THERMAL_IMAGES_ZIP = os.path.join(DATA_DIR, "Thermal.zip")  # Placeholder
THERMAL_CHANNELS = 1            # Termal görüntü kanal sayısı

# =============================================================================
# GENEL MODEL AYARLARI
# =============================================================================
DEVICE = "cuda"                 # Eğitimde otomatik kontrol edilecek
RANDOM_SEED = 42
