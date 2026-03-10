"""
FLAME Projesi - Dataset Modülü
Sınıflandırma ve Segmentasyon için veri yükleyiciler.
Termal görüntü entegrasyonu için altyapı hazır (şimdilik kapalı).
"""

import io
import zipfile
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from config import (
    CLASSIFICATION_TRAIN_ZIP, CLASSIFICATION_TEST_ZIP,
    SEGMENTATION_IMAGES_ZIP, SEGMENTATION_MASKS_ZIP,
    CLF_IMAGE_SIZE, CLF_BATCH_SIZE, CLF_NUM_WORKERS,
    SEG_IMAGE_SIZE, SEG_BATCH_SIZE, SEG_NUM_WORKERS,
    DATA_FRACTION, SEG_VAL_SPLIT, RANDOM_SEED,
    USE_THERMAL, THERMAL_IMAGES_ZIP, THERMAL_CHANNELS,
)


# =============================================================================
# SINIFLANDIRMA DATASET
# =============================================================================
class FlameClassificationDataset(Dataset):
    """
    ZIP içinden Fire/No_Fire görüntülerini okur.
    Label: 1 = Fire, 0 = No_Fire
    """

    def __init__(self, zip_path, transform=None, fraction=1.0,
                 use_thermal=False, thermal_zip_path=None):
        """
        Args:
            zip_path: Training.zip veya Test.zip yolu
            transform: torchvision transforms
            fraction: Verinin ne kadarı kullanılacak (0.0 - 1.0)
            use_thermal: Termal kanal aktif mi (ilerisi için)
            thermal_zip_path: Termal zip dosyası yolu (ilerisi için)
        """
        self.zip_path = zip_path
        self.transform = transform
        self.use_thermal = use_thermal
        self.thermal_zip_path = thermal_zip_path

        # ZIP'i aç ve dosya listesini çıkar
        self.zip_file = zipfile.ZipFile(zip_path, "r")
        all_names = self.zip_file.namelist()

        self.samples = []  # (dosya_adı, label)
        for name in all_names:
            if not name.endswith(".jpg"):
                continue
            if "/Fire/" in name:
                self.samples.append((name, 1))
            elif "/No_Fire/" in name:
                self.samples.append((name, 0))

        # Fraction uygula (tekrarlanabilir shuffle)
        if fraction < 1.0:
            random.seed(RANDOM_SEED)
            random.shuffle(self.samples)
            n = max(1, int(len(self.samples) * fraction))
            self.samples = self.samples[:n]

        # Termal zip (ilerisi için)
        self.thermal_zip = None
        if self.use_thermal and thermal_zip_path:
            self.thermal_zip = zipfile.ZipFile(thermal_zip_path, "r")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, label = self.samples[idx]

        # RGB görüntüyü oku
        img_bytes = self.zip_file.read(name)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # --- TERMAL KANAL (İLERİSİ İÇİN) ---
        if self.use_thermal and self.thermal_zip is not None:
            # İleride: termal dosya adını RGB'den türet ve oku
            # thermal_name = name.replace("Training/", "Thermal/").replace(".jpg", ".png")
            # thermal_bytes = self.thermal_zip.read(thermal_name)
            # thermal_img = Image.open(io.BytesIO(thermal_bytes)).convert("L")
            # thermal_tensor = T.ToTensor()(thermal_img)  # [1, H, W]
            # image = torch.cat([image, thermal_tensor], dim=0)  # [4, H, W]
            pass

        return image, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        if hasattr(self, "zip_file") and self.zip_file:
            self.zip_file.close()
        if hasattr(self, "thermal_zip") and self.thermal_zip:
            self.thermal_zip.close()


# =============================================================================
# SEGMENTASYON DATASET
# =============================================================================
class FlameSegmentationDataset(Dataset):
    """
    ZIP'lerden RGB görüntü + binary maske çiftlerini okur.
    Maske değerleri: 0 = arka plan, 1 = yangın
    """

    def __init__(self, image_names, images_zip_path, masks_zip_path,
                 image_size=256, augment=False,
                 use_thermal=False, thermal_zip_path=None):
        """
        Args:
            image_names: Kullanılacak dosya numaraları listesi (eşleşme için)
            images_zip_path: Images.zip yolu
            masks_zip_path: Masks.zip yolu
            image_size: Çıktı boyutu (kare)
            augment: Veri artırma uygulansın mı
            use_thermal: Termal kanal aktif mi
            thermal_zip_path: Termal zip yolu
        """
        self.image_names = image_names
        self.image_size = image_size
        self.augment = augment
        self.use_thermal = use_thermal

        self.images_zip = zipfile.ZipFile(images_zip_path, "r")
        self.masks_zip = zipfile.ZipFile(masks_zip_path, "r")

        self.thermal_zip = None
        if self.use_thermal and thermal_zip_path:
            self.thermal_zip = zipfile.ZipFile(thermal_zip_path, "r")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        num = self.image_names[idx]
        img_path = f"Images/image_{num}.jpg"
        mask_path = f"Masks/image_{num}.png"

        # RGB görüntü
        img = Image.open(io.BytesIO(self.images_zip.read(img_path))).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Maske
        mask = Image.open(io.BytesIO(self.masks_zip.read(mask_path))).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Veri artırma (aynı transform image ve mask'e uygulanmalı)
        if self.augment:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Tensor'a çevir
        img_tensor = T.ToTensor()(img)                         # [3, H, W], 0-1 arası
        mask_np = np.array(mask, dtype=np.float32)
        mask_np = np.clip(mask_np, 0, 1)                       # Zaten 0/1
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)   # [1, H, W]

        # --- TERMAL KANAL (İLERİSİ İÇİN) ---
        if self.use_thermal and self.thermal_zip is not None:
            # thermal_path = f"Thermal/image_{num}.png"
            # thermal = Image.open(io.BytesIO(self.thermal_zip.read(thermal_path))).convert("L")
            # thermal = thermal.resize((self.image_size, self.image_size), Image.BILINEAR)
            # thermal_tensor = T.ToTensor()(thermal)  # [1, H, W]
            # img_tensor = torch.cat([img_tensor, thermal_tensor], dim=0)  # [4, H, W]
            pass

        return img_tensor, mask_tensor

    def __del__(self):
        for zf in [self.images_zip, self.masks_zip, self.thermal_zip]:
            if zf is not None:
                zf.close()


# =============================================================================
# DATALOADER FABRİKA FONKSİYONLARI
# =============================================================================
def get_classification_loaders():
    """Sınıflandırma için train ve test DataLoader döner."""

    train_transform = T.Compose([
        T.Resize((CLF_IMAGE_SIZE, CLF_IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize((CLF_IMAGE_SIZE, CLF_IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = FlameClassificationDataset(
        zip_path=CLASSIFICATION_TRAIN_ZIP,
        transform=train_transform,
        fraction=DATA_FRACTION,
        use_thermal=USE_THERMAL,
        thermal_zip_path=THERMAL_IMAGES_ZIP if USE_THERMAL else None,
    )

    test_dataset = FlameClassificationDataset(
        zip_path=CLASSIFICATION_TEST_ZIP,
        transform=test_transform,
        fraction=DATA_FRACTION,
        use_thermal=USE_THERMAL,
        thermal_zip_path=THERMAL_IMAGES_ZIP if USE_THERMAL else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CLF_BATCH_SIZE,
        shuffle=True,
        num_workers=CLF_NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CLF_BATCH_SIZE,
        shuffle=False,
        num_workers=CLF_NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[Sınıflandırma] Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader


def get_segmentation_loaders():
    """Segmentasyon için train ve val DataLoader döner."""

    # Eşleşen dosya numaralarını bul
    with zipfile.ZipFile(SEGMENTATION_IMAGES_ZIP, "r") as zf:
        all_nums = []
        for name in zf.namelist():
            if name.endswith(".jpg"):
                num = int(name.split("image_")[1].split(".")[0])
                all_nums.append(num)

    random.seed(RANDOM_SEED)
    random.shuffle(all_nums)

    # Fraction uygula
    n_total = max(1, int(len(all_nums) * DATA_FRACTION))
    all_nums = all_nums[:n_total]

    # Train/Val split
    n_val = max(1, int(len(all_nums) * SEG_VAL_SPLIT))
    val_nums = all_nums[:n_val]
    train_nums = all_nums[n_val:]

    thermal_zip = THERMAL_IMAGES_ZIP if USE_THERMAL else None

    train_dataset = FlameSegmentationDataset(
        image_names=train_nums,
        images_zip_path=SEGMENTATION_IMAGES_ZIP,
        masks_zip_path=SEGMENTATION_MASKS_ZIP,
        image_size=SEG_IMAGE_SIZE,
        augment=True,
        use_thermal=USE_THERMAL,
        thermal_zip_path=thermal_zip,
    )

    val_dataset = FlameSegmentationDataset(
        image_names=val_nums,
        images_zip_path=SEGMENTATION_IMAGES_ZIP,
        masks_zip_path=SEGMENTATION_MASKS_ZIP,
        image_size=SEG_IMAGE_SIZE,
        augment=False,
        use_thermal=USE_THERMAL,
        thermal_zip_path=thermal_zip,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=SEG_BATCH_SIZE,
        shuffle=True,
        num_workers=SEG_NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=SEG_BATCH_SIZE,
        shuffle=False,
        num_workers=SEG_NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[Segmentasyon] Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_loader, val_loader
