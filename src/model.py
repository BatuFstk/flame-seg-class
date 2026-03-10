"""
FLAME Projesi - Model Modülü
Sınıflandırma: ResNet18 tabanlı transfer learning
Segmentasyon: Hafif U-Net mimarisi
Termal kanal desteği: in_channels parametresiyle kontrol edilir.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config import CLF_NUM_CLASSES, SEG_NUM_CLASSES, USE_THERMAL, THERMAL_CHANNELS


def _get_in_channels():
    """RGB (3) + opsiyonel termal kanal sayısını döner."""
    return 3 + THERMAL_CHANNELS if USE_THERMAL else 3


# =============================================================================
# SINIFLANDIRMA MODELİ
# =============================================================================
class FlameClassifier(nn.Module):
    """
    ResNet18 tabanlı yangın sınıflandırıcı.
    - Pretrained ağırlıklar kullanılır (transfer learning)
    - İlk conv katmanı in_channels'a göre uyarlanır (termal destek)
    - Son FC katmanı 2 sınıfa (Fire/No_Fire) ayarlanır
    """

    def __init__(self, num_classes=CLF_NUM_CLASSES, in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = _get_in_channels()

        # Pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # İlk conv katmanını uyarla (RGB=3 dışı kanal sayısı için)
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )
            # RGB ağırlıklarını kopyala, ekstra kanalları sıfırla
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old_conv.weight
                self.backbone.conv1.weight[:, 3:] = 0.0

        # Son FC katmanını değiştir
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# SEGMENTASYON MODELİ - U-Net
# =============================================================================
class ConvBlock(nn.Module):
    """Çift konvolüsyon bloğu: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FlameUNet(nn.Module):
    """
    Hafif U-Net segmentasyon modeli.
    - Encoder: 4 seviye downsampling
    - Decoder: 4 seviye upsampling + skip connections
    - Çıktı: 1 kanal (binary yangın maskesi)
    - in_channels: 3 (RGB) veya 4 (RGB + Termal)
    """

    def __init__(self, in_channels=None, num_classes=SEG_NUM_CLASSES):
        super().__init__()
        if in_channels is None:
            in_channels = _get_in_channels()

        # Encoder (downsampling)
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder (upsampling)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Son katman
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)
