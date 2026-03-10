"""
FLAME Projesi - Yardımcı Fonksiyonlar
Metrikler, checkpoint yönetimi, görselleştirme.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import CHECKPOINT_DIR


# =============================================================================
# METRİKLER
# =============================================================================
def compute_iou(preds, targets, threshold=0.5):
    """
    Binary IoU (Intersection over Union) hesaplar.
    Args:
        preds: Model çıktısı (logits), [B, 1, H, W]
        targets: Ground truth maske, [B, 1, H, W]
        threshold: Sigmoid sonrası eşik değeri
    Returns:
        IoU skoru (float)
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    if union == 0:
        return 1.0  # Her ikisi de boşsa perfect match
    return (intersection / union).item()


def compute_dice(preds, targets, threshold=0.5):
    """
    Dice katsayısı (F1 score'un piksel versiyonu).
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    total = preds.sum() + targets.sum()

    if total == 0:
        return 1.0
    return (2.0 * intersection / total).item()


def compute_pixel_accuracy(preds, targets, threshold=0.5):
    """Piksel bazlı doğruluk."""
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    correct = (preds == targets).sum()
    total = targets.numel()
    return (correct / total).item()


# =============================================================================
# CHECKPOINT YÖNETİMİ
# =============================================================================
def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """Model checkpoint'ini kaydeder."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)
    print(f"  Checkpoint kaydedildi: {path}")


def load_checkpoint(model, optimizer, filename):
    """Checkpoint'ten model yükler."""
    path = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  Checkpoint yüklendi: {path} (Epoch {checkpoint['epoch']})")
    return checkpoint["epoch"], checkpoint["metrics"]


# =============================================================================
# GÖRSELLEŞTİRME
# =============================================================================
def visualize_segmentation(image, mask_true, mask_pred, save_path=None):
    """
    Segmentasyon sonuçlarını 3'lü görselleştirir:
    [Orijinal Görüntü] [Gerçek Maske] [Tahmin Maske]
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Orijinal görüntü (tensor → numpy)
    if isinstance(image, torch.Tensor):
        img = image.cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
    else:
        img = image

    axes[0].imshow(img)
    axes[0].set_title("RGB Görüntü")
    axes[0].axis("off")

    # Gerçek maske
    if isinstance(mask_true, torch.Tensor):
        mask_true = mask_true.cpu().squeeze().numpy()
    axes[1].imshow(mask_true, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Gerçek Maske")
    axes[1].axis("off")

    # Tahmin maske
    if isinstance(mask_pred, torch.Tensor):
        mask_pred = torch.sigmoid(mask_pred).cpu().squeeze().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.float32)
    axes[2].imshow(mask_pred, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title("Tahmin Maske")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
