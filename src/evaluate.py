"""
FLAME Projesi - Değerlendirme Script'i
Eğitilmiş modeli yükleyip test metrikleri hesaplar.

Kullanım:
    python src/evaluate.py --task classification
    python src/evaluate.py --task segmentation
"""

import argparse
import os
import sys

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, CHECKPOINT_DIR
from dataset import get_classification_loaders, get_segmentation_loaders
from model import FlameClassifier, FlameUNet
from utils import (
    compute_iou, compute_dice, compute_pixel_accuracy,
    load_checkpoint, visualize_segmentation,
)


def setup_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# SINIFLANDIRMA DEĞERLENDİRME
# =============================================================================
def evaluate_classification(device):
    print("\n" + "=" * 60)
    print("SINIFLANDIRMA DEĞERLENDİRME")
    print("=" * 60)

    _, test_loader = get_classification_loaders()

    model = FlameClassifier().to(device)
    load_checkpoint(model, None, "best_classifier.pth")
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n--- Classification Report ---")
    print(classification_report(
        all_labels, all_preds,
        target_names=["No_Fire", "Fire"],
    ))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    acc = (all_preds == all_labels).mean()
    print(f"\nGenel Accuracy: {acc:.4f}")


# =============================================================================
# SEGMENTASYON DEĞERLENDİRME
# =============================================================================
def evaluate_segmentation(device):
    print("\n" + "=" * 60)
    print("SEGMENTASYON DEĞERLENDİRME")
    print("=" * 60)

    _, val_loader = get_segmentation_loaders()

    model = FlameUNet().to(device)
    load_checkpoint(model, None, "best_segmentation.pth")
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_pixel_acc = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            total_iou += compute_iou(outputs, masks)
            total_dice += compute_dice(outputs, masks)
            total_pixel_acc += compute_pixel_accuracy(outputs, masks)
            n_batches += 1

    avg_iou = total_iou / n_batches
    avg_dice = total_dice / n_batches
    avg_pixel_acc = total_pixel_acc / n_batches

    print(f"\n--- Segmentasyon Metrikleri ---")
    print(f"  IoU          : {avg_iou:.4f}")
    print(f"  Dice         : {avg_dice:.4f}")
    print(f"  Pixel Acc    : {avg_pixel_acc:.4f}")

    # Örnek görselleştirmeler kaydet
    print("\nÖrnek tahminler kaydediliyor...")
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images_dev = images.to(device)
        preds = model(images_dev)

        vis_dir = os.path.join(CHECKPOINT_DIR, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        for i in range(min(4, images.size(0))):
            save_path = os.path.join(vis_dir, f"seg_sample_{i}.png")
            visualize_segmentation(
                images[i], masks[i], preds[i].cpu(),
                save_path=save_path,
            )
        print(f"  Görseller kaydedildi: {vis_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLAME Değerlendirme")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        required=True,
    )
    args = parser.parse_args()

    device = setup_device()

    if args.task == "classification":
        evaluate_classification(device)
    else:
        evaluate_segmentation(device)


if __name__ == "__main__":
    main()
