"""
FLAME Projesi - Eğitim Script'i
Sınıflandırma ve Segmentasyon eğitimi.
TensorBoard ile canlı metrik takibi.

Kullanım:
    python src/train.py --task classification
    python src/train.py --task segmentation
    python src/train.py --task both
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# src/ dizinini path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CLF_NUM_EPOCHS, CLF_LEARNING_RATE,
    SEG_NUM_EPOCHS, SEG_LEARNING_RATE,
    DEVICE, RANDOM_SEED, TENSORBOARD_DIR,
)
from dataset import get_classification_loaders, get_segmentation_loaders
from model import FlameClassifier, FlameUNet
from utils import compute_iou, compute_dice, compute_pixel_accuracy, save_checkpoint


def setup_device():
    """CUDA varsa GPU, yoksa CPU kullan."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CPU modunda çalışılıyor.")
    return device


# =============================================================================
# SINIFLANDIRMA EĞİTİMİ
# =============================================================================
def train_classification(device):
    print("\n" + "=" * 60)
    print("SINIFLANDIRMA EĞİTİMİ BAŞLIYOR")
    print("=" * 60)

    # Veri
    train_loader, test_loader = get_classification_loaders()

    # Model
    model = FlameClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CLF_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # TensorBoard
    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "classification"))

    best_acc = 0.0

    for epoch in range(1, CLF_NUM_EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CLF_NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_loss = running_loss / total
        train_acc = correct / total

        # --- VALIDATION (Test set üzerinde) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{CLF_NUM_EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        # --- TENSORBOARD LOG ---
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Best model kaydet
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch,
                            {"val_acc": val_acc, "val_loss": val_loss},
                            "best_classifier.pth")

    # Son model
    save_checkpoint(model, optimizer, CLF_NUM_EPOCHS,
                    {"val_acc": val_acc, "val_loss": val_loss},
                    "last_classifier.pth")

    writer.close()
    print(f"\nSınıflandırma tamamlandı! En iyi Val Accuracy: {best_acc:.4f}")
    return model


# =============================================================================
# SEGMENTASYON EĞİTİMİ
# =============================================================================
def train_segmentation(device):
    print("\n" + "=" * 60)
    print("SEGMENTASYON EĞİTİMİ BAŞLIYOR")
    print("=" * 60)

    # Veri
    train_loader, val_loader = get_segmentation_loaders()

    # Model
    model = FlameUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=SEG_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # TensorBoard
    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "segmentation"))

    best_iou = 0.0

    for epoch in range(1, SEG_NUM_EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_dice = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{SEG_NUM_EPOCHS} [Train]")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_iou += compute_iou(outputs, masks)
            running_dice += compute_dice(outputs, masks)
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{running_iou/n_batches:.4f}",
            )

        train_loss = running_loss / n_batches
        train_iou = running_iou / n_batches
        train_dice = running_dice / n_batches

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_pixel_acc = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{SEG_NUM_EPOCHS} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_iou += compute_iou(outputs, masks)
                val_dice += compute_dice(outputs, masks)
                val_pixel_acc += compute_pixel_accuracy(outputs, masks)
                val_batches += 1

        val_loss /= val_batches
        val_iou /= val_batches
        val_dice /= val_batches
        val_pixel_acc /= val_batches

        scheduler.step(val_loss)

        # --- TENSORBOARD LOG ---
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("IoU", {"train": train_iou, "val": val_iou}, epoch)
        writer.add_scalars("Dice", {"train": train_dice, "val": val_dice}, epoch)
        writer.add_scalar("Val/PixelAccuracy", val_pixel_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        # Epoch'ta bir örnek tahmin görseli logla
        if val_batches > 0:
            model.eval()
            with torch.no_grad():
                sample_imgs, sample_masks = next(iter(val_loader))
                sample_imgs = sample_imgs.to(device)
                sample_preds = torch.sigmoid(model(sample_imgs))
                # İlk 4 örneği TensorBoard'a yaz
                for i in range(min(4, sample_imgs.size(0))):
                    writer.add_image(f"Val/Image_{i}", sample_imgs[i].cpu(), epoch)
                    writer.add_image(f"Val/Mask_True_{i}", sample_masks[i], epoch)
                    writer.add_image(f"Val/Mask_Pred_{i}", (sample_preds[i].cpu() > 0.5).float(), epoch)

        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f} IoU={train_iou:.4f} | "
              f"Val Loss={val_loss:.4f} IoU={val_iou:.4f} Dice={val_dice:.4f}")

        # Best model kaydet
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch,
                            {"val_iou": val_iou, "val_dice": val_dice, "val_loss": val_loss},
                            "best_segmentation.pth")

    # Son model
    save_checkpoint(model, optimizer, SEG_NUM_EPOCHS,
                    {"val_iou": val_iou, "val_dice": val_dice, "val_loss": val_loss},
                    "last_segmentation.pth")

    writer.close()
    print(f"\nSegmentasyon tamamlandı! En iyi Val IoU: {best_iou:.4f}")
    return model


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLAME Eğitim Script'i")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation", "both"],
        default="both",
        help="Hangi görev eğitilecek (default: both)",
    )
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    device = setup_device()

    start_time = time.time()

    if args.task in ("classification", "both"):
        train_classification(device)

    if args.task in ("segmentation", "both"):
        train_segmentation(device)

    elapsed = time.time() - start_time
    print(f"\nToplam süre: {elapsed/60:.1f} dakika")


if __name__ == "__main__":
    main()
