"""
FLAME Projesi - Tek Görsel Tahmin Script'i
Eğitilmiş model ile tek bir görsel üzerinde tahmin yapar.

Kullanım:
    python src/predict.py --task classification --image path/to/image.jpg
    python src/predict.py --task segmentation --image path/to/image.jpg
    python src/predict.py --task segmentation --image path/to/image.jpg --output result.png
"""

import argparse
import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, CLF_IMAGE_SIZE, SEG_IMAGE_SIZE
from model import FlameClassifier, FlameUNet
from utils import load_checkpoint


CLASS_NAMES = ["No_Fire", "Fire"]


def setup_device():
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(image_path, image_size, normalize=False):
    """Görseli yükler ve model girdisine dönüştürür."""
    img = Image.open(image_path).convert("RGB")
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        )
    transform = transforms.Compose(transform_list)
    tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    return img, tensor


# =============================================================================
# SINIFLANDIRMA TAHMİNİ
# =============================================================================
def predict_classification(image_path, device):
    print(f"\nGörsel: {image_path}")
    print("-" * 50)

    img, tensor = load_image(image_path, CLF_IMAGE_SIZE, normalize=True)
    tensor = tensor.to(device)

    model = FlameClassifier().to(device)
    load_checkpoint(model, None, "best_classifier.pth")
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_label = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item() * 100

    print(f"\n  Tahmin  : {pred_label}")
    print(f"  Güven   : %{confidence:.1f}")
    print(f"  No_Fire : %{probs[0].item() * 100:.1f}")
    print(f"  Fire    : %{probs[1].item() * 100:.1f}")

    # Görselleştir
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img)
    color = "red" if pred_label == "Fire" else "green"
    ax.set_title(f"{pred_label} (%{confidence:.1f})", fontsize=16, color=color)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# SEGMENTASYON TAHMİNİ
# =============================================================================
def predict_segmentation(image_path, device, output_path=None):
    print(f"\nGörsel: {image_path}")
    print("-" * 50)

    img, tensor = load_image(image_path, SEG_IMAGE_SIZE)
    tensor = tensor.to(device)

    model = FlameUNet().to(device)
    load_checkpoint(model, None, "best_segmentation.pth")
    model.eval()

    with torch.no_grad():
        output = model(tensor)
        mask = torch.sigmoid(output).cpu().squeeze().numpy()
        binary_mask = (mask > 0.5).astype(np.float32)

    fire_ratio = binary_mask.mean() * 100
    print(f"\n  Yangın pikseli oranı: %{fire_ratio:.2f}")

    # Görselleştir: Orijinal | Maske | Overlay
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Orijinal
    img_resized = img.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE))
    axes[0].imshow(img_resized)
    axes[0].set_title("Orijinal Görüntü")
    axes[0].axis("off")

    # Tahmin maskesi
    axes[1].imshow(binary_mask, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title(f"Yangın Maskesi (%{fire_ratio:.1f})")
    axes[1].axis("off")

    # Overlay
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    overlay = img_np.copy()
    overlay[binary_mask > 0.5] = overlay[binary_mask > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Sonuç kaydedildi: {output_path}")
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FLAME Tek Görsel Tahmin")
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation"],
        required=True,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Tahmin yapılacak görsel dosyasının yolu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Sonuç görselini kaydetmek için dosya yolu (opsiyonel)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"HATA: Görsel bulunamadı: {args.image}")
        sys.exit(1)

    device = setup_device()
    print(f"Cihaz: {device}")

    if args.task == "classification":
        predict_classification(args.image, device)
    else:
        predict_segmentation(args.image, device, args.output)


if __name__ == "__main__":
    main()
