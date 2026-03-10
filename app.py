"""
FLAME Projesi - Web Arayüzü (Gradio)
Görsel yükleyip yangın tespiti yapın.

Kullanım:
    python app.py
"""

import os
import sys

import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DEVICE, CLF_IMAGE_SIZE, SEG_IMAGE_SIZE
from model import FlameClassifier, FlameUNet
from utils import load_checkpoint

# =============================================================================
# CİHAZ & MODEL YÜKLEME (bir kez yüklenir)
# =============================================================================
device = torch.device("cuda" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu")

clf_model = FlameClassifier().to(device)
load_checkpoint(clf_model, None, "best_classifier.pth")
clf_model.eval()

seg_model = FlameUNet().to(device)
load_checkpoint(seg_model, None, "best_segmentation.pth")
seg_model.eval()


# =============================================================================
# SINIFLANDIRMA
# =============================================================================
def classify_image(img):
    if img is None:
        return None, "Lütfen bir görsel yükleyin."

    img = Image.fromarray(img).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((CLF_IMAGE_SIZE, CLF_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = clf_model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    return {
        "Fire": float(probs[1]),
        "No Fire": float(probs[0]),
    }


# =============================================================================
# SEGMENTASYON
# =============================================================================
def segment_image(img):
    if img is None:
        return None, "Lütfen bir görsel yükleyin."

    img = Image.fromarray(img).convert("RGB")
    img_resized = img.resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE))

    transform = transforms.Compose([
        transforms.Resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = seg_model(tensor)
        mask = torch.sigmoid(output).cpu().squeeze().numpy()
        binary_mask = (mask > 0.5).astype(np.float32)

    # Overlay oluştur
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    overlay = img_np.copy()
    overlay[binary_mask > 0.5] = overlay[binary_mask > 0.5] * 0.5 + np.array([1, 0, 0]) * 0.5
    overlay = (overlay * 255).astype(np.uint8)

    fire_ratio = binary_mask.mean() * 100
    return overlay, f"Yangın oranı: %{fire_ratio:.1f}"


# =============================================================================
# GRADIO ARAYÜZÜ
# =============================================================================
with gr.Blocks(title="FLAME - Yangın Tespit Sistemi") as demo:
    gr.Markdown("# FLAME - Yangın Tespit Sistemi")
    gr.Markdown("Görsel yükleyerek yangın sınıflandırma ve segmentasyon yapın.")

    with gr.Tab("Sınıflandırma"):
        with gr.Row():
            clf_input = gr.Image(label="Görsel Yükle")
            clf_output = gr.Label(label="Sonuç", num_top_classes=2)
        clf_btn = gr.Button("Analiz Et")
        clf_btn.click(fn=classify_image, inputs=clf_input, outputs=clf_output)

    with gr.Tab("Segmentasyon"):
        with gr.Row():
            seg_input = gr.Image(label="Görsel Yükle")
            seg_output = gr.Image(label="Yangın Maskesi")
        seg_info = gr.Textbox(label="Bilgi")
        seg_btn = gr.Button("Analiz Et")
        seg_btn.click(fn=segment_image, inputs=seg_input, outputs=[seg_output, seg_info])

if __name__ == "__main__":
    demo.launch()
