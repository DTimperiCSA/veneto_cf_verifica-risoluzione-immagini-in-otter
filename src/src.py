import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.segmentation.unet import UNet   # importa la tua classe definita in unet.py


# =====================================================
# Carica modello UNet
# =====================================================
def load_unet_model(model_path, device="cuda"):
    model = UNet(n_channels=3, n_classes=1).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# =====================================================
# Preprocessing + Segmentazione
# =====================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # adattare alla dimensione usata nel training
    transforms.ToTensor()
])

def segment_image(model, img_pil, device="cuda"):
    orig_w, orig_h = img_pil.size
    x = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)

    mask = pred.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)

    # 🔹 Riporta la maschera alla dimensione originale
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized


def clean_mask(mask):
    """Applica morfologia per rimuovere rumore"""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_bounding_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)  # (x,y,w,h)


# =====================================================
# Main test
# =====================================================



# ------------------------------
# Main test
# ------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from src.paths import *

    model_path = SAVE_PATH
    input_root = CONSERVATORIO_DIR / "B001"  # root folder with subfolders
    output_dir = OUTPUT_TMP_DIR / "unet_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Carica modello
    model = load_unet_model(model_path, device)

    # Trova tutte le immagini in input_root (ricorsivamente)
    exts = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    image_files = [p for p in input_root.rglob("*") if p.suffix.lower() in exts]

    print(f"🔍 Trovate {len(image_files)} immagini in {input_root} (incluse sottocartelle)")

    for img_path in image_files:
        print(f"➡️ Elaboro: {img_path}")
        img_pil = Image.open(img_path).convert("RGB")
        img_cv = cv2.imread(str(img_path))

        # Segmenta
        mask = segment_image(model, img_pil, device)
        mask = clean_mask(mask)

        # Salva maschera (mantieni struttura cartelle)
        rel_path = img_path.relative_to(input_root)
        out_mask_path = output_dir / rel_path
        out_mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_mask_path), (mask * 255).astype(np.uint8))

        # Trova bounding box
        bbox = get_bounding_box(mask)
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(img_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_cv, "UNet", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Salva immagine con box (mantieni struttura cartelle)
        out_img_path = output_dir / rel_path
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_img_path), img_cv)

    print(f"\n✅ Elaborazione completata. Risultati salvati in: {output_dir}")
