import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.segmentation.unet import UNet
from src.paths import *

# ========= CONFIG =========
INPUT_FOLDER = Path(CONSERVATORIO_DIR / "B001")
OUTPUT_MASKS = Path(OUTPUT_TMP_DIR / "unet_results" / "masks")
OUTPUT_BBOX = Path(OUTPUT_TMP_DIR / "unet_results" / "bbox")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5  # soglia per binarizzare la maschera

OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
OUTPUT_BBOX.mkdir(parents=True, exist_ok=True)

# ========= MODEL =========
# Assumo che tu abbia definito o caricato un modello UNet
# Esempio generico:
# model = UNet(n_channels=3, n_classes=1)
# model.load_state_dict(torch.load("unet.pth"))
# model.to(DEVICE).eval()
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(SAVE_PATH))
model.to(DEVICE).eval()

# Preprocessing immagini
transform = transforms.Compose([
    transforms.ToTensor(),  # converte in tensor [0,1]
])

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def is_valid_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def predict_mask(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = pred.squeeze().cpu().numpy()

    # Binarizza la maschera
    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    return np.array(img), mask

def get_bbox_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Trova il contorno più grande
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return (x, y, w, h)

def process_folder(folder: Path):
    for img_path in folder.rglob("*"):
        if not is_valid_image_file(img_path):
            continue


        try:
            image, mask = predict_mask(img_path)

            # Salva maschera binaria
            mask_out_path = OUTPUT_MASKS / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)

            # Disegna BBOX
            bbox_img = image.copy()
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(
                    bbox_img,
                    (x, y),
                    (x + w, y + h),
                    (0, 0, 255),  # rosso acceso
                    thickness=6    # molto spessa
                )

            # Salva immagine con BBOX
            bbox_out_path = OUTPUT_BBOX / f"{img_path.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

            print(f"✔ Processata {img_path.name}")
        except Exception as e:
            print(f"Errore con {img_path.name}: {e}")

# ========= RUN =========
if __name__ == "__main__":
    process_folder(INPUT_FOLDER)
