import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.segmentation.unet import UNet
from src.paths import *

# ========= CONFIG =========
INPUT_FOLDER = Path(CONSERVATORIO_DIR / "B001")
OUTPUT_DIR = Path(OUTPUT_TMP_DIR / "unet_segmentation")
OUTPUT_MASKS = OUTPUT_DIR / "masks"
OUTPUT_BBOX = OUTPUT_DIR / "bbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5

OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
OUTPUT_BBOX.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ========= MODEL =========
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(SAVE_PATH))
model.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

# ========= TEMPLATE MATCHING =========
template = cv2.imread(str(TEMPLATE_IMG_PATH))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

def contains_chromatic_band(image_path: Path, threshold: float = 0.7) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val >= threshold

# ========= PREDICT MASK =========
def predict_mask(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = pred.squeeze().cpu().numpy()

    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    return np.array(img), mask

def get_bbox_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Trova il contorno più grande
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

# ========= PROCESS FOLDER =========
def process_folder(folder: Path):
    for img_path in folder.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue

        try:
            if not contains_chromatic_band(img_path):                
                continue

            image, mask = predict_mask(img_path)

            mask_out_path = OUTPUT_MASKS / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)

            bbox_img = image.copy()
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)

            bbox_out_path = OUTPUT_BBOX / f"{img_path.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

            print(f"✔ Processata {img_path.name}")
        except Exception as e:
            print(f"Errore con {img_path.name}: {e}")

# ========= RUN =========
# ========= RUN =========
if __name__ == "__main__":
    MAX_IMAGES = 100
    processed_count = 0

    # Naviga tutte le sottocartelle e prende tutti i file immagine
    all_images = [p for p in INPUT_FOLDER.rglob("*") if p.suffix.lower() in VALID_EXTS]
    print(f"[DEBUG] Totale immagini trovate: {len(all_images)}")

    all_images = all_images[:100]

    for img_path in all_images:
        if processed_count >= MAX_IMAGES:
            print(f"[DEBUG] Raggiunto limite di {MAX_IMAGES} immagini. Stop.")
            break

        print(f"[DEBUG] Elaborando: {img_path}")

        try:
            image, mask = predict_mask(img_path)
            print(f"[DEBUG] Maschera predetta. Shape immagine: {image.shape}, Shape mask: {mask.shape}")

            # salva maschera
            mask_out_path = OUTPUT_MASKS / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)
            print(f"[DEBUG] Maschera salvata in: {mask_out_path}")

            # salva bounding box
            bbox_img = image.copy()
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)
                print(f"[DEBUG] Bounding box: x={x}, y={y}, w={w}, h={h}")
            else:
                print(f"[DEBUG] Nessun oggetto trovato nella maschera.")

            bbox_out_path = OUTPUT_BBOX / f"{img_path.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Immagine con bbox salvata in: {bbox_out_path}")

            print(f"✔ Processata {img_path.name}")
            processed_count += 1

        except Exception as e:
            print(f"[ERROR] Errore con {img_path.name}: {e}")

