import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.segmentation.unet import UNet
from src.paths import *
from src.utils import *
import time
import shutil

# ========= CONFIG =========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5  # You can adjust this threshold depending on your model's output

TMP_SEGMENTATION_MASK_DIR.mkdir(parents=True, exist_ok=True)
TMP_SEGMENTATION_BBOX_DIR.mkdir(parents=True, exist_ok=True)
TMP_SEGMENTATION_ROT_BBOX_DIR.mkdir(parents=True, exist_ok=True)

COL_KP = (255, 0, 0)
template = cv2.imread(str(TEMPLATE_IMG_PATH))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

# ========= MODEL =========


# ========= TRANSFORM FOR INFERENCE =========
transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Resize image to the input size used for training (adjust if needed)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization (ImageNet)
])

# ========= TEMPLATE MATCHING =========
def find_chromatic_band_in_folder(
    folder: Path
) -> str | None:
    folder = Path(folder)
    template_path = TEMPLATE_IMG_PATH

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder {folder} does not exist or is not a directory.")
    if not template_path.exists():
        raise ValueError(f"Template {template_path} does not exist.")

    # Load template (grayscale)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template image: {template_path}")

    best_match = None
    best_val = -1
    total_images = 0

    # scale factors da provare (puoi regolare la lista)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for img_path in folder.iterdir():
        if img_path.is_file():
            if img_path.name.lower() == "thumbs.db":
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
                continue

            total_images += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            i_h, i_w = img.shape[:2]

            # prova template a scale diverse
            for scale in scales:
                new_w = int(template.shape[1] * scale)
                new_h = int(template.shape[0] * scale)

                if new_w <= 5 or new_h <= 5:
                    continue  # template troppo piccolo
                if i_h < new_h or i_w < new_w:
                    continue  # template più grande dell’immagine

                resized_template = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Template matching
                res = cv2.matchTemplate(img, resized_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                if max_val > best_val:
                    best_val = max_val
                    best_match = img_path

    # Soglia per considerare il match valido
    if best_match is not None and best_val > 0.5:
        print(f"✅ Banda trovata: {best_match} (score={best_val:.3f})")
        return Path(best_match)
    else:
        print(f"❌ Nessuna banda trovata (miglior score={best_val:.3f})")
        return None


# ========= PREDICT MASK =========
def safe_imread(path: Path, logger, retries=3, delay=0.5):
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    logger.log_failure(path.name, "imread", f"Cannot read image after {retries} attempts", str(path))
    return None

def predict_mask(image_path: Path, model, logger):
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(tensor)
            pred = torch.sigmoid(pred)
            pred = F.interpolate(pred, size=img.size[::-1], mode="bilinear", align_corners=False)
            pred = pred.squeeze().cpu().numpy()

        mask = (pred > THRESHOLD).astype(np.uint8) * 255
        return np.array(img), mask

    except Exception as e:
        logger.log_failure(image_path.name, "predict_mask", f"{e}\n{traceback.format_exc()}", str(image_path))
        return None, None

def get_bbox_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def measure_document_from_binary(binary_image_path: Path, logger) -> tuple[float, float] | None:
    """Return (long_side_px, short_side_px) from binary image contour."""
    img = safe_imread(binary_image_path, logger)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (_, _), (w, h), _ = rect
    return (max(w, h), min(w, h))

def binaryize_image(image_path: Path, logger, threshold: int = 50) -> Path | None:
    """Convert image to binary (thresholded) and save to OUTPUT_TMP_DIR."""
    valid, msg = is_valid_image_file(image_path)
    if not valid:
        print(f"⚠️ Invalid file: {image_path} | {msg}")
        return None

    img = safe_imread(image_path, logger)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    dest_folder = OUTPUT_TMP_DIR / image_path.parent.name
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / image_path.name

    cv2.imwrite(str(dest_path), binary)
    return dest_path

def analyze_chromatic_band(candidate: Path, unet_model, logger):
    try:
        # --- predizione mask ---
        image, mask = predict_mask(candidate, unet_model, logger)
        if image is None or mask is None:
            return None

        # --- salva mask ---
        TMP_SEGMENTATION_MASK_DIR.mkdir(parents=True, exist_ok=True)
        mask_out_path = TMP_SEGMENTATION_MASK_DIR / f"{candidate.parent.name}_{candidate.stem}_mask.png"
        cv2.imwrite(str(mask_out_path), mask)

        # --- bbox axis-aligned ---
        bbox = get_bbox_from_mask(mask)
        if not bbox:
            logger.log_failure(candidate.name, "bbox", "No object found in mask", str(candidate))
            return None

        TMP_SEGMENTATION_BBOX_DIR.mkdir(parents=True, exist_ok=True)
        bbox_img = image.copy()
        x, y, w, h = bbox
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), 6)
        bbox_out_path = TMP_SEGMENTATION_BBOX_DIR / f"{candidate.parent.name}_{candidate.stem}_bbox.png"
        cv2.imwrite(str(bbox_out_path), bbox_img)

        # --- rotated bbox (minAreaRect) per dimensione banda cromatica ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.log_failure(candidate.name, "minAreaRect", "No contours found", str(candidate))
            return None

        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        (_, _), (w_rect, h_rect), _ = rect
        chromatic_band_long_side_px, chromatic_band_short_side_px = max(w_rect, h_rect), min(w_rect, h_rect)

        TMP_SEGMENTATION_ROT_BBOX_DIR.mkdir(parents=True, exist_ok=True)
        rect_img = image.copy()
        box_points = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(rect_img, [box_points], 0, (0, 255, 0), 2)
        rect_out_path = TMP_SEGMENTATION_ROT_BBOX_DIR / f"{candidate.parent.name}_{candidate.stem}.png"
        cv2.imwrite(str(rect_out_path), rect_img)

        # --- dimensione documento da immagine binaria ---
        bin_img = binaryize_image(candidate, logger)
        bin_img_dim_px = measure_document_from_binary(bin_img, logger)
        if bin_img_dim_px is None:
            logger.log_failure(candidate.name, "analyze_chromatic_band", "Failed document dimension measurement", str(candidate))
            return None

        img_long_side_px, img_short_side_px = max(bin_img_dim_px), min(bin_img_dim_px)

        # --- calcolo scale factor e dimensioni mm ---
        scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px
        img_long_side_mm = img_long_side_px * scale_factor
        img_short_side_mm = img_short_side_px * scale_factor

        width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
        height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

        ppi = 400 if width_ok and height_ok else 600

        # --- ritorno dict pronto per JSON ---
        return {
            "mask_path": str(mask_out_path),
            "bbox_path": str(bbox_out_path),
            "rotated_bbox_path": str(rect_out_path),
            "bbox": bbox,
            "chromatic_band_px": (chromatic_band_long_side_px, chromatic_band_short_side_px),
            "img_px": (img_long_side_px, img_short_side_px),
            "img_mm": (img_long_side_mm, img_short_side_mm),
            "scale_factor": scale_factor,
            "is_A4": width_ok and height_ok,
            "ppi": ppi
        }

    except Exception as e:
        logger.log_failure(
            candidate.name,
            "analyze_chromatic_band",
            f"{e}\n{traceback.format_exc()}",
            str(candidate),
        )
        return None

