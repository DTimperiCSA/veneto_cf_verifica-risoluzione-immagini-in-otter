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
INPUT_FOLDER = Path(CONSERVATORIO_DIR)
OUTPUT_DIR = Path(OUTPUT_TMP_DIR / "unet_segmentation")
OUTPUT_MASKS = OUTPUT_DIR / "masks"
OUTPUT_BBOX = OUTPUT_DIR / "bbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5  # You can adjust this threshold depending on your model's output

OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
OUTPUT_BBOX.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

COL_KP = (255, 0, 0)
template = cv2.imread(str(TEMPLATE_IMG_PATH))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

# ========= MODEL =========
model = UNet(n_channels=3, n_classes=1)
checkpoint = torch.load(
    SAVE_PATH,
    map_location=DEVICE  # or "cuda" if you want GPU
)

# load weights properly
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

# move to GPU (or CPU)
model = model.to(DEVICE)
print("Caricamento modello UNET")

# ========= TRANSFORM FOR INFERENCE =========
transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Resize image to the input size used for training (adjust if needed)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization (ImageNet)
])

# ========= TEMPLATE MATCHING =========
def contains_chromatic_band(image_path: Path, threshold: float = 0.7) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val >= threshold

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
    t_h, t_w = template.shape[:2]

    best_match = None
    best_val = -1
    total_images = 0
    skipped_small = 0

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
            if i_h < t_h or i_w < t_w:
                continue

            # Template matching
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_match = img_path

    if best_match is not None and best_val > 0.5:
        return Path(best_match)
    else:
        return None


# ========= PREDICT MASK =========
def safe_imread(path: Path, retries=3, delay=0.5):
    """Robust cv2.imread with retries."""
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    raise IOError(f"Cannot read image {path} after {retries} attempts")


def predict_mask(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        pred = F.interpolate(pred, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = pred.squeeze().cpu().numpy()

    # Debug: Show raw model output (before thresholding)
    print(f"Raw output (min, max): {pred.min()}, {pred.max()}")

    # Apply threshold to create a binary mask
    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    return np.array(img), mask

def get_bbox_from_mask(mask: np.ndarray):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h

def measure_chromatic_band_dimension(path_input: Path, input_size=(256, 256)):
    """
    Measure chromatic band dimensions in pixels using UNet segmentation.
    Saves a debug visualization to OUTPUT_TMP_DIR/segmentation_results.
    Returns (long_side, short_side) in px.
    """
    img = cv2.imread(str(path_input))
    if img is None:
        print(f"⚠️ Could not read {path_input}")
        return None
    orig_h, orig_w = img.shape[:2]

    img_resized = cv2.resize(img, input_size)
    img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)
        pred = torch.sigmoid(pred)
        mask = (pred > 0.5).float().cpu().numpy()[0, 0]

    mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"⚠️ No Tiffen ruler found in {path_input}.")
        return None

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect).astype(int)

    # Save debug visualization
    out_dir = OUTPUT_TMP_DIR / "segmentation_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path_input.stem}_segmented.png"
    debug = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), debug)

    return (max(w, h), min(w, h))

def measure_document_from_binary(binary_image_path: Path) -> tuple[float, float] | None:
    """Return (long_side_px, short_side_px) from binary image contour."""
    img = safe_imread(binary_image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (_, _), (w, h), _ = rect
    return (max(w, h), min(w, h))

def binaryize_image(image_path: Path, threshold: int = 50) -> Path | None:
    """Convert image to binary (thresholded) and save to OUTPUT_TMP_DIR."""
    valid, msg = is_valid_image_file(image_path)
    if not valid:
        print(f"⚠️ Invalid file: {image_path} | {msg}")
        return None

    img = safe_imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    dest_folder = OUTPUT_TMP_DIR / image_path.parent.name
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / image_path.name

    cv2.imwrite(str(dest_path), binary)
    return dest_path

def analyze_chromatic_band(candidate: Path):
    """
    Analizza un'immagine candidata:
    - Segmentazione con UNet
    - Salvataggio della mask
    - Calcolo bounding box
    - Stima dimensioni in mm e verifica A4
    - Stima PPI
    """
    try:
        # Run segmentation
        image, mask = predict_mask(candidate)
        print(f"[DEBUG] Predicted mask. Image shape: {image.shape}, Mask shape: {mask.shape}")

        # Save mask
        mask_out_path = OUTPUT_MASKS / f"{candidate.stem}_mask.png"
        cv2.imwrite(str(mask_out_path), mask)
        print(f"[DEBUG] Mask saved to: {mask_out_path}")

        # Bounding box
        bbox = get_bbox_from_mask(mask)
        bbox_img = image.copy()

        if not bbox:
            print("[DEBUG] No object found in mask.")
            return None

        # Disegno bbox
        x, y, w, h = bbox
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)

        # Salvataggio bbox image
        bbox_out_path = OUTPUT_BBOX / f"{candidate.stem}_bbox.png"
        cv2.imwrite(str(bbox_out_path), bbox_img)
        print(f"[DEBUG] Bounding box image saved to: {bbox_out_path}")

        # Misura bande cromatiche
        chromatic_band_dim_px = measure_chromatic_band_dimension(candidate)

        # Misura documento
        bin_img = binaryize_image(candidate)
        bin_img_dim_px = measure_document_from_binary(bin_img)

        img_long_side_px, img_short_side_px = max(bin_img_dim_px), min(bin_img_dim_px)
        chromatic_band_long_side_px, chromatic_band_short_side_px = (
            max(chromatic_band_dim_px), min(chromatic_band_dim_px)
        )

        # Calcolo fattore scala
        scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

        img_long_side_mm = img_long_side_px * scale_factor
        img_short_side_mm = img_short_side_px * scale_factor

        print("Measueres")
        print(f"chromatic px: {chromatic_band_long_side_px} ")
        print(f"img px: {img_long_side_px} ")
        print(f"chromatic mm: {CHROMATIC_BAND_MM} ")
        print(f"img mm: {img_long_side_mm} ")
        print(f"scale factor: {scale_factor}")

        # Check dimensioni A4
        width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
        height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

        print("Dimensioni stimate:")
        print(f"{img_long_side_mm:.2f} minore di {A4_HEIGHT_MM}? {'✅' if height_ok else '❌'}")
        print(f"{img_short_side_mm:.2f} minore di {A4_WIDTH_MM}? {'✅' if width_ok else '❌'}")
        print(f"Il file è un A4? {'✅' if (width_ok and height_ok) else '❌'}")

        # Stima PPI
        ppi = 600
        if width_ok and height_ok:
            ppi = 400
        print(f"PPI stimati: {ppi}")

        # Restituisce i risultati come dict
        return {
            "mask_path": mask_out_path,
            "bbox_path": bbox_out_path,
            "bbox": bbox,
            "chromatic_band_px": (chromatic_band_long_side_px, chromatic_band_short_side_px),
            "img_px": (img_long_side_px, img_short_side_px),
            "img_mm": (img_long_side_mm, img_short_side_mm),
            "scale_factor": scale_factor,
            "is_A4": width_ok and height_ok,
            "ppi": ppi
        }

    except Exception as e:
        print(f"[ERROR] Failed to analyze {candidate}: {e}")
        return None



# ========= RUN =========
if __name__ == "__main__":
    # Walk folders recursively
    for folder in INPUT_FOLDER.rglob("*"):
        if not folder.is_dir():
            continue

        # Collect all valid images in this folder
        images_in_folder = [p for p in folder.glob("*") if p.suffix.lower() in VALID_EXTS]
        if not images_in_folder:
            print(f"[DEBUG] Skipping empty folder: {folder}")
            continue

        print(f"[DEBUG] Checking folder: {folder}, {len(images_in_folder)} images")

        # Find the one image with chromatic band
        candidate = find_chromatic_band_in_folder(folder)
        if candidate is None:
            print(f"[DEBUG] No chromatic band found in {folder}, skipping.")
            continue

        print(f"[DEBUG] Found chromatic band in: {candidate}")

        res = analyze_chromatic_band(candidate)
        print(res)
