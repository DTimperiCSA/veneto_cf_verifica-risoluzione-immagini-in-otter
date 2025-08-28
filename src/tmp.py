import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.segmentation.unet import UNet
from src.paths import *
from src.image_utils import *

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



# ========= RUN =========
if __name__ == "__main__":
    processed_count = 0
    MAX_IMAGES = 100

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

        try:
            # Run segmentation only on this one
            image, mask = predict_mask(candidate)
            print(f"[DEBUG] Predicted mask. Image shape: {image.shape}, Mask shape: {mask.shape}")

            # Save mask
            mask_out_path = OUTPUT_MASKS / f"{processed_count}_{candidate.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)
            print(f"[DEBUG] Mask saved to: {mask_out_path}")

            # Bounding box
            bbox = get_bbox_from_mask(mask)
            bbox_img = image.copy()
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)
                chromatic_band_dim_px = measure_chromatic_band_dimension(candidate, model)

                bin_img = binaryize_image(candidate)
                bin_img_dim_px = measure_document_from_binary(bin_img)

                img_long_side_px, img_short_side_px = max(bin_img_dim_px), min(bin_img_dim_px)
                chromatic_band_long_side_px, chromatic_band_short_side_px = max(chromatic_band_dim_px), min(chromatic_band_dim_px)

                scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

                img_long_side_mm = img_long_side_px * scale_factor
                img_short_side_mm = img_short_side_px * scale_factor

                print("Measueres")
                print(f"chromatic px: {chromatic_band_long_side_px} ")
                print(f"img px: {img_long_side_px} ")
                print(f"chromatic mm: {CHROMATIC_BAND_MM} ")
                print(f"img mm: {img_long_side_mm} ")
                print(f"scale factor: {scale_factor}")

                width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
                height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

                print(f"Dimensioni stimate: ")
                print(f"{img_long_side_mm:.2f} minore di {A4_HEIGHT_MM}? {'✅' if height_ok else '❌'}")
                print(f"{img_short_side_mm:.2f} minore di {A4_WIDTH_MM}? {'✅' if width_ok else '❌'}")
                print(f"Il file è un A4? {'✅' if (width_ok and height_ok) else '❌'}")

                ppi = 600

                if width_ok and height_ok:
                    ppi = 400
                
                print(f"PPI stimati: {ppi}")
            else:
                print("[DEBUG] No object found in mask.")

            bbox_out_path = OUTPUT_BBOX / f"{processed_count}_{candidate.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Image with bounding box saved to: {bbox_out_path}")

            processed_count += 1
            if processed_count >= MAX_IMAGES:
                print(f"[DEBUG] Reached max limit of {MAX_IMAGES}. Stopping.")
                break

        except Exception as e:
            print(f"[ERROR] Error with {candidate}: {e}")
