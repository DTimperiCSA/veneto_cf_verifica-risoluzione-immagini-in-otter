"""
Image processing utilities for detecting and measuring the Tiffen chromatic band
and documents in scanned images.
"""

# ========== STANDARD LIBRARIES ==========
import os
import time
import shutil
import traceback
from pathlib import Path
from typing import Tuple

# ========== THIRD-PARTY LIBRARIES ==========
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# ========== PROJECT MODULES ==========
from src.paths import *
from src.config import *
from src.image_utils import *
from src.segmentation.unet import UNet

# ========== CONSTANTS ==========
COL_KP = (255, 0, 0)

# Load template once at import
template = cv2.imread(str(TEMPLATE_IMG_PATH))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5 
MODEL = None  # Global cached UNet model

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((480, 480)),  
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================================
#                              SAFE IO HELPERS
# ============================================================================

def safe_imread(path: Path, retries=3, delay=0.5):
    """Robust cv2.imread with retries."""
    for attempt in range(retries):
        img = cv2.imread(str(path))
        if img is not None:
            return img
        time.sleep(delay)
    raise IOError(f"Cannot read image {path} after {retries} attempts")


def safe_copy(src: Path, dst: Path, retries=3, delay=0.5):
    """Robust shutil.copy2 with retries (handles Windows locks)."""
    for attempt in range(retries):
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        except (PermissionError, OSError) as e:
            if hasattr(e, 'winerror') and e.winerror in (32, 1224):  # file locked
                time.sleep(delay)
            else:
                raise
    raise IOError(f"Cannot copy file {src} -> {dst} after {retries} attempts")


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array (H x W x C) to a PIL Image."""
    return Image.fromarray(array)

# ============================================================================
#                         IMAGE VALIDATION & BINARY
# ============================================================================

def is_valid_image_file(file_path: Path) -> Tuple[bool, str]:
    """
    Deep validation for image files:
    - Checks existence, file type, and file size
    - Tries Image.open, verify, and load
    Returns (True, "") if valid, otherwise (False, error message).
    """
    try:
        if not file_path.exists():
            return False, "File not found"
        if not file_path.is_file():
            return False, "Path is not a file"

        # File size sanity check
        try:
            file_size = file_path.stat().st_size
            if file_size < 10_000:
                return False, f"File too small ({file_size} bytes), likely corrupted"
        except Exception as e:
            return False, f"Error reading file size: {e}"

        # Open & verify
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            return False, f"[VERIFY FAIL] {e}"

        # Reopen & load pixels
        try:
            img = Image.open(file_path)
            img.load()
        except Exception as e:
            return False, f"[LOAD FAIL] {e}"

        return True, ""

    except FileNotFoundError:
        return False, "FileNotFoundError"
    except PermissionError:
        return False, "PermissionError"
    except OSError as e:
        return False, f"OSError: {e}"
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return False, f"[UNEXPECTED] {e} | Traceback: {tb}"


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

# ============================================================================
#                        TEMPLATE & BAND DETECTION
# ============================================================================

def contains_chromatic_band(image_path: Path, threshold: float = 0.7) -> bool:
    """Check if template (chromatic band) is present in image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= threshold


def find_chromatic_band_in_folder(folder: Path) -> str | None:
    """
    Find best-matching image containing the chromatic band in a folder.
    Returns path string or None.
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder: {folder}")
    if not TEMPLATE_IMG_PATH.exists():
        raise ValueError(f"Template missing: {TEMPLATE_IMG_PATH}")

    template = cv2.imread(str(TEMPLATE_IMG_PATH), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template: {TEMPLATE_IMG_PATH}")
    t_h, t_w = template.shape[:2]

    best_match, best_val = None, -1

    for img_path in folder.iterdir():
        if not img_path.is_file():
            continue
        if img_path.name.lower() == "thumbs.db":
            continue
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        i_h, i_w = img.shape[:2]
        if i_h < t_h or i_w < t_w:
            continue

        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > best_val:
            best_val, best_match = max_val, img_path

    return str(best_match) if best_match and best_val > 0.5 else None

# ============================================================================
#                        UNET SEGMENTATION & MEASURES
# ============================================================================

def load_unet(model_path: Path, model_class):
    """Lazy-load UNet model only once."""
    global MODEL
    if MODEL is None:
        model = model_class().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        MODEL = model
    return MODEL


def predict_mask(image_path: Path, model):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)
        # resize mask back to original image size
        pred = F.interpolate(pred, size=img.size[::-1], mode="bilinear", align_corners=False)
        pred = pred.squeeze().cpu().numpy()

    mask = (pred > THRESHOLD).astype(np.uint8) * 255
    return np.array(img), mask



def get_bbox_from_mask(mask: np.ndarray):
    """Return bounding box (x,y,w,h) of largest contour in mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(c)


def measure_chromatic_band_dimension(path_input: Path, model, input_size=(256, 256)):
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

    print("🔍 minAreaRect w,h (in px):", w, h)
    print("   orig image size:", orig_w, orig_h)


    return (max(w, h), min(w, h))

# ============================================================================
#                     DOCUMENT MEASUREMENT & PPI ESTIMATION
# ============================================================================

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


def estimate_ppi_from_chromatic_band(chromatic_band_path: Path, model) -> int | None:
    chromatic_band_path = Path(chromatic_band_path)

    try:
    # Run segmentation only on this one
        image, mask = predict_mask(chromatic_band_path, model)

        # Save mask
        mask_out_path = OUTPUT_TMP_DIR / "segmented_masks" / f"{chromatic_band_path}_mask.png"
        cv2.imwrite(str(mask_out_path), mask)

        # Bounding box
        bbox = get_bbox_from_mask(mask)
        bbox_img = image.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)
            chromatic_band_dim_px = measure_chromatic_band_dimension(chromatic_band_path, model)

            bin_img = binaryize_image(chromatic_band_path)
            bin_img_dim_px = measure_document_from_binary(bin_img)

            chromatic_band_dim_px = measure_chromatic_band_dimension(chromatic_band_path, model)
            if chromatic_band_dim_px is None:
                return None

            chromatic_band_long_side_px, chromatic_band_short_side_px = max(chromatic_band_dim_px), min(chromatic_band_dim_px)

            # Compute scale factor from pixels → mm
            scale_factor = CHROMATIC_BAND_MM / chromatic_band_long_side_px

            # Document in pixels
            bin_img = binaryize_image(chromatic_band_path)
            bin_img_dim_px = measure_document_from_binary(bin_img)
            img_long_side_px, img_short_side_px = max(bin_img_dim_px), min(bin_img_dim_px)

            # Convert document pixels → mm
            img_long_side_mm = img_long_side_px * scale_factor
            img_short_side_mm = img_short_side_px * scale_factor

            
            width_ok = min(img_long_side_mm, img_short_side_mm) <= A4_WIDTH_MM
            height_ok = max(img_long_side_mm, img_short_side_mm) <= A4_HEIGHT_MM

            print("📐 bin_img_dim_px main:", bin_img_dim_px)
            print("   ➜ img_long_side_px:", img_long_side_px)
            print("   ➜ img_short_side_px:", img_short_side_px)

            print("📏 chromatic_band_dim_px:", chromatic_band_dim_px)
            print("   ➜ chromatic_band_long_side_px:", chromatic_band_long_side_px)
            print("   ➜ chromatic_band_short_side_px:", chromatic_band_short_side_px)

            print("⚖️  scale_factor:", scale_factor)

            print("📐 Converted dimensions (mm):")
            print("   ➜ img_long_side_mm:", img_long_side_mm)
            print("   ➜ img_short_side_mm:", img_short_side_mm)

            print("✅ width_ok:", width_ok)
            print("✅ height_ok:", height_ok)

            print(f"Dimensioni stimate: ")
            print(f"Lato lungo: {img_long_side_mm:.2f}mm minore di {A4_HEIGHT_MM}mm? {'✅' if height_ok else '❌'}")
            print(f"Lato corto: {img_short_side_mm:.2f}mm minore di {A4_WIDTH_MM}mm? {'✅' if width_ok else '❌'}")
            print(f"Il file è un A4? {'✅' if (width_ok and height_ok) else '❌'}")

            ppi = 600

            if width_ok and height_ok:
                ppi = 400
                
            print(f"PPI stimati: {ppi}")
            return ppi
    except Exception as e:
        print(f"[ERROR] Error with {chromatic_band_path}: {e}")

# ============================================================================
#                 KEYPOINT MATCHING (ORB) FOR TEMPLATE ALIGNMENT
# ============================================================================

def largest_component_mask(mask):
    """Keep only the largest connected component from a mask."""
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    largest = max(cnts, key=cv2.contourArea)
    big_mask = np.zeros_like(mask)
    cv2.drawContours(big_mask, [largest], -1, 255, thickness=-1)
    return big_mask, largest


def method_keypoint_orb(img, template):    
    """
    Keypoint-based template localization using ORB + RANSAC.
    Returns dict with mask, overlay, ROI, metadata.
    """
    try:
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            return None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:3000]
        if len(matches) < 8:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inliers_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None

        h, w = template.shape[:2]
        corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1, 2)
        transformed = cv2.transform(np.array([corners]), M)[0].astype(int)

        transformed[:,0] = np.clip(transformed[:,0], 0, img.shape[1]-1)
        transformed[:,1] = np.clip(transformed[:,1], 0, img.shape[0]-1)

        mask_poly = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_poly, [transformed], 255)

        big, _ = largest_component_mask(mask_poly)
        if big is not None:
            mask_poly = big

        overlay = img.copy()
        cv2.drawContours(overlay, [transformed], -1, COL_KP, 3)

        minx, maxx = np.min(transformed[:,0]), np.max(transformed[:,0])
        miny, maxy = np.min(transformed[:,1]), np.max(transformed[:,1])
        roi = img[miny:maxy+1, minx:maxx+1].copy() if maxx>minx and maxy>miny else None

        return {"method": "keypoint", "mask": mask_poly, "overlay": overlay, "roi": roi, "meta": {"matches": len(matches)}}
    except Exception:
        return None
