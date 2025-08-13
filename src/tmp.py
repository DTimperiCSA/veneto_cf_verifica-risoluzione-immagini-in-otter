# src/ruler_detection.py
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from src.utils import *
from src.image_processing import *
from src.config import *
from src.paths import *
from src.estimate_ppi_from_ruler import *
from src.paths import INPUT_IMAGES_DIR, OUTPUT_TMP_DIR, TEMPLATE_IMG_PATH

# ----------------------- CONFIG -----------------------
MATCH_THRESHOLD = 0.6
DEBUG_HEIGHT = 3000  # height for comparison images

# Output folders
BASE_OUT = OUTPUT_TMP_DIR / "chromatic_bands"
HSV_DIR = BASE_OUT / "hsv"
TMPL_DIR = BASE_OUT / "template_matching"
TIFFEN_DIR = BASE_OUT / "tiffen_segmentation"
PRECISE_DIR = HSV_DIR / "precise_from_template"
CHROMATIC_ROI_DIR = HSV_DIR / "tmpl_roi"

COMPARISON_DIR = Path("tmp") / "confronti"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

for d in (HSV_DIR, TMPL_DIR, TIFFEN_DIR, PRECISE_DIR, CHROMATIC_ROI_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Load template
TEMPLATE_PATH = TEMPLATE_IMG_PATH
TEMPLATE = cv2.imread(str(TEMPLATE_PATH))
if TEMPLATE is None:
    raise FileNotFoundError(f"Template not found: {TEMPLATE_PATH}")
TEMPLATE_GRAY = cv2.cvtColor(TEMPLATE, cv2.COLOR_BGR2GRAY)


# ----------------------- HELPERS -----------------------
def safe_match_template(img_gray: np.ndarray, template_gray: np.ndarray, method=cv2.TM_CCOEFF_NORMED) -> Optional[np.ndarray]:
    try:
        if img_gray.shape[0] < template_gray.shape[0] or img_gray.shape[1] < template_gray.shape[1]:
            return None
        return cv2.matchTemplate(img_gray, template_gray, method)
    except cv2.error as e:
        if '(-215:Assertion failed)' in str(e):
            print("[WARN] matchTemplate: image smaller than template")
            return None
        raise


def save_debug_image(folder: Path, base_name: str, img: np.ndarray, ext: Optional[str] = None) -> Path:
    """
    Save debug image using ext if provided, otherwise default to .png.
    base_name should NOT include extension.
    Returns saved Path.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if ext is None:
        ext = ".png"
    if not ext.startswith("."):
        ext = "." + ext
    filename = f"{base_name}{ext}"
    path = folder / filename
    cv2.imwrite(str(path), img)
    return path


def minarea_rect_to_box(rect) -> np.ndarray:
    box = cv2.boxPoints(rect)
    return box

def method_precise_stats_hsv_from_template(img: np.ndarray, folder_name: str, file_name: str) -> Optional[Path]:
    """
    Template-match -> select gray pixels in ROI (exclude white/black) -> compute mean/min/max on V
    -> use those stats to segment the whole color image in HSV (semantic segmentation),
    -> pick best elongated component, draw magenta box, save ROI + debug masks (preserve ext).
    """
    # try to localize via template
    loc = template_localize(img)
    if loc is None or loc["max_val"] < MATCH_THRESHOLD:
        print(f"[PREC_STATS] template fail or low confidence ({loc['max_val'] if loc else 'N/A'}) for {file_name}")
        return None

    top_left = loc["top_left"]
    w_temp, h_temp = loc["w"], loc["h"]

    # crop ROI
    roi = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp].copy()
    if roi.size == 0:
        print(f"[PREC_STATS] empty ROI for {file_name}")
        return None

    # convert ROI to HSV and grayscale-like brightness (V)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_v = roi_hsv[:, :, 2]
    roi_s = roi_hsv[:, :, 1]

    # mask gray-ish pixels: low saturation, exclude near-black and near-white
    SAT_THR = 60                # pixels with S <= SAT_THR considered not strongly colored
    V_MIN_EXCLUDE = 10         # exclude near-black
    V_MAX_EXCLUDE = 245        # exclude near-white

    gray_mask = (roi_s <= SAT_THR) & (roi_v >= V_MIN_EXCLUDE) & (roi_v <= V_MAX_EXCLUDE)
    gray_pixels = roi_v[gray_mask]

    if gray_pixels.size == 0:
        print(f"[PREC_STATS] No gray pixels found in ROI for {file_name}")
        return None

    # compute the three statistics requested
    mean_v = float(np.mean(gray_pixels))
    min_v = float(np.min(gray_pixels))
    max_v = float(np.max(gray_pixels))
    print(f"[PREC_STATS] {file_name} stats -> mean_v={mean_v:.1f}, min_v={min_v:.1f}, max_v={max_v:.1f}")

    # Build HSV thresholds for full-image segmentation using those stats
    # Use some margins (tunable). We keep low saturation to remain gray-like.
    MARGIN_LOW = 20
    MARGIN_HIGH = 20

    lower_v = int(max(0, min_v - MARGIN_LOW))
    upper_v = int(min(255, max_v + MARGIN_HIGH))
    lower_hsv = np.array([0, 0, lower_v], dtype=np.uint8)
    upper_hsv = np.array([180, SAT_THR, upper_v], dtype=np.uint8)

    # apply to full color image (semantic segmentation in color space)
    full_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(full_hsv, lower_hsv, upper_hsv)

    # postprocess mask: close gaps, remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel, iterations=1)

    # save ROI and mask debug (preserve extension)
    roi_dir = PRECISE_DIR / "stats_roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_base = f"roi_precstats_{folder_name}_{Path(file_name).stem}"
    cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi)

    debug_dir = PRECISE_DIR / "stats_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    # save mask using same extension as original if possible, otherwise .png
    save_debug_image(debug_dir, f"mask_precstats_{folder_name}_{Path(file_name).stem}", mask_hsv, ext=Path(file_name).suffix)

    # find contours on mask_hsv to pick best elongated component
    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:  # small noise
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if min(w,h) <= 0:
            continue
        aspect_ratio = max(w,h) / min(w,h)
        # require elongated shape (tunable)
        if aspect_ratio < 1.8:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    # if none meet elongated criteria, relax the ratio and pick largest component
    if best_rect is None and contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) >= 300:
            best_rect = cv2.minAreaRect(c)

    if best_rect is None:
        print(f"[PREC_STATS] No suitable component found for {file_name}")
        # still save overlay for inspection
        overlay = cv2.addWeighted(img, 0.8, cv2.cvtColor(mask_hsv, cv2.COLOR_GRAY2BGR), 0.2, 0)
        out_f = PRECISE_DIR / f"precstats_overlay_{folder_name}_{file_name}"
        cv2.imwrite(str(out_f), overlay)
        return out_f

    # map selected rect to box and draw (magenta)
    box = cv2.boxPoints(best_rect).astype(int)
    out_img = img.copy()
    cv2.drawContours(out_img, [box], -1, (255, 0, 255), 2)  # magenta

    out_path = PRECISE_DIR / f"precise_stats_hsv_{folder_name}_{file_name}"
    cv2.imwrite(str(out_path), out_img)
    print(f"[PREC_STATS] saved {out_path}")

    return out_path

# ----------------------- METHODS -----------------------
def template_localize(img: np.ndarray) -> Optional[Dict[str, Any]]:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = safe_match_template(img_gray, TEMPLATE_GRAY)
    if res is None:
        return None
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    h, w = TEMPLATE_GRAY.shape
    return {"max_val": float(max_val), "top_left": (int(max_loc[0]), int(max_loc[1])), "w": w, "h": h}


def method_template_only(img: np.ndarray, folder_name: str, file_name: str) -> Optional[Path]:
    loc = template_localize(img)
    if loc is None or loc["max_val"] < MATCH_THRESHOLD:
        return None
    top_left = loc["top_left"]
    w, h = loc["w"], loc["h"]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    out = img.copy()
    cv2.rectangle(out, top_left, bottom_right, (0, 255, 0), 2)  # green
    save_path = TMPL_DIR / f"chromatic_band_{folder_name}_{file_name}"
    cv2.imwrite(str(save_path), out)

    # Save ROI (preserve original extension)
    roi = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w]
    roi_dir = TMPL_DIR / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_base = f"roi_chromatic_{folder_name}_{Path(file_name).stem}"
    cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi)
    return save_path


def method_chromatic_band_hsv(img: np.ndarray, folder_name: str, file_name: str) -> Optional[Path]:
    loc = template_localize(img)
    if loc is None or loc["max_val"] < MATCH_THRESHOLD:
        return None
    top_left = loc["top_left"]
    w, h = loc["w"], loc["h"]
    roi = img[top_left[1]:top_left[1]+h, top_left[0]:top_left[0]+w].copy()
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 20, 30]), np.array([180, 80, 160]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        roi_dir = CHROMATIC_ROI_DIR / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        roi_base = f"roi_chromatic_{folder_name}_{Path(file_name).stem}"
        cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi)
        # save mask debug with original ext (use .png if original not suitable)
        debug_dir = CHROMATIC_ROI_DIR / "debug"
        save_debug_image(debug_dir, f"mask_chromatic_{folder_name}_{Path(file_name).stem}", mask, ext=Path(file_name).suffix)
        return None

    c = max(contours, key=cv2.contourArea)
    x, y, wc, hc = cv2.boundingRect(c)
    box_pts = np.array([[x, y], [x + wc, y], [x + wc, y + hc], [x, y + hc]], dtype=np.float32)
    box_pts[:, 0] += top_left[0]; box_pts[:, 1] += top_left[1]

    out = img.copy()
    cv2.drawContours(out, [box_pts.astype(int)], -1, (0, 165, 255), 2)  # orange
    save_path = HSV_DIR / "tmpl_roi" / f"hsv_tmpl_roi_{folder_name}_{file_name}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)

    roi_dir = HSV_DIR / "tmpl_roi" / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_base = f"roi_chromatic_{folder_name}_{Path(file_name).stem}"
    cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi)

    mask_dir = HSV_DIR / "tmpl_roi" / "debug"
    mask_dir.mkdir(parents=True, exist_ok=True)
    save_debug_image(mask_dir, f"mask_chromatic_{folder_name}_{Path(file_name).stem}", mask, ext=Path(file_name).suffix)
    return save_path


def method_tiffen_segmentation(img: np.ndarray, folder_name: str, file_name: str) -> Optional[Path]:
    loc = template_localize(img)
    if loc is None or loc["max_val"] < MATCH_THRESHOLD:
        return None
    top_left = loc["top_left"]
    w_temp, h_temp = loc["w"], loc["h"]
    roi_color = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp].copy()
    if roi_color.size == 0:
        return None

    roi_hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    sat = roi_hsv[:, :, 1]
    mask_valid = (sat < 60) & (roi_gray > 10) & (roi_gray < 245)
    pixels = roi_gray[mask_valid]
    if pixels.size == 0:
        return None
    med = np.median(pixels)

    low = int(max(0, med - 30))
    high = int(min(255, med + 30))
    roi_mask = cv2.inRange(roi_gray, low, high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if min(w, h) == 0:
            continue
        ar = max(w, h) / (min(w, h) + 1e-6)
        if ar < 2.0:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    # Save ROI and mask for debugging (preserve ext)
    roi_dir = TIFFEN_DIR / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_base = f"roi_tiffen_{folder_name}_{Path(file_name).stem}"
    cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi_color)

    debug_dir = TIFFEN_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_debug_image(debug_dir, f"roi_mask_{folder_name}_{Path(file_name).stem}", roi_mask, ext=Path(file_name).suffix)

    if best_rect is None:
        overlay = cv2.addWeighted(img, 0.8, cv2.cvtColor(cv2.resize(roi_mask, (roi_mask.shape[1], roi_mask.shape[0])), cv2.COLOR_GRAY2BGR), 0.2, 0)
        out_path = TIFFEN_DIR / f"tiffen_segmentation_{folder_name}_{file_name}"
        cv2.imwrite(str(out_path), overlay)
        return out_path

    box = cv2.boxPoints(best_rect).astype(int)
    box[:, 0] += top_left[0]; box[:, 1] += top_left[1]
    out = img.copy()
    cv2.drawContours(out, [box], -1, (255, 0, 0), 2)  # blue
    out_path = TIFFEN_DIR / f"tiffen_segmentation_{folder_name}_{file_name}"
    cv2.imwrite(str(out_path), out)
    return out_path


def method_precise_hsv_from_template(img: np.ndarray, folder_name: str, file_name: str) -> Optional[Path]:
    loc = template_localize(img)
    if loc is None or loc["max_val"] < MATCH_THRESHOLD:
        return None
    top_left = loc["top_left"]
    w_temp, h_temp = loc["w"], loc["h"]
    roi = img[top_left[1]:top_left[1]+h_temp, top_left[0]:top_left[0]+w_temp].copy()
    if roi.size == 0:
        return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = roi_hsv[:, :, 1]
    mask_valid = (sat < 60) & (roi_gray > 10) & (roi_gray < 245)
    pixels = roi_gray[mask_valid]
    if pixels.size == 0:
        print(f"[INFO] No valid pixels for median on {file_name}")
        return None
    med = np.median(pixels)

    lower_v = int(max(0, med - 40))
    upper_v = int(min(255, med + 40))
    lower_hsv = np.array([0, 0, lower_v], dtype=np.uint8)
    upper_hsv = np.array([180, 60, upper_v], dtype=np.uint8)
    mask_hsv = cv2.inRange(roi_hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save ROI+mask debug (preserve ext)
    roi_dir = PRECISE_DIR / "roi"
    roi_dir.mkdir(parents=True, exist_ok=True)
    roi_base = f"roi_precise_{folder_name}_{Path(file_name).stem}"
    cv2.imwrite(str(roi_dir / f"{roi_base}{Path(file_name).suffix}"), roi)

    debug_dir = PRECISE_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_debug_image(debug_dir, f"mask_precise_{folder_name}_{Path(file_name).stem}", mask_hsv, ext=Path(file_name).suffix)

    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 250:
            continue
        rect = cv2.minAreaRect(c)
        (_, _), (w, h), _ = rect
        if min(w, h) == 0:
            continue
        ar = max(w, h) / min(w, h)
        if ar < 1.5:
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    if best_rect is None:
        return None

    box = cv2.boxPoints(best_rect).astype(int)
    box[:, 0] += top_left[0]; box[:, 1] += top_left[1]
    out = img.copy()
    cv2.drawContours(out, [box], -1, (255, 0, 255), 2)  # magenta
    out_path = PRECISE_DIR / f"precise_hsv_tmpl_{folder_name}_{file_name}"
    cv2.imwrite(str(out_path), out)
    return out_path


# ----------------------- COMPARISON IMAGE -----------------------
def create_comparison_image(folder_name: str, file_name: str) -> Optional[Path]:
    expected_methods = [
        ("hsv_tmpl_roi", HSV_DIR / "tmpl_roi"),
        ("chromatic_band", TMPL_DIR),
        ("tiffen_segmentation", TIFFEN_DIR),
        ("precise_hsv_tmpl", PRECISE_DIR)
    ]

    imgs = []
    labels = []
    orig_ext = Path(file_name).suffix or ".png"
    stem = Path(file_name).stem

    for method_label, dir_path in expected_methods:
        fn = f"{method_label}_{folder_name}_{file_name}"
        p = dir_path / fn
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                imgs.append(im); labels.append(method_label)
                continue
        # try with extension suffix fallback (in case saved without ext)
        p2 = dir_path / f"{method_label}_{folder_name}_{stem}{orig_ext}"
        if p2.exists():
            im = cv2.imread(str(p2))
            if im is not None:
                imgs.append(im); labels.append(method_label)

    if not imgs:
        return None

    target_h = DEBUG_HEIGHT
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(im, (new_w, target_h)))

    total_w = sum(im.shape[1] for im in resized)
    canvas_h = target_h + 30
    canvas = np.ones((canvas_h, total_w, 3), dtype=np.uint8) * 255

    x = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for im, lab in zip(resized, labels):
        text_size = cv2.getTextSize(lab, font, 0.7, 2)[0]
        text_x = x + (im.shape[1] - text_size[0]) // 2
        cv2.putText(canvas, lab, (text_x, 20), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        canvas[30:30 + im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1]

    out_path = COMPARISON_DIR / f"confronto_{folder_name}_{Path(file_name).stem}{orig_ext}"
    cv2.imwrite(str(out_path), canvas)
    return out_path


# ----------------------- ORCHESTRATION -----------------------
def process_image(img_path: Path, folder_name: str):
    file_name = img_path.name
    img = safe_imread(img_path)
    if img is None:
        print(f"[ERR] cannot read {img_path}")
        return

    print(f"[INFO] Processing {folder_name}/{file_name}")

    results: Dict[str, Optional[Path]] = {}

    try:
        p = method_chromatic_band_hsv(img, folder_name, file_name)
        results['hsv_tmpl_roi'] = p
        print(f"[HSV_TMPL_ROI] {p}")
    except Exception as e:
        print(f"[ERR] hsv tmpl roi failed: {e}")

    try:
        p = method_template_only(img, folder_name, file_name)
        results['chromatic_band'] = p
        print(f"[TMPL] {p}")
    except Exception as e:
        print(f"[ERR] template-only failed: {e}")

    try:
        p = method_tiffen_segmentation(img, folder_name, file_name)
        results['tiffen_segmentation'] = p
        print(f"[TIFFEN_SEG] {p}")
    except Exception as e:
        print(f"[ERR] tiffen_seg failed: {e}")

    try:
        p = method_precise_hsv_from_template(img, folder_name, file_name)
        results['precise_hsv_tmpl'] = p
        print(f"[PRECISE_HSV_TMPL] {p}")
    except Exception as e:
        print(f"[ERR] precise hsv failed: {e}")

    try:
        p = method_precise_stats_hsv_from_template(img, folder_name, file_name)
        results['precise_stats_hsv'] = p
        print(f"[PRECISE_STATS_HSV] {p}")
    except Exception as e:
        print(f"[ERR] precise stats hsv failed: {e}")


    try:
        comp = create_comparison_image(folder_name, file_name)
        if comp:
            print(f"[CONFRONTO] {comp}")
    except Exception as e:
        print(f"[ERR] create comparison failed: {e}")

    loc = template_localize(img)
    if loc:
        print(f"[MATCH] template_confidence={loc['max_val']:.3f}, top_left={loc['top_left']}")
    else:
        print("[MATCH] template not usable")

    return results


def process_all_folders(root: Path):
    folders_processed = 0

    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue

        image_files = [f for f in folder.iterdir()
                       if f.is_file() and f.name.lower() != "thumbs.db" and is_valid_image_file(f)[0]]

        if not image_files:
            continue

        image_files = sorted(image_files)
        last_image = image_files[-1]

        print(f"\n🚀 Folder: {folder.name} images: {len(image_files)} -> processing: {last_image.name}")

        try:
            process_image(last_image, folder.name)
        except Exception as e:
            print(f"[ERR] processing {last_image.name}: {e}")

        folders_processed += 1

    print(f"\n✅ Done. Processed folders: {folders_processed}")


if __name__ == "__main__":
    process_all_folders(INPUT_IMAGES_DIR)
