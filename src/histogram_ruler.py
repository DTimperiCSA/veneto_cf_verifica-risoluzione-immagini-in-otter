# src/ruler_detection_hist_methods.py
"""
ruler_detection_hist_methods.py

Due metodi basati su istogramma grigi:
 - method_global_histogram_gray: stima grigio sull'intera immagine (escludendo pixel saturi)
   e segmenta l'immagine a colori con le soglie trovate.
 - method_roi_template_histogram_gray: trova il template (single-scale), calcola istogramma
   dei grigi sulla ROI e applica le soglie all'intera immagine.

Per ogni immagine (solo l'ultima in ogni cartella) salva:
 - overlay image con bounding box (estensione originale)
 - mask (estensione originale)
 - roi crop (se presente)
 - histogram plot (.png)
 - metadata (.txt)
Cartelle di output: OUTPUT_TMP_DIR/chromatic_bands/hist_methods/{method}/...
"""
from pathlib import Path
import cv2
import numpy as np
import math
import json
import sys

# plotting
import matplotlib.pyplot as plt

from src.paths import INPUT_IMAGES_DIR, OUTPUT_TMP_DIR, TEMPLATE_IMG_PATH

# CONFIG
MATCH_THRESHOLD = 0.60
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

OUT_BASE = Path(OUTPUT_TMP_DIR) / "chromatic_bands" / "hist_methods"
GLOBAL_DIR = OUT_BASE / "global_hist"
ROI_DIR = OUT_BASE / "roi_hist"

HIST_DIRNAME = "histograms"
MASK_DIRNAME = "masks"
ROI_SUBDIR = "roi"
OVERLAY_DIRNAME = "overlay"
META_DIRNAME = "meta"

# morphological params
KERNEL = np.ones((5,5), np.uint8)

# helper utils
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_with_ext(path: Path, img: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

def gaussian_smooth(hist, sigma=3):
    radius = int(3*sigma)
    x = np.arange(-radius, radius+1)
    kernel = np.exp(-0.5*(x/sigma)**2)
    kernel = kernel / kernel.sum()
    return np.convolve(hist, kernel, mode='same')

def find_peak_interval(hist, bins_edges, frac=0.25):
    """
    hist: smoothed histogram counts
    bins_edges: edges from np.histogram
    returns (lower_bin_value, upper_bin_value, peak_bin_value)
    """
    if hist.sum() == 0:
        return None
    peak_idx = int(np.argmax(hist))
    peak_val = (bins_edges[peak_idx] + bins_edges[peak_idx+1]) / 2.0
    thresh = hist.max() * frac
    # expand left
    left = peak_idx
    while left-1 >= 0 and hist[left-1] >= thresh:
        left -= 1
    right = peak_idx
    while right+1 < hist.size and hist[right+1] >= thresh:
        right += 1
    low = int(bins_edges[left])
    high = int(bins_edges[right+1]) if (right+1) < bins_edges.size else int(bins_edges[-1])
    return low, high, peak_val

def largest_rect_from_mask(mask, min_area=500):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    best = None
    best_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None, None
    rect = cv2.minAreaRect(best)
    box = cv2.boxPoints(rect).astype(int)
    return rect, box

def overlay_box(img, box, color=(255,0,255), thickness=3):
    out = img.copy()
    if box is None:
        return out
    cv2.drawContours(out, [np.array(box)], -1, color, thickness)
    return out

def compute_and_save_histogram(vals, outpath_png):
    """vals is 1D array of values (0..255). Save histogram png."""
    ensure_dir(outpath_png.parent)
    if vals.size == 0:
        # save empty placeholder
        fig = plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, "no data", ha='center', va='center')
        plt.axis('off')
        fig.savefig(str(outpath_png), bbox_inches='tight')
        plt.close(fig)
        return
    hist, edges = np.histogram(vals, bins=256, range=(0,255))
    hist_s = gaussian_smooth(hist.astype(float), sigma=3)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = plt.figure(figsize=(8,3))
    plt.plot(centers, hist, alpha=0.3, label='raw')
    plt.plot(centers, hist_s, label='smoothed')
    plt.xlabel("gray value")
    plt.ylabel("count")
    plt.legend()
    fig.savefig(str(outpath_png), bbox_inches='tight')
    plt.close(fig)

def safe_match_template(img_gray, templ_gray):
    """Return (max_val, top_left) or None if can't apply."""
    if img_gray is None or templ_gray is None:
        return None
    if img_gray.shape[0] < templ_gray.shape[0] or img_gray.shape[1] < templ_gray.shape[1]:
        return None
    try:
        res = cv2.matchTemplate(img_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
    return float(maxv), (int(maxloc[0]), int(maxloc[1]))

# ---------------- Method 1: global histogram ----------------
def method_global_histogram_gray(img, folder_name, file_name, ext):
    """
    Estimate gray from whole-image histogram (considering low-saturation pixels),
    then segment the color image using thresholds found.
    """
    out_folder = GLOBAL_DIR
    ensure_dir(out_folder / HIST_DIRNAME)
    ensure_dir(out_folder / MASK_DIRNAME)
    ensure_dir(out_folder / ROI_SUBDIR)
    ensure_dir(out_folder / OVERLAY_DIRNAME)
    ensure_dir(out_folder / META_DIRNAME)

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # mask low saturation AND exclude near-black and near-white in V
    sat_mask = (Sc <= 70)
    val_mask = (Vc > 10) & (Vc < 245)
    valid_mask = sat_mask & val_mask

    vals_gray = gray[valid_mask]
    vals_v = Vc[valid_mask]

    if vals_gray.size == 0:
        # fallback: use global gray excluding extremes
        vals_gray = gray[(gray > 5) & (gray < 250)]
        if vals_gray.size == 0:
            print("[GLOBAL HIST] No valid pixels for estimation")
            vals_gray = np.array([], dtype=np.uint8)

    # compute hist and save
    hist_png = GLOBAL_DIR / HIST_DIRNAME / f"global_hist_{folder_name}_{file_name}.png"
    compute_and_save_histogram(vals_gray, hist_png)

    # analyze histogram to get interval
    if vals_gray.size > 0:
        hist, edges = np.histogram(vals_gray, bins=256, range=(0,255))
        hist_s = gaussian_smooth(hist.astype(float), sigma=3)
        found = find_peak_interval(hist_s, edges, frac=0.25)
        if found is None:
            lower, upper = max(0, int(np.median(vals_gray)-30)), min(255, int(np.median(vals_gray)+30))
            peak = float(np.median(vals_gray))
        else:
            lower, upper, peak = found
    else:
        lower, upper, peak = 50, 180, 120  # generic fallback

    # apply thresholds on V (preferred) + low saturation to segment on whole colored image
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hf, Sf, Vf = cv2.split(hsv_full)
    mask_v = cv2.inRange(Vf, int(lower), int(upper))
    mask_s = cv2.inRange(Sf, 0, 80)
    mask = cv2.bitwise_and(mask_v, mask_s)

    # morphology
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=1)

    rect, box = largest_rect_from_mask(mask, min_area=300)
    overlay = img.copy()
    roi_crop = None
    bbox_coords = None
    if box is not None:
        overlay = overlay_box(overlay, box, color=(128,0,128), thickness=3)
        # compute bounding rect coords (minx,miny,maxx,maxy)
        xs = box[:,0]; ys = box[:,1]
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox_coords = (x1, y1, x2, y2)
        roi_crop = img[y1:y2, x1:x2].copy()

    # save outputs preserving extension
    base_name = f"globalhist_{folder_name}_{file_name}"
    save_with_ext = lambda p, i: (ensure_dir(p.parent) or cv2.imwrite(str(p), i))
    save_with_ext(GLOBAL_DIR / OVERLAY_DIRNAME / f"{base_name}{ext}", overlay)
    save_with_ext(GLOBAL_DIR / MASK_DIRNAME / f"{base_name}{ext}", mask)
    if roi_crop is not None:
        save_with_ext(GLOBAL_DIR / ROI_SUBDIR / f"{base_name}{ext}", roi_crop)

    # meta save
    meta = {"method":"global_histogram_gray", "folder":str(folder_name), "file":str(file_name),
            "lower":int(lower), "upper":int(upper), "peak":float(peak),
            "bbox":bbox_coords}
    meta_path = GLOBAL_DIR / META_DIRNAME / f"{base_name}.json"
    ensure_dir(meta_path.parent)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    # print found dims
    if bbox_coords:
        w_px = bbox_coords[2]-bbox_coords[0]; h_px = bbox_coords[3]-bbox_coords[1]
        print(f"[GLOBAL HIST] {file_name}: bbox {w_px}px x {h_px}px (lower={lower}, upper={upper}, peak={peak:.1f})")
    else:
        print(f"[GLOBAL HIST] {file_name}: no bbox found (lower={lower}, upper={upper}, peak={peak:.1f})")

    return meta

# ---------------- Method 2: ROI-template histogram ----------------
def method_roi_template_histogram_gray(img, template_path, folder_name, file_name, ext):
    """
    Do single-scale template matching. If match found (>=MATCH_THRESHOLD) compute histogram of grayscale
    on the ROI (exclude saturated pixels / whites / blacks) and then apply thresholds derived from ROI histogram
    to segment the whole image (color). Save ROI histogram, mask, overlay, roi, meta.
    """
    out_folder = ROI_DIR
    ensure_dir(out_folder / HIST_DIRNAME)
    ensure_dir(out_folder / MASK_DIRNAME)
    ensure_dir(out_folder / ROI_SUBDIR)
    ensure_dir(out_folder / OVERLAY_DIRNAME)
    ensure_dir(out_folder / META_DIRNAME)

    template = cv2.imread(str(template_path))
    if template is None:
        print("[ROI HIST] template not found:", template_path)
        return None
    # use grayscale template
    templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    match = safe_match_template(img_gray, templ_gray)
    if match is None:
        print(f"[ROI HIST] template match failed (image too small or error) for {file_name}")
        return None
    maxv, top_left = match
    if maxv < MATCH_THRESHOLD:
        print(f"[ROI HIST] template match too weak ({maxv:.2f}) for {file_name}")
        return None

    x1, y1 = top_left
    h_t, w_t = templ_gray.shape[:2]
    x2, y2 = x1 + w_t, y1 + h_t
    # clip
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
    roi = img[y1:y2, x1:x2].copy()
    if roi.size == 0:
        print("[ROI HIST] empty roi")
        return None

    # compute grey histogram on ROI excluding saturated/colorful pixels
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv_roi)
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    sat_mask = (Sc <= 70)
    val_mask = (Vc > 10) & (Vc < 245)
    valid_mask = sat_mask & val_mask
    vals_gray = gray_roi[valid_mask]

    if vals_gray.size == 0:
        # fallback: use gray roi excluding extremes
        vals_gray = gray_roi[(gray_roi > 5) & (gray_roi < 250)]
        if vals_gray.size == 0:
            print("[ROI HIST] no valid gray pixels in roi")
            vals_gray = np.array([], dtype=np.uint8)

    # save histogram of ROI
    hist_png = ROI_DIR / HIST_DIRNAME / f"roi_hist_{folder_name}_{file_name}.png"
    compute_and_save_histogram(vals_gray, hist_png)

    # analyze histogram
    if vals_gray.size > 0:
        hist, edges = np.histogram(vals_gray, bins=256, range=(0,255))
        hist_s = gaussian_smooth(hist.astype(float), sigma=3)
        found = find_peak_interval(hist_s, edges, frac=0.25)
        if found is None:
            lower, upper = max(0, int(np.median(vals_gray)-20)), min(255, int(np.median(vals_gray)+20))
            peak = float(np.median(vals_gray))
        else:
            lower, upper, peak = found
    else:
        lower, upper, peak = 50, 180, 120

    # apply thresholds to whole color image (use V channel + low S)
    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    Hf, Sf, Vf = cv2.split(hsv_full)
    mask_v = cv2.inRange(Vf, int(lower), int(upper))
    mask_s = cv2.inRange(Sf, 0, 80)
    mask = cv2.bitwise_and(mask_v, mask_s)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=1)

    # largest rect
    rect, box = largest_rect_from_mask(mask, min_area=300)
    overlay = img.copy()
    roi_saved = None
    bbox_coords = None
    if box is not None:
        overlay = overlay_box(overlay, box, color=(255,0,255), thickness=3)
        xs = box[:,0]; ys = box[:,1]
        x1b, y1b, x2b, y2b = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox_coords = (x1b, y1b, x2b, y2b)
        roi_saved = img[y1b:y2b, x1b:x2b].copy()

    # save outputs preserving extension
    base_name = f"roi_hist_{folder_name}_{file_name}"
    save_with_ext = lambda p, i: (ensure_dir(p.parent) or cv2.imwrite(str(p), i))
    save_with_ext(ROI_DIR / OVERLAY_DIRNAME / f"{base_name}{ext}", overlay)
    save_with_ext(ROI_DIR / MASK_DIRNAME / f"{base_name}{ext}", mask)
    save_with_ext(ROI_DIR / ROI_SUBDIR / f"{base_name}{ext}", roi)  # original ROI from template
    if roi_saved is not None:
        save_with_ext(ROI_DIR / ROI_SUBDIR / f"{base_name}_segroi{ext}", roi_saved)

    # meta
    meta = {"method":"roi_template_histogram_gray", "folder":str(folder_name), "file":str(file_name),
            "lower":int(lower), "upper":int(upper), "peak":float(peak),
            "template_conf": float(maxv), "template_top_left": (int(x1), int(y1)),
            "bbox": bbox_coords}
    meta_path = ROI_DIR / META_DIRNAME / f"{base_name}.json"
    ensure_dir(meta_path.parent)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    # print dims
    if bbox_coords:
        w_px = bbox_coords[2]-bbox_coords[0]; h_px = bbox_coords[3]-bbox_coords[1]
        print(f"[ROI HIST] {file_name}: bbox {w_px}px x {h_px}px (lower={lower}, upper={upper}, peak={peak:.1f}) conf={maxv:.2f}")
    else:
        print(f"[ROI HIST] {file_name}: no bbox found (lower={lower}, upper={upper}, peak={peak:.1f}) conf={maxv:.2f}")

    return meta

# ---------------- Walk folders & run only last image ----------------
def get_folders_with_last_image(root: Path):
    folders = []
    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        image_files = [
            f for f in folder.iterdir()
            if f.is_file() and f.suffix.lower() in EXTS and f.name.lower() != "thumbs.db"
        ]
        if not image_files:
            continue
        image_files = sorted(image_files)
        last_image = image_files[-1]
        folders.append((folder, last_image))
    return folders

def process_all(input_root: Path, template_path: Path):
    ensure_dir(GLOBAL_DIR); ensure_dir(ROI_DIR)
    folders = get_folders_with_last_image(input_root)
    if not folders:
        print("[INFO] No folders with images found under", input_root)
        return
    template_exists = Path(template_path).exists()
    if not template_exists:
        print("[WARN] Template file not found:", template_path, "- ROI-template method will skip.")
    for folder, last_image in folders:
        try:
            print(f"\n➡ Folder: {folder} -> last image: {last_image.name}")
            img = cv2.imread(str(last_image))
            if img is None:
                print("[ERR] cannot read", last_image)
                continue
            ext = last_image.suffix
            # Method A: global histogram
            try:
                method_global_histogram_gray(img, folder.name, last_image.name, ext)
            except Exception as e:
                print("[ERR] method_global_histogram_gray failed:", e)

            # Method B: roi template histogram (only if template exists)
            if template_exists:
                try:
                    method_roi_template_histogram_gray(img, TEMPLATE_IMG_PATH, folder.name, last_image.name, ext)
                except Exception as e:
                    print("[ERR] method_roi_template_histogram_gray failed:", e)
        except Exception as e:
            print("[ERR] processing", folder, e)

    print("\nDone. Outputs saved under:", OUT_BASE)

if __name__ == "__main__":
    print("INPUT_IMAGES_DIR:", INPUT_IMAGES_DIR)
    print("TEMPLATE_IMG_PATH:", TEMPLATE_IMG_PATH)
    process_all(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH)
