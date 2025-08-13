# src/ruler_detection_all_methods.py
"""
Pipeline multi-metodo per rilevare la banda Tiffen (righello + patch) nelle immagini.
Incluso: metodo basato su istogramma per stimare il "grigio" più rappresentativo.
- processa SOLO l'ultima immagine di ogni cartella (ricorsivo sotto INPUT_IMAGES_DIR)
- salva overlay, mask, roi, debug images e un'immagine comparativa con tutti i metodi.
- preserva estensione originale per immagini salvate.
"""
from pathlib import Path
import cv2
import numpy as np
import itertools
import math
import traceback

# optional: scikit-image (SLIC)
try:
    from skimage.segmentation import slic
    from skimage.color import rgb2lab
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

from src.paths import INPUT_IMAGES_DIR, OUTPUT_TMP_DIR, TEMPLATE_IMG_PATH

# ---------------- CONFIG ----------------
MATCH_THRESHOLD = 0.60
LAB_TOLS = [5, 10, 15, 20]
HSV_S_TOLS = [30, 40, 50, 60]
SLIC_N_SEGMENTS = 400
K_GRAY = 200
MAX_EXPAND_STEPS = 5
SCALE_FACTOR = 1.5

OUTPUT_BASE = Path(OUTPUT_TMP_DIR) / "all_methods_results"
ROI_DIRNAME = OUTPUT_BASE / "roi"
MASK_DIRNAME = OUTPUT_BASE / "masks"
DEBUG_DIRNAME = OUTPUT_BASE / "debug"
COMPARE_DIR = OUTPUT_BASE / "confronti"

# extension accepted
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# colors for overlays (BGR)
COLORS = {
    "template": (0, 255, 0),       # green
    "keypoint_orb": (0, 165, 255), # orange
    "hough": (0, 140, 255),        # dark orange
    "slic": (255, 165, 0),         # light color
    "mser": (0, 255, 255),         # yellow
    "watershed": (255, 0, 0),      # blue
    "grabcut": (0, 0, 255),        # red
    "labhsv_bruteforce": (0, 165, 255),
    "precise_gray_expand": (255, 0, 255), # magenta
    "histogram_gray": (128, 0, 128),      # purple
}

from collections import namedtuple
Result = namedtuple("Result", ["method", "mask", "overlay", "roi", "score", "meta"])

# ---------------- Helpers ----------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_with_ext(path: Path, img: np.ndarray):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img)

def get_last_image_in_folder(folder: Path):
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS and p.name.lower() != "thumbs.db"]
    if not images:
        return None
    images = sorted(images)
    return images[-1]

def largest_component_mask(mask):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    largest = max(cnts, key=cv2.contourArea)
    big_mask = np.zeros_like(mask)
    cv2.drawContours(big_mask, [largest], -1, 255, thickness=-1)
    return big_mask, largest

def min_area_rect_from_mask(mask):
    big_mask, largest = largest_component_mask(mask)
    if largest is None:
        return None, None
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(int)
    return rect, box

def overlay_box(img, box_pts, color=(255,0,255), thickness=3):
    out = img.copy()
    pts = np.array(box_pts)
    if pts.shape[0] >= 4:
        cv2.drawContours(out, [pts], -1, color, thickness)
    elif pts.shape[0] == 2:
        cv2.rectangle(out, tuple(pts[0]), tuple(pts[1]), color, thickness)
    return out

def compute_basic_metrics(mask, contour=None):
    h, w = mask.shape[:2]
    A = int(np.count_nonzero(mask))
    if A == 0:
        return {"area": 0, "aspect_ratio": 0, "extent": 0, "solidity": 0, "bbox_area": 0}
    if contour is None:
        _, contour = largest_component_mask(mask)
    if contour is None:
        return {"area": A, "aspect_ratio": 0, "extent": 0, "solidity": 0, "bbox_area": 0}
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = bw * bh
    aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-8)
    extent = A / (bbox_area + 1e-8)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) if hull is not None else bbox_area
    solidity = A / (hull_area + 1e-8)
    return {"area": A, "aspect_ratio": aspect_ratio, "extent": extent, "solidity": solidity, "bbox_area": bbox_area}

def line_alignment_score(img_gray, mask):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0.0
    mask_edges = cv2.Canny(mask, 50, 150)
    count = 0
    for l in lines:
        x1, y1, x2, y2 = l[0]
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= my < mask_edges.shape[0] and 0 <= mx < mask_edges.shape[1] and mask_edges[my, mx] > 0:
            count += 1
    return min(1.0, count / 5.0)

def score_mask(mask, img):
    big_mask, contour = largest_component_mask(mask)
    metrics = compute_basic_metrics(mask, contour)
    if metrics["area"] == 0:
        return 0.0, metrics
    area_score = min(1.0, metrics["area"] / (img.shape[0] * img.shape[1] * 0.5))
    AR = metrics["aspect_ratio"]
    ar_score = 0.0
    if AR > 1:
        ar_score = min(1.0, math.log(AR + 1) / math.log(50))
    extent_score = min(1.0, metrics["extent"] / 0.9)
    solidity_score = min(1.0, metrics["solidity"] / 0.9)
    line_score = line_alignment_score(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
    score = 0.18 * area_score + 0.22 * ar_score + 0.2 * extent_score + 0.15 * solidity_score + 0.25 * line_score
    return float(score), metrics

def build_comparison_image(results, img_shape, label_size=28, per_row=3):
    if not results:
        return None
    imgs = []
    for r in results:
        if r.overlay is None:
            continue
        overlay = r.overlay.copy()
        txt = r.method
        cv2.putText(overlay, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        imgs.append(overlay)
    if not imgs:
        return None
    target_h = 400
    resized = []
    for im in imgs:
        h, w = im.shape[:2]
        scale = target_h / float(h)
        new_w = int(w * scale)
        imr = cv2.resize(im, (new_w, target_h))
        resized.append(imr)
    rows = []
    for i in range(0, len(resized), per_row):
        row_imgs = resized[i:i+per_row]
        max_w = max(im.shape[1] for im in row_imgs)
        normed = []
        for im in row_imgs:
            if im.shape[1] < max_w:
                pad = np.zeros((target_h, max_w - im.shape[1], 3), dtype=np.uint8)
                im = np.concatenate([im, pad], axis=1)
            normed.append(im)
        row = np.concatenate(normed, axis=1)
        rows.append(row)
    montage = np.concatenate(rows, axis=0)
    return montage

# ---------------- Histogram helper for gray estimation ----------------
def gaussian_kernel1d(sigma=3, radius=8):
    x = np.arange(-radius, radius+1)
    k = np.exp(-(x**2)/(2*sigma*sigma))
    k = k / k.sum()
    return k

def estimate_gray_from_histogram(img_bgr, roi_bbox=None, s_thresh=60, bins=256, smooth_sigma=3):
    """
    Estimate representative grey from histogram of low-saturation pixels.
    Returns dict with mask and thresholds.
    (unchanged helper; returns hist info but NOT histogram arrays)
    """
    h, w = img_bgr.shape[:2]
    if roi_bbox is not None:
        x1,y1,x2,y2 = roi_bbox
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        roi = img_bgr[y1:y2, x1:x2]
    else:
        roi = img_bgr
    if roi is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vf = cv2.split(hsv)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    Lc = lab[:, :, 0]

    sat_mask = (Sc <= s_thresh)
    valid_mask = sat_mask & (Vf > 10) & (Vf < 245)

    vals_V = Vf[valid_mask].astype(np.int32)
    vals_L = Lc[valid_mask].astype(np.int32)

    if vals_V.size == 0:
        sat_mask = (Sc <= max(120, s_thresh))
        valid_mask = sat_mask & (Vf > 5) & (Vf < 250)
        vals_V = Vf[valid_mask].astype(np.int32)
        vals_L = Lc[valid_mask].astype(np.int32)
        if vals_V.size == 0:
            return None

    def analyze_values(values):
        hist, edges = np.histogram(values, bins=bins, range=(0,255))
        kernel = gaussian_kernel1d(sigma=smooth_sigma, radius=6)
        hist_s = np.convolve(hist.astype(np.float32), kernel, mode='same')
        edges_centers = (edges[:-1] + edges[1:]) / 2.0
        valid_range = (edges_centers >= 10) & (edges_centers <= 245)
        hist_masked = hist_s * valid_range
        peak_idx = int(np.argmax(hist_masked))
        peak_val = edges_centers[peak_idx]
        frac = 0.25
        thr = hist_s.max() * frac
        above = np.where(hist_s >= thr)[0]
        if above.size == 0:
            median = np.median(values)
            low = max(0, median - 20); high = min(255, median + 20)
            return {"median": float(median), "min": float(values.min()), "max": float(values.max()), "lower": int(low), "upper": int(high), "peak": float(peak_val)}
        left = peak_idx
        while left-1 >= 0 and hist_s[left-1] >= thr:
            left -= 1
        right = peak_idx
        while right+1 < hist_s.size and hist_s[right+1] >= thr:
            right += 1
        lower = int(max(0, edges_centers[left] - (edges[1]-edges[0])/2.0))
        upper = int(min(255, edges_centers[right] + (edges[1]-edges[0])/2.0))
        bin_mask = (values >= lower) & (values <= upper)
        if bin_mask.sum() > 0:
            med = float(np.median(values[bin_mask]))
            vmin = float(values[bin_mask].min())
            vmax = float(values[bin_mask].max())
        else:
            med = float(np.median(values))
            vmin = float(values.min()); vmax = float(values.max())
        return {"median": med, "min": vmin, "max": vmax, "lower": lower, "upper": upper, "peak": float(peak_val)}

    resV = analyze_values(vals_V)
    resL = analyze_values(vals_L)

    widthV = resV["upper"] - resV["lower"]
    widthL = resL["upper"] - resL["lower"]
    chosen = resV if widthV <= widthL else resL
    chosen_space = 'V' if chosen is resV else 'L'

    full_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hf, Sf, Vf_full = cv2.split(full_hsv)
    mask_v = cv2.inRange(Vf_full, chosen["lower"], chosen["upper"])
    mask_s = cv2.inRange(Sf, 0, max(60, s_thresh))
    mask_final = cv2.bitwise_and(mask_v, mask_s)
    kernel = np.ones((5,5), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
    return {
        "space": chosen_space,
        "median": chosen["median"],
        "min": chosen["min"],
        "max": chosen["max"],
        "lower": chosen["lower"],
        "upper": chosen["upper"],
        "mask": mask_final,
        "histV_info": resV,
        "histL_info": resL
    }

# ---------------- Methods (as before, unchanged except histogram replaced) ----------------
def method_template(img, template_gray):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < template_gray.shape[0] or gray.shape[1] < template_gray.shape[1]:
            return None
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < MATCH_THRESHOLD:
        return None
    h, w = template_gray.shape[:2]
    top_left = (int(max_loc[0]), int(max_loc[1]))
    x1, y1 = top_left
    roi = img[y1:y1+h, x1:x1+w].copy()
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(hsv[:, :, 1], 0, 90)
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[y1:y1+h, x1:x1+w] = mask_roi
    rect = np.array([[x1, y1], [x1+w, y1], [x1+w, y1+h], [x1, y1+h]])
    overlay = overlay_box(img, rect, COLORS["template"])
    score, metrics = score_mask(full_mask, img)
    return Result("template", full_mask, overlay, roi, score, {"conf": float(max_val)})

def method_keypoint_orb(img, template):
    try:
        orb = cv2.ORB_create(2000)
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:200]
        if len(matches) < 8:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        transformed = cv2.transform(np.array([corners]), M)[0].astype(int)
        mask_poly = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_poly, [transformed], 255)
        overlay = overlay_box(img, transformed, COLORS["keypoint_orb"])
        roi = img[np.min(transformed[:,1]):np.max(transformed[:,1])+1, np.min(transformed[:,0]):np.max(transformed[:,0])+1].copy()
        score, metrics = score_mask(mask_poly, img)
        return Result("keypoint_orb", mask_poly, overlay, roi, score, {"matches": len(matches)})
    except Exception:
        return None

# ... keep other methods (hough, slic, mser, watershed, grabcut, labhsv_bruteforce, precise_gray_expand) unchanged ...
# For brevity I omit repeating them here — they remain exactly as in your file.
# (In your actual file keep the implementations above as they were.)

# ---------------- Replaced histogram method ----------------
def method_histogram_gray(img, template_gray=None, roi_bbox=None, s_thresh=60, out_base: Path=None):
    """
    Replaced histogram method:
    - uses the same selection of low-saturation pixels as estimate_gray_from_histogram
    - computes smoothed histogram of chosen channel (V or L)
    - creates and saves a histogram visualization image into out_base/DEBUG_DIRNAME
    - returns Result(...), meta includes histogram info and path to saved histogram image
    """
    try:
        chosen_roi_bbox = None
        tm_conf = None
        if template_gray is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape[0] >= template_gray.shape[0] and gray.shape[1] >= template_gray.shape[1]:
                try:
                    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= MATCH_THRESHOLD:
                        h, w = template_gray.shape[:2]
                        x1,y1 = int(max_loc[0]), int(max_loc[1])
                        x2, y2 = x1 + w, y1 + h
                        chosen_roi_bbox = (x1,y1,x2,y2)
                        tm_conf = float(max_val)
                except cv2.error:
                    chosen_roi_bbox = None
        if chosen_roi_bbox is None and roi_bbox is not None:
            chosen_roi_bbox = roi_bbox

        # Use estimate_gray_from_histogram to get mask & meta
        est = estimate_gray_from_histogram(img, roi_bbox=chosen_roi_bbox, s_thresh=s_thresh)
        if est is None:
            return None
        mask = est["mask"]

        # Build overlay with box if available
        overlay = img.copy()
        big_mask, largest = largest_component_mask(mask)
        if largest is not None:
            box = cv2.boxPoints(cv2.minAreaRect(largest)).astype(int)
            overlay = overlay_box(img, box, COLORS["histogram_gray"])

        roi_img = None
        if chosen_roi_bbox is not None:
            x1,y1,x2,y2 = chosen_roi_bbox
            # clamp
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                roi_img = img[y1:y2, x1:x2].copy()

        # --- create histogram visualization from ROI (or whole image) ---
        # choose same valid pixels selection used in the estimator
        if chosen_roi_bbox is not None:
            x1,y1,x2,y2 = chosen_roi_bbox
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
            area = img[y1:y2, x1:x2]
        else:
            area = img

        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        Hc, Sc, Vf = cv2.split(hsv)
        lab = cv2.cvtColor(area, cv2.COLOR_BGR2LAB)
        Lc = lab[:, :, 0]

        sat_mask = (Sc <= s_thresh)
        valid_mask = sat_mask & (Vf > 10) & (Vf < 245)
        vals_V = Vf[valid_mask].astype(np.int32)
        vals_L = Lc[valid_mask].astype(np.int32)
        if vals_V.size == 0:
            sat_mask = (Sc <= max(120, s_thresh))
            valid_mask = sat_mask & (Vf > 5) & (Vf < 250)
            vals_V = Vf[valid_mask].astype(np.int32)
            vals_L = Lc[valid_mask].astype(np.int32)
            if vals_V.size == 0:
                return None

        # compute histogram + smoothed curve (like estimator)
        bins = 256
        histV, edges = np.histogram(vals_V, bins=bins, range=(0,255))
        histL, _ = np.histogram(vals_L, bins=bins, range=(0,255))
        kernel = gaussian_kernel1d(sigma=3, radius=6)
        histV_s = np.convolve(histV.astype(np.float32), kernel, mode='same')
        histL_s = np.convolve(histL.astype(np.float32), kernel, mode='same')
        edges_centers = (edges[:-1] + edges[1:]) / 2.0

        # decide chosen channel same as estimator: pick narrower width
        # reuse est["histV_info"] and est["histL_info"] for lower/upper, but if not present, fallback
        lower = est.get("lower", None) if isinstance(est, dict) else None
        upper = est.get("upper", None) if isinstance(est, dict) else None
        # actually est structure: has 'min','max','lower','upper','median' keys at top level
        # prefer est["lower"]/["upper"] if present
        lower_v = est.get("lower", None)
        upper_v = est.get("upper", None)
        median_v = est.get("median", None)

        # create histogram image (w=512, h=200)
        Hh = 200; Ww = 512
        hist_img = np.zeros((Hh, Ww, 3), dtype=np.uint8)
        # choose which smoothed hist to draw: use V histogram if est['space']=='V'
        chosen_space = est.get("space", "V")
        hist_plot = histV_s if chosen_space == 'V' else histL_s
        # normalize to height
        if hist_plot.max() > 0:
            hp = (hist_plot / (hist_plot.max())) * (Hh - 20)
        else:
            hp = hist_plot
        step = Ww / float(bins)
        for i in range(bins):
            x = int(i * step)
            x2 = int((i+1) * step)
            hval = int(hp[i])
            cv2.rectangle(hist_img, (x, Hh-1), (x2, Hh-1 - hval), (180,180,180), -1)

        # draw vertical lines for lower/upper/peak if available
        if lower_v is not None:
            x_l = int((lower_v / 255.0) * (Ww - 1))
            cv2.line(hist_img, (x_l, 0), (x_l, Hh-1), (0, 0, 255), 2)  # red = lower
        if upper_v is not None:
            x_u = int((upper_v / 255.0) * (Ww - 1))
            cv2.line(hist_img, (x_u, 0), (x_u, Hh-1), (0, 255, 0), 2)  # green = upper
        if median_v is not None:
            x_m = int((median_v / 255.0) * (Ww - 1))
            cv2.line(hist_img, (x_m, 0), (x_m, Hh-1), (255, 0, 0), 2)  # blue = median

        # annotate text
        txt = f"space={chosen_space} median={est.get('median'):.1f} lower={est.get('lower')} upper={est.get('upper')}"
        cv2.putText(hist_img, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)

        # save histogram image if out_base provided
        hist_rel_path = None
        if out_base is not None:
            debug_dir = Path(out_base) / DEBUG_DIRNAME
            ensure_dir(debug_dir)
            # prefer png for histogram visualization
            hist_fn = debug_dir / f"histogram_{out_base.name}_{str(np.random.randint(0,1e6))}.png"
            # Better filename: we will attempt to name using ROI bbox when available
            # but the process_all caller will know folder/file names, so it will also save meta info
            cv2.imwrite(str(hist_fn), hist_img)
            hist_rel_path = str(hist_fn)

        score, metrics = score_mask(mask, img)
        meta = {
            "median": est.get("median"),
            "min": est.get("min"),
            "max": est.get("max"),
            "lower": est.get("lower"),
            "upper": est.get("upper"),
            "tm_conf": tm_conf,
            "histogram_png": hist_rel_path
        }
        return Result("histogram_gray", mask, overlay, roi_img, score, meta)
    except Exception:
        return None

# ---------------- Orchestration ----------------
def process_all(input_dir: Path, template_path: Path, out_base: Path):
    ensure_dir(out_base)
    ensure_dir(COMPARE_DIR)
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    folders = []
    for p in input_dir.rglob("*"):
        if p.is_dir():
            last = get_last_image_in_folder(p)
            if last:
                folders.append((p, last))
    if not folders:
        print("[WARN] No folders with images found.")
        return

    for folder, last_image in folders:
        try:
            print(f"\n➡ Processing folder: {folder} -> last image: {last_image.name}")
            img = cv2.imread(str(last_image))
            if img is None:
                print(f"[ERR] Cannot read {last_image}")
                continue
            results = []

            # 1) template
            r = method_template(img, template_gray)
            if r:
                results.append(r)
                save_with_ext(out_base / f"template_{folder.name}_{last_image.name}", r.overlay)
                save_with_ext(out_base / MASK_DIRNAME / f"template_{folder.name}_{last_image.name}", r.mask)
                if r.roi is not None:
                    save_with_ext(out_base / ROI_DIRNAME / f"template_{folder.name}_{last_image.name}", r.roi)

            # 2) keypoint ORB
            r = method_keypoint_orb(img, template)
            if r:
                results.append(r)
                save_with_ext(out_base / f"keypoint_orb_{folder.name}_{last_image.name}", r.overlay)
                save_with_ext(out_base / MASK_DIRNAME / f"keypoint_orb_{folder.name}_{last_image.name}", r.mask)
                if r.roi is not None:
                    save_with_ext(out_base / ROI_DIRNAME / f"keypoint_orb_{folder.name}_{last_image.name}", r.roi)

            # ... (other methods kept as before) ...

            # 10) histogram-based gray estimation (REPLACED) - pass out_base so it can save histogram png
            r = method_histogram_gray(img, template_gray=template_gray, out_base=out_base)
            if r:
                results.append(r)
                save_with_ext(out_base / f"histogram_gray_{folder.name}_{last_image.name}", r.overlay)
                save_with_ext(out_base / MASK_DIRNAME / f"histogram_gray_{folder.name}_{last_image.name}", r.mask)
                if r.roi is not None:
                    save_with_ext(out_base / ROI_DIRNAME / f"histogram_gray_{folder.name}_{last_image.name}", r.roi)
                # save meta info as text
                meta_path = out_base / DEBUG_DIRNAME / f"histogram_meta_{folder.name}_{last_image.name}.txt"
                ensure_dir(meta_path.parent)
                with open(meta_path, "w") as fh:
                    fh.write(str(r.meta))

            # Save debug V and S channels
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            Hc, Sc, Vf = cv2.split(hsv)
            save_with_ext(out_base / DEBUG_DIRNAME / f"V_{folder.name}_{last_image.name}", Vf)
            save_with_ext(out_base / DEBUG_DIRNAME / f"S_{folder.name}_{last_image.name}", Sc)

            # Compose comparison montage from available overlays
            montage = build_comparison_image(results, img.shape)
            if montage is not None:
                save_with_ext(COMPARE_DIR / f"{folder.name}_{last_image.name}", montage)

            # Score selection
            if results:
                best = max(results, key=lambda x: x.score if x.score is not None else 0.0)
                print(f"[RESULT] best method for {folder.name}/{last_image.name}: {best.method} score={best.score:.3f} meta={best.meta}")
                save_with_ext(out_base / "best" / f"{best.method}_{folder.name}_{last_image.name}", best.overlay)
                save_with_ext(out_base / "best" / MASK_DIRNAME / f"{best.method}_{folder.name}_{last_image.name}", best.mask)
            else:
                print("[RESULT] no method produced result for this image.")

        except Exception as e:
            print(f"[ERROR] processing folder {folder}: {e}")
            traceback.print_exc()

    print(f"\nDone. Results saved under: {out_base}")

# ---------------- Small test runner ----------------
def run_quick_test_on_examples(example_paths):
    template = cv2.imread(str(TEMPLATE_IMG_PATH))
    if template is None:
        print("Template not found.")
        return
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    for p in example_paths:
        p = Path(p)
        if not p.exists():
            print(f"[TEST] {p} not found")
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"[TEST] cannot read {p}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < template_gray.shape[0] or gray.shape[1] < template_gray.shape[1]:
            print(f"[TEST] {p.name}: image smaller than template -> skip")
            continue
        res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        minv, maxv, minloc, maxloc = cv2.minMaxLoc(res)
        h, w = template_gray.shape[:2]
        print(f"[TEST] {p.name}: match_conf={maxv:.3f}, bbox=({maxloc[0]},{maxloc[1]}) - ({maxloc[0]+w},{maxloc[1]+h})")
# ---------------- CLI runner (pipeline + quick test on last images) ----------------
if __name__ == "__main__":
    ensure_dir(OUTPUT_TMP_DIR)
    print("INPUT_IMAGES_DIR:", INPUT_IMAGES_DIR)
    print("OUTPUT base:", OUTPUT_TMP_DIR)
    process_all(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH, OUTPUT_TMP_DIR)

    # build example list from the same folders (last image of each)
    example_list = []
    for p in INPUT_IMAGES_DIR.rglob("*"):
        if p.is_dir():
            last = get_last_image_in_folder(p)
            if last:
                example_list.append(last)
    if example_list:
        print("\nRunning quick template-match test on last images from folders...")
        run_quick_test_on_examples(example_list)
    else:
        print("\nNo last images found for quick tests.")

