# src/histkey_compare.py
"""
Compare ROI-histogram vs keypoint (ORB).
- processes only the last image of each folder under INPUT_IMAGES_DIR
- runs two methods:
    1) ROI-histogram (requires template match; sample low-sat pixels inside template ROI,
       compute lower/upper on V channel, segment whole image using those bounds)
    2) keypoint ORB (feature-based localization -> polygon mask)
- builds comparison overlay, computes IoU/area metrics, saves images + stats + histogram plots.
- colors: BLUE = keypoint, MAGENTA = ROI-histogram
"""
from pathlib import Path
import cv2
import numpy as np
import traceback

from src.paths import INPUT_IMAGES_DIR, OUTPUT_TMP_DIR, TEMPLATE_IMG_PATH

# ---------- CONFIG ----------
MATCH_THRESHOLD = 0.60
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

OUTPUT_BASE = Path(OUTPUT_TMP_DIR) / "histkey_compare"
COMPARE_DIR = OUTPUT_BASE / "confronti"
ROI_DIR = OUTPUT_BASE / "roi"
MASK_DIR = OUTPUT_BASE / "masks"
DEBUG_DIR = OUTPUT_BASE / "debug"
BEST_DIR = OUTPUT_BASE / "best"

COL_KP = (255, 0, 0)     # blue (BGR) - keypoint
COL_ROI_HIST = (255, 0, 255)  # magenta - roi-based histogram

# ---------- Helpers ----------
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

def mask_iou(mask1, mask2):
    if mask1 is None or mask2 is None:
        return 0.0
    inter = np.logical_and(mask1>0, mask2>0).sum()
    union = np.logical_or(mask1>0, mask2>0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def box_from_mask(mask):
    big, cnt = largest_component_mask(mask)
    if cnt is None:
        return None
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(int)
    return box

def draw_histogram_image(values, bins=256, size=(600,200), color=(255,255,255)):
    hist, edges = np.histogram(values, bins=bins, range=(0,255))
    hist = hist.astype(np.float32)
    hist_max = hist.max() if hist.max() > 0 else 1.0
    h, w = size[1], size[0]
    canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30
    for i in range(bins):
        val = hist[i]
        bar_h = int((val / hist_max) * (h - 20))
        x1 = int(i * (w / bins))
        x2 = int((i + 1) * (w / bins))
        cv2.rectangle(canvas, (x1, h - 10), (x2, h - 10 - bar_h), color, -1)
    cv2.rectangle(canvas, (0,0), (w-1, h-1), (200,200,200), 1)
    return canvas

def overlay_boxes_on_image(img, box_kp, box_roi_hist):
    out = img.copy()
    if box_kp is not None:
        cv2.drawContours(out, [np.array(box_kp)], -1, COL_KP, 3)
    if box_roi_hist is not None:
        cv2.drawContours(out, [np.array(box_roi_hist)], -1, COL_ROI_HIST, 3)
    # legend background
    cv2.rectangle(out, (10,10), (480, 90), (0,0,0), -1)
    cv2.putText(out, "BLUE = keypoint (ORB)", (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COL_KP, 2, cv2.LINE_AA)
    cv2.putText(out, "MAGENTA = ROI-histogram (template ROI)", (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_ROI_HIST, 2, cv2.LINE_AA)
    return out

# ---------- Estimation helper ----------
def estimate_gray_hist_values_from_roi(roi, s_thresh=60, bins=256):
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

    hist, edges = np.histogram(vals_V, bins=bins, range=(0,255))
    peak_idx = int(np.argmax(hist))
    thr = hist.max() * 0.25
    left = peak_idx
    while left-1 >= 0 and hist[left-1] >= thr:
        left -= 1
    right = peak_idx
    while right+1 < hist.size and hist[right+1] >= thr:
        right += 1
    lower = int(max(0, edges[left]))
    upper = int(min(255, edges[right+1] if (right+1) < edges.size else edges[-1]))
    median_v = float(np.median(vals_V)) if vals_V.size else float(peak_idx)
    return {"median": median_v, "lower": lower, "upper": upper, "vals_V": vals_V}

# ---------- Methods ----------
def method_histogram_roi(img, template_gray, s_thresh=60):
    """
    1) locate template in the image (requires match >= MATCH_THRESHOLD)
    2) build histogram of low-saturation V inside that ROI
    3) derive lower/upper and segment full image using those bounds
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < template_gray.shape[0] or gray.shape[1] < template_gray.shape[1]:
            return None
        try:
            res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
        except cv2.error:
            return None
        if max_val < MATCH_THRESHOLD:
            return None
        h, w = template_gray.shape[:2]
        x1, y1 = int(max_loc[0]), int(max_loc[1])
        x2, y2 = x1 + w, y1 + h
        # roi with small margin
        margin = 6
        rx1 = max(0, x1 - margin); ry1 = max(0, y1 - margin)
        rx2 = min(img.shape[1], x2 + margin); ry2 = min(img.shape[0], y2 + margin)
        roi = img[ry1:ry2, rx1:rx2].copy()
        if roi is None or roi.size == 0:
            return None
        est = estimate_gray_hist_values_from_roi(roi, s_thresh=s_thresh)
        if est is None:
            return None
        lower = est["lower"]; upper = est["upper"]
        # segment full image using those bounds
        hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_v = cv2.inRange(hsv_full[:,:,2], lower, upper)
        mask_s = cv2.inRange(hsv_full[:,:,1], 0, max(60, s_thresh))
        mask_final = cv2.bitwise_and(mask_v, mask_s)
        kernel = np.ones((5,5), np.uint8)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel, iterations=1)
        # keep largest component only (likely ruler)
        big_mask, largest = largest_component_mask(mask_final)
        if big_mask is not None:
            mask_final = big_mask
        overlay = img.copy()
        if largest is not None:
            box = cv2.boxPoints(cv2.minAreaRect(largest)).astype(int)
            cv2.drawContours(overlay, [box], -1, COL_ROI_HIST, 3)
        meta = {"lower": lower, "upper": upper, "median": est["median"], "tm_conf": float(max_val), "vals_V_sample_count": int(est["vals_V"].size)}
        return {"method": "roi_hist", "mask": mask_final, "overlay": overlay, "roi": roi, "meta": meta}
    except Exception:
        return None

def method_keypoint_orb(img, template):
    try:
        orb = cv2.ORB_create(3000)
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(img, None)
        if des1 is None or des2 is None:
            return None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:3000]  # cap
        if len(matches) < 8:
            return None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, inliers_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None:
            return None
        h, w = template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 2)
        transformed = cv2.transform(np.array([corners]), M)[0].astype(int)
        # clamp coords
        transformed[:,0] = np.clip(transformed[:,0], 0, img.shape[1]-1)
        transformed[:,1] = np.clip(transformed[:,1], 0, img.shape[0]-1)
        mask_poly = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_poly, [transformed], 255)
        # keep largest connected comp (to remove tiny noisy polygons)
        big, largest = largest_component_mask(mask_poly)
        if big is not None:
            mask_poly = big
        overlay = img.copy()
        cv2.drawContours(overlay, [transformed], -1, COL_KP, 3)
        # roi bounding rect from transformed poly
        minx = np.min(transformed[:,0]); maxx = np.max(transformed[:,0])
        miny = np.min(transformed[:,1]); maxy = np.max(transformed[:,1])
        if maxx<=minx or maxy<=miny:
            roi = None
        else:
            roi = img[miny:maxy+1, minx:maxx+1].copy()
        meta = {"matches": len(matches)}
        return {"method": "keypoint", "mask": mask_poly, "overlay": overlay, "roi": roi, "meta": meta}
    except Exception:
        return None

# ---------- Compare + save ----------
def compare_and_save_two(roi_hist_res, kp_res, img, folder_name, base_name, out_base, ext):
    ensure_dir(out_base)
    ensure_dir(COMPARE_DIR)
    ensure_dir(ROI_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(DEBUG_DIR)
    ensure_dir(BEST_DIR)

    mask_r = roi_hist_res["mask"] if roi_hist_res else None
    mask_k = kp_res["mask"] if kp_res else None

    iou_rk = mask_iou(mask_r, mask_k)
    area_r = int(np.count_nonzero(mask_r)) if mask_r is not None else 0
    area_k = int(np.count_nonzero(mask_k)) if mask_k is not None else 0

    box_r = box_from_mask(mask_r) if mask_r is not None else None
    box_k = box_from_mask(mask_k) if mask_k is not None else None

    combined = overlay_boxes_on_image(img, box_k, box_r)
    combined_name = COMPARE_DIR / f"kp_vs_roi_{folder_name}_{base_name}{ext}"
    save_with_ext(combined_name, combined)

    # save masks and rois (with extension)
    if mask_r is not None:
        save_with_ext(MASK_DIR / f"roi_hist_mask_{folder_name}_{base_name}{ext}", mask_r)
    if mask_k is not None:
        save_with_ext(MASK_DIR / f"kp_mask_{folder_name}_{base_name}{ext}", mask_k)
    if roi_hist_res and roi_hist_res.get("roi") is not None:
        save_with_ext(ROI_DIR / f"roi_hist_roi_{folder_name}_{base_name}{ext}", roi_hist_res["roi"])
    if kp_res and kp_res.get("roi") is not None:
        save_with_ext(ROI_DIR / f"kp_roi_{folder_name}_{base_name}{ext}", kp_res["roi"])

    # histogram images (cap samples to 3000)
    def save_vals_hist(vals, name, color):
        if vals is None or vals.size == 0:
            vals_arr = np.array([], dtype=np.int32)
        else:
            vals_arr = np.array(vals)
            if vals_arr.size > 3000:
                inds = np.linspace(0, vals_arr.size-1, 3000, dtype=np.int32)
                vals_arr = vals_arr[inds]
        hist_img = draw_histogram_image(vals_arr, bins=256, size=(600,200), color=color)
        save_with_ext(DEBUG_DIR / f"{name}_{folder_name}_{base_name}{ext}", hist_img)

    vals_r = None
    if roi_hist_res and "meta" in roi_hist_res and "vals_V" in roi_hist_res["meta"]:
        vals_r = np.array(roi_hist_res["meta"]["vals_V"])
    save_vals_hist(vals_r, "hist_roi_values", COL_ROI_HIST)

    # save stats
    stats_path = DEBUG_DIR / f"compare_stats_{folder_name}_{base_name}.txt"
    with open(stats_path, "w") as fh:
        fh.write(f"folder: {folder_name}\nfile: {base_name}\n\n")
        fh.write("ROI HIST method:\n")
        fh.write(str(roi_hist_res["meta"]) + "\n" if roi_hist_res and "meta" in roi_hist_res else "None\n")
        fh.write(f"area_roi (px): {area_r}\n\n")
        fh.write("KEYPOINT method:\n")
        fh.write(str(kp_res["meta"]) + "\n" if kp_res and "meta" in kp_res else "None\n")
        fh.write(f"area_kp (px): {area_k}\n\n")
        fh.write(f"IoU roi vs kp: {iou_rk:.4f}\n")

    # pick "best" by IoU+area heuristic: if roi_hist exists and area>0 choose it, else use kp
    best_choice = None
    if roi_hist_res and area_r > 0:
        best_choice = ("roi_hist", roi_hist_res)
    elif kp_res and area_k > 0:
        best_choice = ("keypoint", kp_res)

    if best_choice:
        name, res = best_choice
        save_with_ext(BEST_DIR / f"{name}_{folder_name}_{base_name}{ext}", res["overlay"])
        if res.get("mask") is not None:
            save_with_ext(BEST_DIR / "masks" / f"{name}_{folder_name}_{base_name}{ext}", res["mask"])

    print(f"[COMPARE] saved combined image/stats for {folder_name}/{base_name} -> IoU rk={iou_rk:.3f}, areas: roi={area_r}, kp={area_k}")

# ---------- Orchestration ----------
def process_all(input_dir: Path, template_path: Path, out_base: Path):
    ensure_dir(out_base)
    ensure_dir(COMPARE_DIR)
    ensure_dir(ROI_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(DEBUG_DIR)
    ensure_dir(BEST_DIR)

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
            base_name = last_image.stem
            ext = last_image.suffix

            # roi-histogram (requires template)
            roi_hist_res = method_histogram_roi(img, template_gray)

            # keypoint
            kp_res = method_keypoint_orb(img, template)

            # save individual overlays/masks/roi if present (preserve extension)
            if roi_hist_res:
                save_with_ext(out_base / f"roi_hist_{folder.name}_{base_name}{ext}", roi_hist_res["overlay"])
                if roi_hist_res.get("mask") is not None:
                    save_with_ext(MASK_DIR / f"roi_hist_mask_{folder.name}_{base_name}{ext}", roi_hist_res["mask"])
                if roi_hist_res.get("roi") is not None:
                    save_with_ext(ROI_DIR / f"roi_hist_roi_{folder.name}_{base_name}{ext}", roi_hist_res["roi"])
            if kp_res:
                save_with_ext(out_base / f"keypoint_{folder.name}_{base_name}{ext}", kp_res["overlay"])
                if kp_res.get("mask") is not None:
                    save_with_ext(MASK_DIR / f"kp_mask_{folder.name}_{base_name}{ext}", kp_res["mask"])
                if kp_res.get("roi") is not None:
                    save_with_ext(ROI_DIR / f"kp_roi_{folder.name}_{base_name}{ext}", kp_res["roi"])

            # compare and save combined outputs/stats
            compare_and_save_two(roi_hist_res, kp_res, img, folder.name, base_name, out_base, ext)

        except Exception as e:
            print(f"[ERROR] processing folder {folder}: {e}")
            traceback.print_exc()

    print(f"\nDone. Results saved under: {out_base}")

# ---------- CLI ----------
if __name__ == "__main__":
    ensure_dir(OUTPUT_TMP_DIR / "histkey_compare")
    print("INPUT_IMAGES_DIR:", INPUT_IMAGES_DIR)
    print("OUTPUT base:", OUTPUT_TMP_DIR / "histkey_compare")
    process_all(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH, OUTPUT_TMP_DIR / "histkey_compare")
