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
from src.utils import *


# ---------- CONFIG ----------
MATCH_THRESHOLD = 0.60
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

OUTPUT_BASE = Path(OUTPUT_TMP_DIR) / "RES_BRUTE_FORCE"
ROI_DIR = OUTPUT_BASE / "roi"
MASK_DIR = OUTPUT_BASE / "masks"
DEBUG_DIR = OUTPUT_BASE / "debug"

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

# ---------- Orchestration ----------
# ---------- Orchestration ----------
def process_all_images(input_dir: Path, template_path: Path, out_base: Path):
    ensure_dir(out_base)
    ensure_dir(ROI_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(DEBUG_DIR)

    input_dir = Path(r"C:\Users\andre\Desktop\x_transfer")

    # Load template
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Find all valid images in the folder (recursively)
    images = [
        img_path for img_path in input_dir.rglob("*")
        if img_path.is_file() and is_valid_image_file(img_path)
    ]
    if not images:
        print("[WARN] No valid images found.")
        return

    print(f"\n➡ Processing {len(images)} images from: {input_dir}")

    for image_path in images:
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[ERR] Cannot read {image_path}")
                continue

            base_name = image_path.stem
            ext = image_path.suffix
            folder_name = image_path.parent.name  # keeps track of which subfolder the image was in

            # keypoint
            kp_res = method_keypoint_orb(img, template)

            if kp_res:
                save_with_ext(out_base / f"{folder_name}_{base_name}{ext}", kp_res["overlay"])
            else:
                print("not possible")

        except Exception as e:
            print(f"[ERROR] processing image {image_path}: {e}")
            traceback.print_exc()

    print(f"\nDone. Results saved under: {out_base}")


def process_all(input_dir: Path, template_path: Path, out_base: Path):
    ensure_dir(out_base)
    ensure_dir(ROI_DIR)
    ensure_dir(MASK_DIR)
    ensure_dir(DEBUG_DIR)

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

            # keypoint
            kp_res = method_keypoint_orb(img, template)

            if kp_res:
                save_with_ext(out_base / f"{folder.name}_{base_name}{ext}", kp_res["overlay"])
            else:
                print("errore non esiste")




        except Exception as e:
            print(f"[ERROR] processing folder {folder}: {e}")
            traceback.print_exc()

    print(f"\nDone. Results saved under: {out_base}")

# ---------- CLI ----------
if __name__ == "__main__":
    ensure_dir(OUTPUT_BASE)
    INPUT_IMAGES_DIR = Path(r"C:\Users\andre\Desktop\x_transfer")
    print("INPUT_IMAGES_DIR:", INPUT_IMAGES_DIR)
    print("OUTPUT base:", OUTPUT_BASE)
    
    
    process_all_images(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH, OUTPUT_BASE)

    #process_all(INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH, OUTPUT_BASE)
