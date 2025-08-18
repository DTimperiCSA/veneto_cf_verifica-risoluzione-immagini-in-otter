from src.segmentation.infer_cnn import segment_with_cnn
import cv2
from pathlib import Path
import torch
import numpy as np

from src.paths import *

import cv2
import numpy as np
import torch
from pathlib import Path
from src.segmentation.infer_cnn import segment_with_cnn

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
COL_KP = (255, 0, 0)     # blue (BGR) - keypoint
COL_ROI_HIST = (255, 0, 255)  # magenta - roi-based histogram

# ---------- Helpers ----------

def extract_mask(res):
    """Support both dict with 'mask' or mask directly."""
    if res is None:
        return None
    if isinstance(res, dict) and "mask" in res:
        return res["mask"]
    return res


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

# Helper: overlay mask on original image
def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.5):
    """Overlay mask on image with given color and transparency."""
    if mask is None:
        return img
    overlay = img.copy()
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)

# Helper: keypoint method
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

# Helper: histogram method
def method_histogram_roi(img, template_gray):
    hist_template = cv2.calcHist([template_gray], [0], None, [256], [0, 256])
    hist_template = cv2.normalize(hist_template, hist_template).flatten()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(mask)
    if max_val < 0.5:
        return None
    th, tw = template_gray.shape
    top_left = max_loc
    bottom_right = (top_left[0] + tw, top_left[1] + th)
    roi = img_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if roi.size == 0:
        return None
    hist_roi = cv2.calcHist([roi], [0], None, [256], [0, 256])
    hist_roi = cv2.normalize(hist_roi, hist_roi).flatten()
    if cv2.compareHist(hist_template, hist_roi, cv2.HISTCMP_CORREL) < 0.7:
        return None
    mask_img = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask_img, top_left, bottom_right, 255, -1)
    return mask_img

# Main comparison function
def compare_methods(image_path, template_path, model_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    template = cv2.imread(str(template_path))
    if img is None or template is None:
        print(f"[WARN] Could not read {image_path} or {template_path}")
        return
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CNN
    mask_cnn = segment_with_cnn(model_path, image_path, device=device)

    # Keypoint
    mask_kp = extract_mask(method_keypoint_orb(img, template))
    mask_hist = extract_mask(method_histogram_roi(img, template_gray))

    # Create overlays
    cnn_overlay = overlay_mask(img, mask_cnn, color=(0, 0, 255))
    kp_overlay = overlay_mask(img, mask_kp, color=(0, 255, 0))
    hist_overlay = overlay_mask(img, mask_hist, color=(255, 0, 0))

    # Side-by-side comparison image
    combined = np.hstack([
        img,
        cnn_overlay,
        kp_overlay,
        hist_overlay
    ])

    # Add labels on top of each panel
    panel_width = img.shape[1]
    panel_height = img.shape[0]
    labels = ["Original", "CNN", "Keypoint (ORB)", "Histogram ROI"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    for i, label in enumerate(labels):
        x = i * panel_width + 10
        y = 30
        # draw text with black outline for readability
        cv2.putText(combined, label, (x, y), font, font_scale, (0,0,0), font_thickness+2, cv2.LINE_AA)
        cv2.putText(combined, label, (x, y), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)

    # Save single comparison image
    cv2.imwrite(str(out_dir / f"{image_path.stem}_comparison.png"), combined)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    images_dir = Path(r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\data\test_images")
    template_path = TEMPLATE_IMG_PATH
    model_path = Path(r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\model\tiffen_segmenter.pth")
    out_dir = OUTPUT_TMP_DIR / "comparison_results"

    # Create output dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Valid extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in valid_exts:
            print(f"Processing {img_path.name}...")
            compare_methods(img_path, template_path, model_path, out_dir)

