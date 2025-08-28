import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from src.segmentation.unet import UNet
from src.paths import *

# ========= CONFIG =========
INPUT_FOLDER = Path(CONSERVATORIO_DIR / "B001")
OUTPUT_DIR = Path(OUTPUT_TMP_DIR / "unet_segmentation")
OUTPUT_MASKS = OUTPUT_DIR / "masks"
OUTPUT_BBOX = OUTPUT_DIR / "bbox"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5  # You can adjust this threshold depending on your model's output

OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)
OUTPUT_BBOX.mkdir(parents=True, exist_ok=True)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ========= MODEL =========
model = UNet(n_channels=3, n_classes=1)
checkpoint = torch.load(
    Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\veneto_cf_verifica-risoluzione-immagini-in-otter\model\checkpoints\checkpoint_epoch_26.pth"),
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
template = cv2.imread(str(TEMPLATE_IMG_PATH))
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_w, template_h = template_gray.shape[::-1]

def contains_chromatic_band(image_path: Path, threshold: float = 0.7) -> bool:
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val >= threshold

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

# ========= PROCESS FOLDER =========
def process_folder(folder: Path):
    for img_path in folder.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue

        try:
            if not contains_chromatic_band(img_path):
                continue

            image, mask = predict_mask(img_path)

            # Save the mask
            mask_out_path = OUTPUT_MASKS / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)

            # Save the image with bounding box
            bbox_img = image.copy()
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)

            bbox_out_path = OUTPUT_BBOX / f"{img_path.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))

            print(f"✔ Processed {img_path.name}")
        except Exception as e:
            print(f"Error with {img_path.name}: {e}")

# ========= RUN =========
if __name__ == "__main__":
    processed_count = 0

    PATH = Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\dataset\val_images")
    MAX_IMAGES = 100

    # Get all images from the input folder
    all_images = [p for p in PATH.rglob("*") if p.suffix.lower() in VALID_EXTS]
    print(f"[DEBUG] Total images found: {len(all_images)}")

    all_images = all_images[:MAX_IMAGES]  # Limit the number of images to process

    for img_path in all_images:
        if processed_count >= MAX_IMAGES:
            print(f"[DEBUG] Reached max limit of {MAX_IMAGES} images. Stopping.")
            break

        print(f"[DEBUG] Processing: {img_path}")

        try:
            image, mask = predict_mask(img_path)
            print(f"[DEBUG] Predicted mask. Image shape: {image.shape}, Mask shape: {mask.shape}")

            # Save the predicted mask
            mask_out_path = OUTPUT_MASKS / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(mask_out_path), mask)
            print(f"[DEBUG] Mask saved to: {mask_out_path}")

            # Save bounding box
            bbox_img = image.copy()
            bbox = get_bbox_from_mask(mask)
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), thickness=6)
                print(f"[DEBUG] Bounding box: x={x}, y={y}, w={w}, h={h}")
            else:
                print(f"[DEBUG] No object found in the mask.")

            bbox_out_path = OUTPUT_BBOX / f"{img_path.stem}_bbox.png"
            cv2.imwrite(str(bbox_out_path), cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR))
            print(f"[DEBUG] Image with bounding box saved to: {bbox_out_path}")

            processed_count += 1
            print(f"✔ Processed {img_path.name}")
        except Exception as e:
            print(f"[ERROR] Error with {img_path.name}: {e}")
