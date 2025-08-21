# ==========================================
# prepare_dataset_with_no_ruler.py
# ==========================================

from pathlib import Path
import shutil
import random
from PIL import Image
import albumentations as A
import numpy as np

from src.paths import *

# =========================
# Albumentations augmentations
# =========================
def get_augmentations(img_size=512):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(0.8, 1.0),
                            ratio=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.5),
    ])


def augment_and_save(img_path, mask_path, out_img_dir, out_mask_dir, aug_index):
    aug = get_augmentations()
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) if mask_path.exists() \
        else np.zeros(img.shape[:2], dtype=np.uint8)

    augmented = aug(image=img, mask=mask)

    aug_img = Image.fromarray(np.uint8(augmented['image']))
    aug_mask = Image.fromarray(np.uint8(augmented['mask']))

    img_name = img_path.stem + f"_aug{aug_index}" + img_path.suffix
    mask_name = mask_path.stem + f"_aug{aug_index}" + mask_path.suffix
    aug_img.save(out_img_dir / img_name)
    aug_mask.save(out_mask_dir / mask_name)


# =========================
# Build dataset with 'no ruler' images
# =========================
# =========================
# Build dataset con ricerca flessibile
# =========================
def build_dataset(mask_dir, input_images_root, tmp_out):
    mask_dir = Path(mask_dir)
    input_images_root = Path(input_images_root)
    tmp_out = Path(tmp_out)

    images_out = tmp_out / "images"
    masks_out = tmp_out / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    paired = []

    for mask_path in sorted(mask_dir.glob("*")):
        if not mask_path.is_file():
            continue

        stem = mask_path.stem  # es: B001.015_0007
        dir_name = stem.split("_")[0]  # es: B001.015
        number_part = stem.split("_")[1] if "_" in stem else stem
        folder = input_images_root / dir_name

        if not folder.exists():
            print(f"⚠️ Skipping {mask_path.name}: folder {folder} not found")
            continue

        # Trova immagine col righello cercando numero nel nome
        img_candidates = [p for p in folder.glob("*") if p.suffix.lower() in valid_exts]
        ruler_img = next((c for c in img_candidates if number_part in c.stem), None)
        if ruler_img is None:
            print(f"⚠️ No ruler image found for {mask_path.name} in {folder}")
            continue

        # Copia immagine col righello + mask
        dst_img = images_out / ruler_img.name
        dst_mask = masks_out / mask_path.name
        shutil.copy(ruler_img, dst_img)
        shutil.copy(mask_path, dst_mask)
        paired.append((dst_img, dst_mask))

        # Trova prima immagine della cartella che non è quella del righello
        no_ruler_img = next((c for c in img_candidates if c != ruler_img), None)
        if no_ruler_img is None:
            print(f"⚠️ No 'no_ruler' image found in {folder}")
            continue

        dst_name = f"noruler_{stem}{no_ruler_img.suffix}"
        dst_img_nr = images_out / dst_name
        dst_mask_nr = masks_out / f"noruler_{stem}.png"

        shutil.copy(no_ruler_img, dst_img_nr)

        # Crea mask nera con stesse dimensioni
        with Image.open(no_ruler_img) as im:
            black = Image.new("L", im.size, 0)
            black.save(dst_mask_nr)

        paired.append((dst_img_nr, dst_mask_nr))

    print(f"✅ Built paired dataset: {len(paired)} samples")
    return images_out, masks_out


# =========================
# Dataset split + augmentation
# =========================
def split_dataset(images_dir, masks_dir, output_dir,
                  train_ratio=0.7, val_ratio=0.15,
                  seed=42, augment_factor=2):

    random.seed(seed)
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    all_images = [p for p in images_dir.glob("*") if p.is_file()]
    random.shuffle(all_images)

    total = len(all_images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    splits = {
        "train": all_images[:train_count],
        "val": all_images[train_count:train_count + val_count],
        "test": all_images[train_count + val_count:]
    }

    for split_name, files in splits.items():
        print(f"Processing {split_name}...")
        img_out = output_dir / f"{split_name}_images"
        mask_out = output_dir / f"{split_name}_masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            mask_path = masks_dir / img_path.name
            # Copia immagine originale + mask
            shutil.copy(img_path, img_out / img_path.name)
            if not mask_path.exists():
                with Image.open(img_path) as img:
                    black_mask = Image.new("L", img.size, 0)
                    black_mask.save(mask_out / img_path.name)
            else:
                shutil.copy(mask_path, mask_out / mask_path.name)

            # Solo training: augmentations
            if split_name == "train":
                for i in range(augment_factor):
                    augment_and_save(img_path, mask_path, img_out, mask_out, i)

        total_imgs = len(files) * (1 + augment_factor) if split_name == "train" else len(files)
        print(f"ℹ️  Split '{split_name}': {total_imgs} images")

    print(f"✅ Dataset split and augmentation complete: {output_dir}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    INPUT_IMAGES_DIR = INPUT_IMAGES_DIR
    MASK_DIR = r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\images\dataset\mask"
    TMP_OUT = Path(OUTPUT_TMP_DIR / "tmp_dataset")
    FINAL_OUT = Path("data_final")

    # Step 1: build paired dataset (ruler + no_ruler)
    images_dir, masks_dir = build_dataset(MASK_DIR, INPUT_IMAGES_DIR, TMP_OUT)

    # Step 2: split + augment
    split_dataset(images_dir, masks_dir, FINAL_OUT,
                  train_ratio=0.7, val_ratio=0.15, augment_factor=5)


