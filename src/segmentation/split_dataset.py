from pathlib import Path
import shutil
import random
from PIL import Image
import numpy as np
import albumentations as A

from src.paths import *

# =========================
# Config
# =========================
TRAIN_RATIO = 0.4
VAL_RATIO = 0.5
TEST_RATIO = 1 - VAL_RATIO - TRAIN_RATIO
SEED = 42
AUGMENT_FACTOR = 5
IMG_SIZE = 512            # output dataset

# =========================
# Augmentations
# =========================
def get_augmentations(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size, ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ])

def augment_and_save(img_path, mask_path, out_img_dir, out_mask_dir, aug_index):
    aug = get_augmentations()
    img = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))
    augmented = aug(image=img, mask=mask)
    aug_img = Image.fromarray(augmented['image'])
    aug_mask = Image.fromarray(augmented['mask'])
    img_name = img_path.stem + f"_aug{aug_index}" + img_path.suffix
    mask_name = mask_path.stem + f"_aug{aug_index}" + mask_path.suffix
    aug_img.save(out_img_dir / img_name)
    aug_mask.save(out_mask_dir / mask_name)

# =========================
# Step 1: Collect datasets
# =========================
def collect_datasets(src_img_dir, src_mask_dir, src_no_mask_dir):
    with_masks = []
    without_masks = []

    # images with masks
    for img_path in src_img_dir.rglob("*.*"):
        mask_path = src_mask_dir / img_path.name # adjust if masks have another extension
        if mask_path.exists():
            with_masks.append((img_path, mask_path))
        else:
            print(f"⚠️ Mask not found for {img_path.name}, skipping")

    # images without masks
    """
    for img_path in src_no_mask_dir.rglob("*.*"):
        without_masks.append((img_path, None))
    """

    print(f"Found {len(with_masks)} images WITH masks, {len(without_masks)} WITHOUT masks")
    return with_masks, without_masks

# =========================
# Step 2: Stratified split
# =========================
def stratified_split(with_masks, without_masks,
                     train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    random.seed(SEED)

    def split_list(lst):
        lst = lst.copy()
        random.shuffle(lst)
        total = len(lst)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        return {
            "train": lst[:train_count],
            "val": lst[train_count:train_count + val_count],
            "test": lst[train_count + val_count:]
        }

    splits_with = split_list(with_masks)
    splits_without = split_list(without_masks)

    # Combine both categories
    splits = {}
    for k in ["train", "val", "test"]:
        splits[k] = splits_with[k] + splits_without[k]
        random.shuffle(splits[k])

        # Debug info
        num_with = sum(1 for _, m in splits[k] if m is not None)
        num_without = sum(1 for _, m in splits[k] if m is None)
        print(f"Split {k}: {len(splits[k])} (with mask={num_with}, without mask={num_without})")

    return splits

# =========================
# Step 3: Save splits and augment train
# =========================
def save_splits(splits, DATASET_DIR=DATASET_DIR, augment_factor=AUGMENT_FACTOR):
    for split_name, pairs in splits.items():
        if split_name == "train": 
            print(f"Processing {split_name}... ({len(pairs)*AUGMENT_FACTOR} samples)")
        else:
            print(f"Processing {split_name}... ({len(pairs)} samples )")
        img_out = DATASET_DIR / f"{split_name}_images"
        mask_out = DATASET_DIR / f"{split_name}_masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            dst_img = img_out / img_path.name
            shutil.copy(img_path, dst_img)

            if mask_path:  # real mask
                dst_mask = mask_out / mask_path.name
                shutil.copy(mask_path, dst_mask)
            else:  # fake black mask
                with Image.open(img_path) as im:
                    black_mask = Image.new("L", im.size, 0)
                    dst_mask = mask_out / img_path.name
                    black_mask.save(dst_mask)

            # Augment only train
            if split_name == "train":
                for i in range(augment_factor):
                    augment_and_save(dst_img, dst_mask, img_out, mask_out, i)

        print(f"ℹ️ Split '{split_name}' saved with {len(pairs)} images")

    print(f"✅ All splits saved in {DATASET_DIR}")

# =========================
# Run pipeline
# ============
