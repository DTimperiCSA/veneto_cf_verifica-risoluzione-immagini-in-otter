from pathlib import Path
import shutil
import random
import os
from typing import Tuple, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.segmentation.unet import UNet
from src.segmentation.train_cnn import *
from src.paths import *

# =========================
# Config
# =========================
SRC_IMAGES = Path(r"C:\Users\andre\Desktop\data\images")               # with masks
SRC_MASKS = Path(r"C:\Users\andre\Desktop\data\mask")
SRC_NO_MASK = Path(r"C:\Users\andre\Desktop\data\images_no_mask")  # no masks
SPLIT_DIR = Path(r"C:\Users\andre\Desktop\dataset")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
SEED = 42
AUGMENT_FACTOR = 15
IMG_SIZE = 512


# =========================
# Augmentations
# =========================
def get_augmentations(img_size=IMG_SIZE):
    return A.Compose([
        A.Resize(img_size, img_size),
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

    for img_path in src_img_dir.glob("*"):
        mask_path = src_mask_dir / img_path.name
        if mask_path.exists():
            with_masks.append((img_path, mask_path))
        else:
            print(f"⚠️ Missing mask for {img_path}, skipped")

    for img_path in src_no_mask_dir.glob("*"):
        without_masks.append((img_path, None))

    return with_masks, without_masks


# =========================
# Step 2: Stratified split
# =========================
# =========================
# Step 2: Stratified split (bilanciato con e senza maschere)
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

    # Splitta separatamente
    splits_with = split_list(with_masks)
    splits_without = split_list(without_masks)

    # Combina garantendo la presenza di entrambe le categorie
    splits = {}
    for k in ["train", "val", "test"]:
        splits[k] = splits_with[k] + splits_without[k]
        random.shuffle(splits[k])

        # 🔍 Debug info
        num_with = sum(1 for _, m in splits[k] if m is not None)
        num_without = sum(1 for _, m in splits[k] if m is None)
        print(f"Split {k}: {len(splits[k])} (with mask={num_with}, without mask={num_without})")

    return splits


# =========================
# Step 3: Save splits
# =========================
def save_splits(splits, output_dir, augment_factor=AUGMENT_FACTOR):
    for split_name, pairs in splits.items():
        print(f"Processing {split_name}... ({len(pairs)} samples)")
        img_out = output_dir / f"{split_name}_images"
        mask_out = output_dir / f"{split_name}_masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            dst_img = img_out / img_path.name
            shutil.copy(img_path, dst_img)

            if mask_path:  # real mask
                dst_mask = mask_out / img_path.name
                shutil.copy(mask_path, dst_mask)
            else:  # fake black mask
                with Image.open(img_path) as im:
                    black_mask = Image.new("L", im.size, 0)
                    dst_mask = mask_out / img_path.name
                    black_mask.save(dst_mask)

            if split_name == "train":
                for i in range(augment_factor):
                    augment_and_save(dst_img, dst_mask, img_out, mask_out, i)

        print(f"ℹ️ Split '{split_name}': {len(pairs)} (before augmentation)")

    print(f"✅ Stratified split saved to {output_dir}")


# =========================
# Main workflow
# =========================
if __name__ == "__main__":
    # 1️⃣ Gather datasets
    with_masks, without_masks = collect_datasets(SRC_IMAGES, SRC_MASKS, SRC_NO_MASK)

    # 2️⃣ Stratified split
    splits = stratified_split(with_masks, without_masks)

    # 3️⃣ Save splits with augmentation
    save_splits(splits, SPLIT_DIR)

    # 4️⃣ Start training process
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    train_model()
