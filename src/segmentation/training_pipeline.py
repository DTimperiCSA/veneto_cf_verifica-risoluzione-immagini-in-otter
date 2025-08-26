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
from src.segmentation.split_dataset import *

SEED = 42

# =========================
# Main workflow
# =========================
if __name__ == "__main__":
    # 1️⃣ Gather datasets
    with_masks, without_masks = collect_datasets(SRC_IMAGES, SRC_MASKS, SRC_NO_MASK)

    # 2️⃣ Check if splits already exist
    if (SPLIT_DIR / "train_images").exists() and (SPLIT_DIR / "val_images").exists() and (SPLIT_DIR / "test_images").exists():
        print("✅ Split dataset already exists, skipping split/augmentation.")
    else:
        print("🚀 Creating new stratified split...")
        splits = stratified_split(with_masks, without_masks)

        # 3️⃣ Save splits with augmentation
        save_splits(splits, SPLIT_DIR)

    # 4️⃣ Start training process
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    train_model()
