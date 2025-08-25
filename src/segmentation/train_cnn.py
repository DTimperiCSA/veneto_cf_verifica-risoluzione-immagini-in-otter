import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.segmentation.unet import UNet
from src.paths import *


# =========================
# Config
# =========================
MAX_EPOCHS = 30
BATCH_SIZE = 4
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 8  # early stopping patience
NUM_WORKERS = 2
SEED = 42

# Checkpoint
CHECKPOINT_INTERVAL = 1  # salva ogni N epoche
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Utils: seed & reproducibility
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.images = sorted([p for p in self.img_dir.glob("*") if p.is_file()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

        if isinstance(mask, torch.Tensor):
            mask = mask.float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                mask = mask[:1]
            mask = (mask > 0.5).float()
        else:
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
            mask = (mask > 0.5).float()

        return image, mask


# =========================
# Loss: BCEWithLogits + Dice
# =========================
class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.squeeze(1)
        targets = targets.squeeze(1)

        intersection = (probs * targets).sum(dim=(1, 2))
        denom = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice.mean()
        return bce_loss + dice_loss


# =========================
# Metrics
# =========================
@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets).sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)
    return dice.mean().item()


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return (correct / total).item()


# =========================
# Transforms & DataLoader
# =========================
def get_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"),
                         key=lambda x: int(x.stem.split("_")[-1]))
    return checkpoints[-1] if checkpoints else None


def get_transforms(img_size=480):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(translate_percent=(0.1, 0.1), scale=(0.9, 1.1), rotate=(-15, 15)),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_tf, val_tf


def create_loaders() -> Tuple[DataLoader, DataLoader]:
    train_tf, val_tf = get_transforms()
    train_ds = SegmentationDataset(DATASET_TRAIN_IMAGES_DIR, DATASET_TRAIN_MASK_DIR, transform=train_tf)
    val_ds = SegmentationDataset(DATASET_VAL_IMAGES_DIR, DATASET_VAL_MASK_DIR, transform=val_tf)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin_memory, persistent_workers=NUM_WORKERS > 0
    )
    return train_loader, val_loader


# =========================
# Training loop with checkpoint
# =========================
def train_model(resume_checkpoint: Path = None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders()
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = DiceBCELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_dice = 0.0   # <--- aggiunto
    epochs_no_improve = 0


    # Automatically resume from latest checkpoint if available
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        resume_checkpoint = latest_ckpt
        print(f"🔄 Found latest checkpoint: {resume_checkpoint}")
    else:
        resume_checkpoint = None

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if resume_checkpoint and resume_checkpoint.exists():
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"🔄 Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # ======= Train =======
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss_accum += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = train_loss_accum / max(1, len(train_loader))

        # ======= Validation =======
        model.eval()
        val_loss_accum = 0.0
        val_iou_accum = 0.0
        val_dice_accum = 0.0
        val_acc_accum = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [val]"):
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                val_loss_accum += loss.item()
                val_iou_accum += iou_score(logits, masks)
                val_dice_accum += dice_score(logits, masks)
                val_acc_accum += pixel_accuracy(logits, masks)

        val_loss = val_loss_accum / max(1, len(val_loader))
        val_iou = val_iou_accum / max(1, len(val_loader))
        val_dice = val_dice_accum / max(1, len(val_loader))
        val_acc = val_acc_accum / max(1, len(val_loader))
        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_iou: {val_iou:.4f} | "
              f"val_dice: {val_dice:.4f} | val_acc: {val_acc:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # ======= Early Stopping & Best Model =======
        # Best model salvato in base al val_dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾  Best model aggiornato → {SAVE_PATH} (val_dice={val_dice:.4f})")

        # Early stopping invece basato su val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("⏹ Early stopping attivato.")
                break


        # ======= Checkpoint =======
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"💾  Checkpoint salvato → {checkpoint_path}")

    print("✅ Training completato.")
    print(f"☑️  Best val_loss: {best_val_loss:.4f}")
    if SAVE_PATH.exists():
        print(f"📦 Modello salvato: {SAVE_PATH}")
