# src/segmentation/train_cnn.py

import os
from pathlib import Path
from typing import Tuple, Dict

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

TRAIN_IMG_DIR = r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\data\train_images"
TRAIN_MASK_DIR = r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\data\train_masks"
VAL_IMG_DIR = r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\data\val_images"
VAL_MASK_DIR = r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\data\val_masks"

SAVE_PATH = Path(
    r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\model\tiffen_segmenter_best.pth"
)


# =========================
# Utils: seed & reproducibility
# =========================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class SegmentationDataset(Dataset):
    """
    Restituisce:
      - image: FloatTensor [3, H, W] (0..1)
      - mask:  FloatTensor [1, H, W] binaria {0,1}
    """
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

        # albumentations lavora con dict
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]   # torch.FloatTensor [3,H,W], già 0..1
            mask = augmented["mask"]     # torch.FloatTensor [H,W] o [1,H,W] a seconda di ToTensorV2

        # Assicura dtype e shape
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        else:
            image = image.float()
            if image.max() > 1.0:
                image = image / 255.0

        if isinstance(mask, torch.Tensor):
            mask = mask.float()
            # ToTensorV2 di solito ritorna [H,W] per la mask: aggiungi canale
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                # se per qualche motivo è [C,H,W] con C!=1, prendi un canale
                mask = mask[:1]
            # binarizza
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
        # logits: [B,1,H,W], targets: [B,1,H,W]
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        # squeeze canale per calcolare dice per immagine
        probs = probs.squeeze(1)      # [B,H,W]
        targets = targets.squeeze(1)  # [B,H,W]

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
    # ensure shapes match
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


# =========================
# Train / Val loops
# =========================
def get_transforms(img_size=256):
    train_tf = A.Compose([
        A.Resize(img_size, img_size),  # forza shape uniforme
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_tf = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_tf, val_tf



def create_loaders() -> Tuple[DataLoader, DataLoader]:
    train_tf, val_tf = get_transforms()
    train_ds = SegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transform=train_tf)
    val_ds = SegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_tf)

    # Pin memory utile su CUDA, no-harm su CPU
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0
    )
    return train_loader, val_loader


def train_model():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders()

    model = UNet(n_channels=3, n_classes=1).to(device)

    # Optimizer + Scheduler + Loss
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = DiceBCELoss()

    # Mixed precision solo se CUDA
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ======= Train =======
        model.train()
        train_loss_accum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)  # [B,1,H,W]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)                # [B,1,H,W]
                loss = criterion(logits, masks)    # BCELogits+Dice richiede logits

            scaler.scale(loss).backward()
            # opzionale: gradient clipping per stabilità
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

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                val_loss_accum += loss.item()
                val_iou_accum += iou_score(logits, masks)
                val_dice_accum += dice_score(logits, masks)

        val_loss = val_loss_accum / max(1, len(val_loader))
        val_iou = val_iou_accum / max(1, len(val_loader))
        val_dice = val_dice_accum / max(1, len(val_loader))

        # Step del scheduler su metrica di validazione (loss)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_iou: {val_iou:.4f} | "
            f"val_dice: {val_dice:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # ======= Early Stopping & Checkpoint =======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾  Best model aggiornato → {SAVE_PATH} (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("⏹ Early stopping attivato.")
                break

    print("✅ Training completato.")
    print(f"☑️  Best val_loss: {best_val_loss:.4f}")
    if SAVE_PATH.exists():
        print(f"📦 Modello salvato: {SAVE_PATH}")


if __name__ == "__main__":
    # Evita crash casuali tra worker dataloader
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    train_model()
