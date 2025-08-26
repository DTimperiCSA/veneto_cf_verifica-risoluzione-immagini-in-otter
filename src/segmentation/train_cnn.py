import os
from pathlib import Path
from typing import Tuple
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.segmentation.unet import UNet
from src.paths import *


# -------------------------
# Replace these imports with your project's paths module
# -------------------------
from src.paths import (
    DATASET_TRAIN_IMAGES_DIR,
    DATASET_TRAIN_MASK_DIR,
    DATASET_VAL_IMAGES_DIR,
    DATASET_VAL_MASK_DIR,
    CHECKPOINT_DIR,
    SAVE_PATH
)

# =========================
# Config
# =========================
MAX_EPOCHS = 30
BATCH_SIZE = 8
LR = 1e-4                # base lr (OneCycleLR will use LR_MAX)
LR_MAX = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8             # early stopping patience (based on val combo loss)
NUM_WORKERS = 4
SEED = 42
IMG_SIZE = 480

CHECKPOINT_DIR = Path(CHECKPOINT_DIR)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = REPO_DIR / "debug_mask"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = Path(SAVE_PATH)


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

def overlay_true_vs_sharp(true_mask: np.ndarray, sharp_mask: np.ndarray) -> np.ndarray:
    """
    Overlay ground-truth mask (RED) and sharpened mask (GREEN).
    Both inputs must be uint8 (0–255), shape HxW.
    Returns a 3-channel RGB overlay image.
    """
    # Ensure masks are uint8
    true_mask = true_mask.astype(np.uint8)
    sharp_mask = sharp_mask.astype(np.uint8)

    # Create empty RGB image
    overlay = np.zeros((true_mask.shape[0], true_mask.shape[1], 3), dtype=np.uint8)

    # RED channel: ground truth
    overlay[:, :, 2] = true_mask

    # GREEN channel: sharpened mask
    overlay[:, :, 1] = sharp_mask

    # Optional faint background: average of both
    gray_bg = ((true_mask.astype(np.float32) + sharp_mask.astype(np.float32)) / 2 * 0.3).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 1.0, cv2.cvtColor(gray_bg, cv2.COLOR_GRAY2BGR), 0.3, 0)

    return overlay



def sharpen_mask(prob_map: np.ndarray, threshold: float = 0.5, min_area: int = 500) -> np.ndarray:
    """
    Post-processes a probability map to produce a sharp rectangular mask.
    - Thresholds the prob map
    - Finds largest contours
    - Fits rectangles to them
    """
    # Convert to binary mask
    mask = (prob_map > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)

    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            rect = cv2.minAreaRect(cnt)   # rotated rectangle
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(clean_mask, [box], 0, 255, -1)

    return clean_mask


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
# Loss: Combo = BCEWithLogits + Dice + Focal
# =========================
class ComboLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, smooth: float = 1e-6,
                 bce_weight: float = 1.0, focal_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE
        bce_loss = self.bce(logits, targets)

        # Focal
        bce_per_element = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_per_element)
        focal_term = self.alpha * ((1 - pt) ** self.gamma) * bce_per_element
        focal_loss = focal_term.mean()

        # Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return (
            self.bce_weight * bce_loss +
            self.focal_weight * focal_loss +
            self.dice_weight * dice_loss
        )


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
    return float(iou.mean().item())

@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    denom = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)
    return float(dice.mean().item())

@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    correct = (preds == targets).float().sum()
    total = torch.numel(preds)
    return float((correct / total).item())



# =========================
# Transforms & DataLoader
# =========================

        # rotazioni e tutto già inserite nel augmented dataset
        
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.3),
        #A.Rotate(limit=15, p=0.5),

def get_transforms(img_size=480):
    train_tf = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.6),       
        A.Affine(translate_percent=(0.1,0.1), scale=(0.85,1.15), rotate=(-15,15)),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.35),
        A.GaussianBlur(blur_limit=(3,7), p=0.2),
        A.GaussNoise(noise_scale_factor=0.5, p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    val_tf = A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),
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
# Checkpoint utils
# =========================
def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"),
                         key=lambda x: int(x.stem.split("_")[-1]) if x.stem.split("_")[-1].isdigit() else -1)
    return checkpoints[-1] if checkpoints else None

def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, scheduler, scaler, best_val_loss: float):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'best_val_loss': best_val_loss
    }
    torch.save(state, path)

# =========================
# Debug saving helpers
# =========================
def tensor_to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """Expect CHW tensor in [0,1] -> HWC uint8"""
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.numpy() * 255).astype(np.uint8)
    if arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0))
    elif arr.shape[0] == 1:
        return arr[0]
    else:
        return np.transpose(arr, (1, 2, 0))

def save_debug_batch(imgs, masks, logits, epoch, prefix="val", sharp_masks=None):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    for i in range(len(imgs)):
        img_np = tensor_to_uint8_image(imgs[i])
        mask_np = (masks[i].numpy().squeeze() * 255).astype(np.uint8)
        prob_np = (probs[i].numpy().squeeze() * 255).astype(np.uint8)
        pred_np = (preds[i].numpy().squeeze() * 255).astype(np.uint8)

        out_dir = DEBUG_DIR / f"epoch_{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(img_np).save(out_dir / f"{prefix}_sample{i}_img.png")
        Image.fromarray(mask_np).save(out_dir / f"{prefix}_sample{i}_target.png")
        Image.fromarray(prob_np).save(out_dir / f"{prefix}_sample{i}_prob.png")
        Image.fromarray(pred_np).save(out_dir / f"{prefix}_sample{i}_pred.png")

        # Compute or use existing sharp mask
        if sharp_masks is not None:
            sharp_mask = sharp_masks[i]
        else:
            prob_map = probs[i, 0]
            sharp_mask = sharpen_mask(prob_map, threshold=0.5)

        Image.fromarray(sharp_mask).save(out_dir / f"{prefix}_sample{i}_sharp.png")

        # Save overlay
        true_mask_np = (masks[i].numpy().squeeze() * 255).astype(np.uint8)
        overlay_img = overlay_true_vs_sharp(true_mask_np, sharp_mask)
        out_path = out_dir / f"{prefix}_sample{i}_overlay.png"
        cv2.imwrite(str(out_path), overlay_img)




# =========================
# Training loop
# =========================
def train_model(resume_checkpoint: Optional[Path] = None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_loaders()
    model = UNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = ComboLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    start_epoch = 1
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # Resume logic
    latest_ckpt = get_latest_checkpoint(CHECKPOINT_DIR)
    if latest_ckpt:
        resume_checkpoint = latest_ckpt
        print(f"🔄 Found latest checkpoint: {resume_checkpoint}")

    if resume_checkpoint and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') is not None:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                print("⚠️ Scheduler state load failed; continuing with fresh scheduler.")
        if ckpt.get('scaler_state_dict') is not None:
            try:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            except Exception:
                print("⚠️ GradScaler state load failed; continuing with fresh scaler.")
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        print(f"🔄 Resuming training from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        # ======= Train =======
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [train]")
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(imgs)   # logits!
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            # OneCycleLR step (per batch)
            try:
                scheduler.step()
            except Exception:
                pass

            train_loss_accum += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.3e}")

        train_loss = train_loss_accum / max(1, len(train_loader))

        # ======= Validation =======
        model.eval()
        val_loss_accum = 0.0
        val_iou_accum = 0.0
        val_dice_accum = 0.0
        val_acc_accum = 0.0

        first_val_batch = None
        with torch.no_grad():
            for i, (imgs, masks) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS} [val]")):
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                # === sharpen predictions ===
                probs = torch.sigmoid(logits).cpu().numpy()
                sharp_preds = []
                for b in range(probs.shape[0]):
                    prob_map = probs[b, 0]  # assuming (B,1,H,W)
                    sharp_mask = sharpen_mask(prob_map, threshold=0.5)
                    sharp_preds.append(sharp_mask)

                # Convert back to torch for metric calculation
                sharp_preds_t = torch.tensor(
                    np.stack(sharp_preds), dtype=torch.float32, device=device
                ).unsqueeze(1) / 255.0  # shape (B,1,H,W), values 0–1

                val_loss_accum += float(loss.item())
                val_iou_accum += iou_score(sharp_preds_t, masks)
                val_dice_accum += dice_score(sharp_preds_t, masks)
                val_acc_accum += pixel_accuracy(sharp_preds_t, masks)

                # === Save the first batch for debug later ===
                if first_val_batch is None:
                    # store CPU tensors and sharpened masks
                    first_val_batch = (imgs.cpu(), masks.cpu(), logits.cpu(), sharp_preds)

        # ======= Save debug masks for the first val batch (after loop) =======
        if first_val_batch is not None:
            imgs_b, masks_b, logits_b, sharp_b = first_val_batch
            save_debug_batch(imgs_b, masks_b, logits_b, epoch, prefix="val", sharp_masks=sharp_b)
        else:
            print("⚠️  Warning: first_val_batch is None, skipping debug save for this epoch")




        val_loss = val_loss_accum / max(1, len(val_loader))
        val_iou = val_iou_accum / max(1, len(val_loader))
        val_dice = val_dice_accum / max(1, len(val_loader))
        val_acc = val_acc_accum / max(1, len(val_loader))

        print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.6f} | val_iou: {val_iou:.4f} | "
              f"val_dice: {val_dice:.4f} | val_acc: {val_acc:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e}")

        # ======= Save debug masks for the first val batch (every epoch) =======
        if first_val_batch is not None:
            imgs_b, masks_b, logits_b, sharp_b = first_val_batch
            save_debug_batch(imgs_b, masks_b, logits_b, epoch, prefix="val", sharp_masks=sharp_b)


        # ======= Best model & Early Stopping based on validation ComboLoss =======
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"💾  Best model updated → {SAVE_PATH} (val_loss={val_loss:.6f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement in val_loss for {epochs_no_improve}/{PATIENCE} epochs.")

        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping activated (val_loss did not improve).")
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, scaler, best_val_loss)
            print(f"💾  Final checkpoint saved → {checkpoint_path}")
            break

        # ======= Regular checkpointing =======
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, scaler, best_val_loss)
        print(f"💾  Checkpoint saved → {checkpoint_path}")

    print("✅ Training completed.")
    print(f"☑️  Best val_loss: {best_val_loss:.6f}")
    if SAVE_PATH.exists():
        print(f"📦 Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train_model()