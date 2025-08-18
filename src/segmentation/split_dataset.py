from pathlib import Path
import shutil
import random

def split_dataset(images_dir, masks_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.seed(seed)

    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    output_dir = Path(output_dir)

    all_images = [p for p in images_dir.glob("*") if p.is_file()]
    random.shuffle(all_images)

    train_count = int(len(all_images) * train_ratio)
    val_count = int(len(all_images) * val_ratio)

    splits = {
        "train": all_images[:train_count],
        "val": all_images[train_count:train_count + val_count],
        "test": all_images[train_count + val_count:]
    }

    for split_name, files in splits.items():
        img_out = output_dir / f"{split_name}_images"
        mask_out = output_dir / f"{split_name}_masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            mask_path = masks_dir / img_path.name
            if not mask_path.exists():
                print(f"[WARNING] Mask not found for {img_path.name}, skipping.")
                continue
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(mask_path, mask_out / img_path.name)

    print(f"✅ Dataset split complete: {output_dir}")

if __name__ == "__main__":
    split_dataset(
        images_dir=r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\images\dataset\images",
        masks_dir=r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\images\dataset\mask",
        output_dir="data",
        train_ratio=0.7,
        val_ratio=0.15
    )
