import cv2
import numpy as np
from pathlib import Path
import shutil

from src.paths import INPUT_IMAGES_DIR, TEMPLATE_IMG_PATH


import cv2
from pathlib import Path
import shutil

def find_chromatic_band_in_folder(
    folder: Path,
    template_path: Path,
    output_dir: Path = Path("tmp/chromatic_bands_found")
) -> str | None:
    folder = Path(folder)
    template_path = Path(template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder {folder} does not exist or is not a directory.")
    if not template_path.exists():
        raise ValueError(f"Template {template_path} does not exist.")

    # Load template (grayscale)
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise ValueError(f"Could not load template image: {template_path}")
    t_h, t_w = template.shape[:2]

    print(f"📁 Exploring folder: {folder}")

    best_match = None
    best_val = -1
    total_images = 0
    skipped_small = 0

    for img_path in folder.iterdir():
        if img_path.is_file():
            if img_path.name.lower() == "thumbs.db":
                continue
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]:
                continue

            total_images += 1
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️  Could not read image: {img_path.name}")
                continue

            i_h, i_w = img.shape[:2]
            if i_h < t_h or i_w < t_w:
                skipped_small += 1
                print(f"  ⬇ Skipping {img_path.name}: smaller than template ({i_w}x{i_h})")
                continue

            # Template matching
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            print(f"  🔹 Checked {img_path.name} - match value: {max_val:.3f}")

            if max_val > best_val:
                best_val = max_val
                best_match = img_path
                print(f"    ✅ New best match: {best_match.name} (val: {best_val:.3f})")

    print(f"📊 Folder summary: {total_images} images checked, {skipped_small} skipped")
    if best_match is not None and best_val > 0.5:
        new_name = f"{folder.name}_{best_match.name}"
        shutil.copy(best_match, output_dir / new_name)
        print(f"🎯 Chromatic band found and copied: {best_match.name} -> {new_name}\n")
        return str(best_match)
    else:
        print("❌ No chromatic band found above threshold\n")
        return None



if __name__ == "__main__":
    root_dir = INPUT_IMAGES_DIR 
    template = TEMPLATE_IMG_PATH
    res = find_chromatic_band_in_folder(root_dir, template)
    print("result", res)

    res = find_chromatic_band_in_folder(root_dir / "B001.001", template)
    print("result", res)
