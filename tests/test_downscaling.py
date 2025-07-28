import torch
import os
import numpy as np
from PIL import Image
from pathlib import Path

from utils import *
from model.SR_Script.super_resolution import SA_SuperResolution


# Converte un array NumPy (H x W x C) RGB in un oggetto PIL.Image
def numpy_to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)


# Applica la super-risoluzione 2x su una lista di immagini e salva i risultati nella cartella di output
def create_super_resolution_images(image_paths: list[str]):
    print("▶ Starting super resolution algorithm:")

    for img_filename in image_paths:
        print(f"- Processing: {img_filename}")
        img_path = IMAGES_INPUT_DIR / img_filename

        img_np = np.array(Image.open(img_path).convert("RGB"))
        upscaled_image_np = model.run(img_np)
        output_img = numpy_to_image(upscaled_image_np)

        output_path = IMAGES_SR_X2_DIR / f"sr_{img_filename}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        output_img.save(output_path)
        print(f"  ✅ Saved super-resolved image to {output_path}")


# Ridimensiona affinché il righello cromatico misuri esattamente 190 mm
def create_personalized_downscaling(image_paths: list[str]):
    print("▶ Starting personalized downscaling:")

    for img_filename in image_paths:
        print(f"- Processing: {img_filename}")
        img_path = IMAGES_SR_X2_DIR / img_filename

        requested_PPI = int(img_filename.split("_")[-1].split(".")[0])
        if requested_PPI == 400:
            ruler = CROMATIC_SCALE_RULER_400_PPI
        elif requested_PPI == 600:
            ruler = CROMATIC_SCALE_RULER_600_PPI
        else:
            print(f"  ⚠️ Skip {img_filename} (Unsupported PPI)")
            continue

        # Righello dopo super-resolution
        ruler_width_px_sr = ruler["width_pixel"] * SUPER_RESOLUTION_PAR

        # Calcola fattore per portare il righello da x pixel a 190 mm
        scale_factor = CHROMATIC_SCALE_RULER_MM / ruler["width_mm"] / SUPER_RESOLUTION_PAR

        sr_image = Image.open(img_path)
        new_width = int(sr_image.width * scale_factor)
        new_height = int(sr_image.height * scale_factor)

        print(f"  ℹ️  Original ruler: {ruler['width_pixel']} px → SR: {ruler_width_px_sr} px")
        print(f"  🎯 Target physical size: 190 mm")
        print(f"  📏 Scaling factor: {scale_factor:.6f}")
        print(f"  🖼️  New image size: {new_width} x {new_height} px")

        resized_img = sr_image.resize((new_width, new_height), resample=Image.LANCZOS)
        output_path = IMAGES_DOWNSIZED_DIR / f"ds_{img_filename}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized_img.save(output_path)
        print(f"  ✅ Saved downsized image to {output_path}")


# Main: inizializza modello, esegue super-risoluzione e downscaling personalizzato
if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Please check your setup."

    model = SA_SuperResolution(
        models_dir=SR_SCRIPT_MODEL_DIR,
        model_scale=SUPER_RESOLUTION_PAR,
        tile_size=128,
        gpu_id=0,
        verbosity=True,
    )

    # Assicura che le cartelle esistano
    IMAGES_SR_X2_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DOWNSIZED_DIR.mkdir(parents=True, exist_ok=True)

    # Super-resolution
    image_paths = os.listdir(IMAGES_INPUT_DIR)
    create_super_resolution_images(image_paths)

    # Downscaling
    super_res_images = os.listdir(IMAGES_SR_X2_DIR)
    create_personalized_downscaling(super_res_images)
