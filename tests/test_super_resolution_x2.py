import sys
from pathlib import Path

# Aggiungi la root del progetto al path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils import *  # Ora funzionerà
from PIL import Image

print(IMAGES_DIR)
print(IMAGES_INPUT_DIR)
print(IMAGES_OUTPUT_DIR)

def test_super_resolution_output(scale=SCALE):
    input_images = sorted([p for p in IMAGES_INPUT_DIR.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])
    output_images = sorted([p for p in IMAGES_OUTPUT_DIR.glob("sr_*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]])

    assert len(input_images) == len(output_images), f"Numero immagini input ({len(input_images)}) diverso da output ({len(output_images)})"

    all_passed = True

    for in_path, out_path in zip(input_images, output_images):
        img_in = Image.open(in_path)
        img_out = Image.open(out_path)

        in_w, in_h = img_in.size
        out_w, out_h = img_out.size

        expected_w = in_w * scale
        expected_h = in_h * scale

        if out_w != expected_w or out_h != expected_h:
            print(f"❌ ERRORE: {in_path.name} → {out_path.name}")
            print(f"   Atteso: {expected_w}x{expected_h}, Trovato: {out_w}x{out_h}")
            all_passed = False
        else:
            print(f"✅ OK: {in_path.name} → {out_path.name} | {in_w}x{in_h} → {out_w}x{out_h}")

    if all_passed:
        print("\n✅ Tutti i test superati: risoluzione aumentata correttamente.")
    else:
        print("\n❌ Alcune immagini non hanno la risoluzione attesa.")

if __name__ == "__main__":
    test_super_resolution_output()
