from pathlib import Path
from PIL import Image

VALID_EXTENSIONS = [".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"]

def resize_half(input_path: Path, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)  # crea la cartella se non esiste
    img = Image.open(input_path)
    new_w, new_h = img.width // 2, img.height // 2
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    if input_path.suffix.lower() in [".tif", ".tiff"]:
        resized.save(output_file, format="TIFF", compression="tiff_lzw")
    else:
        resized.save(output_file)
    print(f"✅ {input_path.name} -> {output_file} ({new_w}x{new_h})")


def batch_resize_folder(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in input_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in VALID_EXTENSIONS:
            output_file = output_dir / img_path.parent.name / img_path.name  # corretto: file completo
            resize_half(img_path, output_file)


if __name__ == "__main__":
    input_folder = Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio\B001\B001.004")
    output_folder = Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\resize_lossless")
    batch_resize_folder(input_folder, output_folder)
