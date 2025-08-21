from pathlib import Path
import shutil

def split_images_into_folders(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"La cartella di input {input_dir} non esiste o non è una cartella.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in input_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"]:
            folder_name = img_path.stem  # nome senza estensione
            target_folder = output_dir / folder_name
            target_folder.mkdir(parents=True, exist_ok=True)

            shutil.copy(img_path, target_folder / img_path.name)  # copia l’immagine

    print(f"✅ Finito! Immagini salvate in {output_dir}")

# Esempio d’uso
if __name__ == "__main__":
    split_images_into_folders(r"C:\Users\andre\Desktop\x_transfer", r"C:\Users\andre\Desktop\test_all_mask")
