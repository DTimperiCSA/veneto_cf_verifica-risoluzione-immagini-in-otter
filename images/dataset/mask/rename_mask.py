from pathlib import Path

# Folder containing the files
folder = Path(r"C:\Users\andre\Desktop\veneto_cf_verifica-risoluzione-immagini-in-otter\images\dataset\mask")

# Prefix to remove
prefix = "kp_mask_x_transfer_"

for file in folder.iterdir():
    if file.is_file() and file.name.startswith(prefix):
        new_name = file.name[len(prefix):]
        new_path = file.with_name(new_name)
        file.rename(new_path)
        print(f"Renamed: {file.name} -> {new_name}")
