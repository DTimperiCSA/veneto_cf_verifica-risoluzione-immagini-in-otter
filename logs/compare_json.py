import json
from pathlib import Path
from typing import Any

# 🔧 Imposta qui i percorsi delle due cartelle
FOLDER1 = Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\unet_segmentation\json")
FOLDER2 = Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\unet_segmentation_approximated\json")

def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def count_ppi_changes(folder1: Path, folder2: Path) -> None:
    json_files1 = {f.name: f for f in folder1.glob("*.json")}
    json_files2 = {f.name: f for f in folder2.glob("*.json")}

    common_files = set(json_files1.keys()).intersection(json_files2.keys())

    if not common_files:
        print("⚠️ Nessun file JSON con lo stesso nome trovato nelle due cartelle.")
        return

    total_diff = 0
    count_400_to_600 = 0
    count_600_to_400 = 0

    for fname in sorted(common_files):
        json1 = load_json(json_files1[fname])
        json2 = load_json(json_files2[fname])

        ppi1 = json1.get("ppi")
        ppi2 = json2.get("ppi")

        if ppi1 != ppi2:
            total_diff += 1
            if ppi1 == 400 and ppi2 == 600:
                count_400_to_600 += 1
            elif ppi1 == 600 and ppi2 == 400:
                count_600_to_400 += 1

    print(f"📊 Totale file con differenze di ppi: {total_diff} su {len(common_files)} file confrontati")
    print(f"   🔹 Da 400 → 600: {count_400_to_600}")
    print(f"   🔹 Da 600 → 400: {count_600_to_400}")

if __name__ == "__main__":
    count_ppi_changes(FOLDER1, FOLDER2)
