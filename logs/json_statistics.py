import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_jsons(directory: Path):
    """Load all JSON files from a directory into a list of dicts."""
    data = []
    for file in directory.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"❌ Could not read {file}: {e}")
    return data


def summarize_data(data):
    """Convert list of dicts to pandas DataFrame and print summary stats."""
    df = pd.DataFrame(data)

    print("📊 General Statistics")
    print(f"Total JSON files: {len(df)}")
    print(df.describe(include="all"))

    # Counts for A4
    if "is_A4" in df.columns:
        print("\nCounts for A4:")
        print(df["is_A4"].value_counts())

    # Histogram of scale_factor
    if "scale_factor" in df.columns:
        plt.hist(df["scale_factor"].dropna(), bins=20, color="skyblue", edgecolor="black")
        plt.xlabel("Scale Factor")
        plt.ylabel("Count")
        plt.title("Distribution of Scale Factor")
        plt.show()

    # Scatter plot of image dimensions
    if "img_px" in df.columns:
        try:
            xs = [x[0] for x in df["img_px"] if isinstance(x, (list, tuple))]
            ys = [x[1] for x in df["img_px"] if isinstance(x, (list, tuple))]
            plt.scatter(xs, ys, alpha=0.5)
            plt.xlabel("Width (px)")
            plt.ylabel("Height (px)")
            plt.title("Image Dimensions in Pixels")
            plt.show()
        except Exception as e:
            print(f"⚠️ Could not plot image dimensions: {e}")

    # Histogram of PPI
    if "ppi" in df.columns:
        plt.hist(df["ppi"].dropna(), bins=20, color="lightgreen", edgecolor="black")
        plt.xlabel("PPI")
        plt.ylabel("Count")
        plt.title("Distribution of PPI")
        plt.show()

    return df

def detect_a4(df, tolerance=0.03):
    """
    Update DataFrame with 'is_A4' column based on image dimensions and ppi.

    Parameters:
        df: pandas DataFrame with 'img_px' and 'ppi' columns
        tolerance: relative tolerance (default 3%)
    """
    import numpy as np

    a4_mm = np.array([210, 297])  # Width x Height in mm
    df = df.copy()


if __name__ == "__main__":
    folder = Path(r"C:\Users\cultura\Desktop\github\Davide Timperi - Conservatorio Venezia\unet_segmentation\json")  # 👈 change this
    data = load_jsons(folder)

    if not data:
        print("No JSON files found!")
    else:
        df = pd.DataFrame(data)
        summarize_data(df)
