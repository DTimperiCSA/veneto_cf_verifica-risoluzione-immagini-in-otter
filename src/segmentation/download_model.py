import requests
from pathlib import Path

from src.paths import SAVE_PATH

def download_file_from_google_drive(file_id: str, destination: Path):
    """
    Download a file from Google Drive (public/shared with link).
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Handle confirmation token for large files
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    # Save to file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

    print(f"✅ Download complete: {destination}")


if __name__ == "__main__":
    # Example: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
    FILE_ID = "1ED3Uk962YiYdprTc538VFBHlKP8H4Ihi"

    download_file_from_google_drive(FILE_ID, SAVE_PATH)

