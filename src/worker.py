import threading
from pathlib import Path

from src.utils import *
from src.paths import *
from src.config import *
from src.image_processing import *
from logs.logger import CSVLogger

class ImageWorker(threading.Thread):
    def __init__(self, image_path: Path, sr_output_dir: Path, ds_output_dir: Path, logger: CSVLogger):
        super().__init__()
        self.image_path = image_path
        self.sr_output_dir = sr_output_dir
        self.ds_output_dir = ds_output_dir
        self.logger = logger

    def run(self):
        try:
            final_output_path = self.ds_output_dir / self.image_path.name

            # Skip se output finale esiste
            if final_output_path.exists():
                print(f"Skipping {self.image_path.name}, output already exists.")
                return

            # Step 1: Super resolution
            try:
                sr_image_path = apply_super_resolution_single(self.image_path, self.sr_output_dir)
            except Exception as e:
                self.logger.log_failure(self.image_path.name, "super_resolution", str(e))
                return

            # Step 2: Downscaling
            try:
                ds_image_path = apply_personalized_downscaling_single(sr_image_path, self.ds_output_dir)
            except Exception as e:
                self.logger.log_failure(self.image_path.name, "downscaling", str(e))
                return

        except Exception as e:
            self.logger.log_crash(f"Crash processing {self.image_path.name}: {e}")
