import os

from pathlib import Path

from src.processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger: CSVLogger, output_sr_dir: Path, output_final_dir: Path):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir

    def run(self, image_path: Path):
        filename = image_path.name

        # Skip if already completed
        if self.logger.is_success(filename, "completed"):
            return

        try:
            # Step 1: Super-resolution
            try:
                output_sr_path = apply_super_resolution_single(image_path, self.output_sr_dir)
                self.logger.log(filename, "super_resolution", success=True)
            except Exception as e:
                self.logger.log(filename, "super_resolution", success=False, error=str(e))
                return

            # Step 2: Downscaling
            try:
                output_final_path = apply_personalized_downscaling_single(output_sr_path, self.output_final_dir)
                self.logger.log(filename, "downscale", success=True)
            except Exception as e:
                self.logger.log(filename, "downscale", success=False, error=str(e))
                return

        except Exception as e:
            self.logger.log_crash(f"Unexpected error with {filename}: {e}")
