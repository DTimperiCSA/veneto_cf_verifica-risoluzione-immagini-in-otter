from pathlib import Path

from mutliprocess_and_multithread.utils import is_valid_image_file
from mutliprocess_and_multithread.paths import *
from mutliprocess_and_multithread.config import *
from mutliprocess_and_multithread.image_processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger: CSVLogger, output_sr_dir: Path, output_final_dir: Path, sr_model):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir
        self.sr_model = sr_model

    def run(self, image_path: Path):
        filename = image_path.name

        try:
            if (self.output_final_dir / filename).exists():
                return

            valid, err = is_valid_image_file(image_path)
            if not valid:
                self.logger.log(filename, "validate_input", success=False, error=f"Input image invalid: {err}")
                return
            
            if not (self.output_sr_dir / filename).exists():
                try:
                    output_sr_path = apply_super_resolution_single(image_path, self.output_sr_dir, self.sr_model)
                except Exception as e:
                    self.logger.log(filename, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    return

            valid_sr, err_sr = is_valid_image_file(output_sr_path)
            if not valid_sr:
                self.logger.log(filename, "validate_super_resolution", success=False, error=f"Super-resolved image invalid: {err_sr}")
                return
            self.logger.log(filename, "super_resolution", success=True)

            try:
                output_final_path = apply_personalized_downscaling_single(output_sr_path, self.output_final_dir)
            except Exception as e:
                self.logger.log(filename, "downscale", success=False, error=f"Errore downscale: {e}")
                return

            valid_ds, err_ds = is_valid_image_file(output_final_path)
            if not valid_ds:
                self.logger.log(filename, "validate_downscale", success=False, error=f"Downscaled image invalid: {err_ds}")
                return
            self.logger.log(filename, "downscale", success=True)

        except Exception as e:
            self.logger.log_crash(f"Unexpected error with {filename}: {e}")