from pathlib import Path

from src.utils import *
from src.paths import *
from src.config import *
from src.image_processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger, output_sr_dir, output_final_dir, sr_model):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir
        self.sr_model = sr_model

    def run(self, image_path: Path):
        filename = image_path.name

        try:
            # Skip if already completed
            if (self.output_final_dir / filename).exists():
                return

            valid, err = is_valid_image_file(image_path)
            if not valid:
                self.logger.log(filename, "validate_input", success=False, error=f"Input image invalid: {err}")
                return
            
            # Step 1: Super-resolution, passiamo il modello già caricato
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

            # Step 2: Downscaling
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
