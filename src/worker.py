from pathlib import Path
from src.utils import is_valid_image_file
from src.paths import *
from src.config import *
from src.image_processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger: CSVLogger, output_sr_dir: Path, output_final_dir: Path, sr_model):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir
        self.sr_model = sr_model

    def run(self, image_path: Path):
        filename = image_path.name
        top_folder = image_path.parent.name  # 'A' or 'B'

        try:
            sr_output_subdir = self.output_sr_dir / top_folder
            final_output_subdir = self.output_final_dir / top_folder

            sr_output_path = sr_output_subdir / filename
            final_output_path = final_output_subdir / filename

            if final_output_path.exists():
                return

            valid, err = is_valid_image_file(image_path)
            if not valid:
                self.logger.log(filename, "validate_input", success=False, error=f"Input image invalid: {err}")
                return
            
            if not sr_output_path.exists():
                try:
                    output_sr_path = apply_super_resolution_single(image_path, sr_output_subdir, self.sr_model)
                except Exception as e:
                    self.logger.log(filename, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    return
            else:
                output_sr_path = sr_output_path

            valid_sr, err_sr = is_valid_image_file(output_sr_path)
            if not valid_sr:
                self.logger.log(filename, "validate_super_resolution", success=False, error=f"Super-resolved image invalid: {err_sr}")
                return
            self.logger.log(filename, "super_resolution", success=True)

            try:
                output_final_path = apply_personalized_downscaling_single(output_sr_path, final_output_subdir)
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