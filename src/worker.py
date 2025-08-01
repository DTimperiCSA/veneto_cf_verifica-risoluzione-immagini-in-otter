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
        try:
            filename = image_path.name
            top_folder = image_path.parent.name

            sr_output_dir = self.output_sr_dir / top_folder
            downscale_output_dir = self.output_final_dir / top_folder

            sr_output_path = sr_output_dir / filename
            final_output_path = downscale_output_dir / filename

            if final_output_path.exists():
                return  # Già elaborata

            # 1. Validazione immagine input
            valid, err = is_valid_image_file(image_path)
            if not valid:
                self.logger.log(image_path, "validate_input", success=False, error=f"Input image invalid: {err}")
                return

            # 2. Super-Risoluzione (se non già fatta)
            if not sr_output_path.exists():
                try:
                    sr_output_path = apply_super_resolution_single(image_path, sr_output_dir, self.sr_model)
                except Exception as e:
                    self.logger.log(image_path, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    return

            # 3. Validazione SR
            valid_sr, err_sr = is_valid_image_file(sr_output_path)
            if not valid_sr:
                self.logger.log(image_path, "validate_super_resolution", success=False, error=f"Super-resolved image invalid: {err_sr}")
                return

            # 4. Downscaling personalizzato
            try:
                final_output_path = apply_personalized_downscaling_single(sr_output_path, downscale_output_dir)
            except Exception as e:
                self.logger.log(image_path, "downscale", success=False, error=f"Errore downscale: {e}")
                return

            # 5. Validazione downscale
            valid_ds, err_ds = is_valid_image_file(final_output_path)
            if not valid_ds:
                self.logger.log(image_path, "validate_downscale", success=False, error=f"Downscaled image invalid: {err_ds}")
                return

        except Exception as e:
            self.logger.log_crash(error=f"Unexpected error with {image_path}: {e}", context_path=image_path)
