from pathlib import Path
from src.utils import is_valid_image_file, validate_image_with_logging
from src.paths import *
from src.config import *
from src.estimate_ppi_from_ruler import *
from src.image_processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger: CSVLogger, output_sr_dir: Path, output_final_dir: Path, sr_model, ppi: int):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir
        self.sr_model = sr_model
        self.ppi = ppi

    def run(self, image_path: Path):
        try:
            filename = image_path.name
            top_folder = image_path.parent.name
            complete_path = image_path.parent

            # Directory output
            sr_output_dir = self.output_sr_dir / top_folder
            downscale_output_dir = self.output_final_dir / top_folder
            sr_output_path = sr_output_dir / filename
            final_output_path = downscale_output_dir / filename

            if final_output_path.exists():
                return  # Già elaborata

            # 4. Applica super-risoluzione
            if not sr_output_path.exists():
                try:
                    sr_output_path = apply_super_resolution_single(image_path, sr_output_dir, self.sr_model)
                except Exception as e:
                    self.logger.log(image_path, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    return

            # 5. Validazione SR
            if not validate_image_with_logging(sr_output_path, "validate_super_resolution", self.logger):
                return

            # 6. Applica downscaling personalizzato
            try:
                final_output_path = apply_personalized_downscaling_single(sr_output_path, downscale_output_dir, ppi=self.ppi)
            except Exception as e:
                self.logger.log(image_path, "downscale", success=False, error=f"Errore downscale: {e}")
                return

            # 7. Validazione downscale
            if not validate_image_with_logging(final_output_path, "validate_downscale", self.logger):
                return

        except Exception as e:
            self.logger.log_crash(error=f"Unexpected error with {image_path}: {e}", full_path=image_path)
