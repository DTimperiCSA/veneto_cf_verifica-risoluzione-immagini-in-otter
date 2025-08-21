from pathlib import Path
from src.utils import *
from src.paths import *
from src.config import *
from src.image_utils import *
from src.image_processing import apply_super_resolution_single, apply_personalized_downscaling_single
from logs.logger import CSVLogger

class ImageWorker:
    def __init__(self, logger: CSVLogger, output_sr_dir: Path, output_final_dir: Path, sr_model, ppi: int = None):
        self.logger = logger
        self.output_sr_dir = output_sr_dir
        self.output_final_dir = output_final_dir
        self.sr_model = sr_model
        self.ppi = ppi
        print(f"[INIT] Worker creato per SR dir: {output_sr_dir}, Downscale dir: {output_final_dir}")

    def run(self, image_path: Path):
        print(f"[RUN] Inizio elaborazione: {image_path}")
        try:
            filename = image_path.name
            top_folder = image_path.parent.name
            complete_path = image_path.parent

            # Directory output
            sr_output_dir = self.output_sr_dir / top_folder
            downscale_output_dir = self.output_final_dir / top_folder
            sr_output_path = sr_output_dir / filename
            final_output_path = downscale_output_dir / filename

            print(f"[PATHS] SR: {sr_output_path}, Final: {final_output_path}")

            if final_output_path.exists():
                print(f"[SKIP] File già esistente: {final_output_path}")
                return  # Già elaborata

            # 4. Applica super-risoluzione
            if not sr_output_path.exists():
                try:
                    print(f"[SR] Applico super-risoluzione a {image_path}")
                    sr_output_path = apply_super_resolution_single(image_path, sr_output_dir, self.sr_model)
                    print(f"[SR] Salvata SR: {sr_output_path}")
                except Exception as e:
                    self.logger.log(image_path, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    print(f"[ERROR] Super-risoluzione fallita: {e}")
                    return

            # 5. Validazione SR
            print(f"[VALIDATE] Validazione SR: {sr_output_path}")
            if not validate_image_with_logging(sr_output_path, "validate_super_resolution", self.logger):
                print(f"[VALIDATE] SR non valida: {sr_output_path}")
                return

            # 6. Applica downscaling personalizzato
            try:
                print(f"[DOWNSCALE] Applico downscaling a {sr_output_path} con PPI={self.ppi}")
                final_output_path = apply_personalized_downscaling_single(sr_output_path, downscale_output_dir, ppi=self.ppi)
                print(f"[DOWNSCALE] Salvata immagine finale: {final_output_path}")
            except Exception as e:
                self.logger.log(image_path, "downscale", success=False, error=f"Errore downscale: {e}")
                print(f"[ERROR] Downscale fallito: {e}")
                return

            # 7. Validazione downscale
            print(f"[VALIDATE] Validazione Downscale: {final_output_path}")
            if not validate_image_with_logging(final_output_path, "validate_downscale", self.logger):
                print(f"[VALIDATE] Downscale non valida: {final_output_path}")
                return

            print(f"[SUCCESS] Elaborazione completata: {image_path}")

        except Exception as e:
            self.logger.log_crash(error=f"Unexpected error with {image_path}: {e}", full_path=image_path)
            print(f"[CRASH] Errore imprevisto con {image_path}: {e}")
