from pathlib import Path
from src.utils import is_valid_image_file, validate_image_with_logging
from src.paths import *
from src.config import *
from src.estimate_ppi_from_ruler import *
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
            complete_path = image_path.parent

            # 0. Cerca banda cromatica
            chromatic_band_img = find_chromatic_band_in_folder(complete_path)
            if not chromatic_band_img:
                print(f"⚠️ Nessuna banda cromatica trovata in {top_folder}.")
                self.logger.log(image_path, "find_chromatic_band", success=False, error="Chromatic band image not found")
                return

            # 1. Validazione immagine input
            if not validate_image_with_logging(image_path, "validate_input", self.logger):
                return
            
            chromatic_band_dim_px = measure_chromatic_band_dimension(chromatic_band_img)

            # 2. Binarizzazione immagine
            binarized_image_path = binaryize_image(image_path)
            if not binarized_image_path:
                self.logger.log(image_path, "binarize", success=False, error="Binarization failed")
                return

            # 3. Trova rettangolo e misura lati in mm
            img_rect_dim_px = measure_document_from_binary(binarized_image_path)
            if img_rect_dim_px is None:
                self.logger.log(image_path, "measure_document", success=False, error="Failed to measure document dimensions")
                return

            # 4. Stima PPI
            estimated_ppi = estimate_ppi_from_dimensions(img_rect_dim_px, chromatic_band_dim_px)
            print(f"🖼️ PPI stimati: {estimated_ppi}")

            # Directory output
            sr_output_dir = self.output_sr_dir / top_folder
            downscale_output_dir = self.output_final_dir / top_folder
            sr_output_path = sr_output_dir / filename
            final_output_path = downscale_output_dir / filename

            if final_output_path.exists():
                return  # Già elaborata

            # 5. Applica super-risoluzione (con PPI stimato, se il modello lo supporta)
            if not sr_output_path.exists():
                try:
                    # Se il modello può ricevere PPI, passalo (modifica la funzione se serve)
                    sr_output_path = apply_super_resolution_single(image_path, sr_output_dir, self.sr_model)
                except Exception as e:
                    self.logger.log(image_path, "super_resolution", success=False, error=f"Errore super_resolution: {e}")
                    return

            # 6. Validazione SR
            if not validate_image_with_logging(sr_output_path, "validate_super_resolution", self.logger):
                return

            # 7. Applica downscaling personalizzato (potresti passare ppi qui se serve)
            try:
                final_output_path = apply_personalized_downscaling_single(sr_output_path, downscale_output_dir, ppi=estimated_ppi)
            except Exception as e:
                self.logger.log(image_path, "downscale", success=False, error=f"Errore downscale: {e}")
                return

            # 8. Validazione downscale
            if not validate_image_with_logging(final_output_path, "validate_downscale", self.logger):
                return

        except Exception as e:
            self.logger.log_crash(error=f"Unexpected error with {image_path}: {e}", full_path=image_path)
