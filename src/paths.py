from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONSERVATORIO_DIR = Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello\Conservatorio")
DESKTOP_DIR = Path.home() / "Desktop"
REPO_DIR = DESKTOP_DIR / "github" / "Davide Timperi - Conservatorio Venezia"

IMAGES_DIR = BASE_DIR / "images"
INPUT_IMAGES_DIR = DESKTOP_DIR / "B001"
OUTPUT_IMAGES_DIR = Path(r"Z:\Digital Library\Conservatorio Benedetto Marcello") / "resolved_images"
OUTPUT_TMP_DIR = REPO_DIR / "tmp"

TEMPLATE_IMG_PATH = IMAGES_DIR / "assets" / "tiffen_template.tif"

CSV_LOG_DIR = BASE_DIR / "logs"
CSV_LOG_PATH = CSV_LOG_DIR / "processing_log.csv"

MODEL_DIR = BASE_DIR / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
SR_SCRIPT_MODEL_DIR = MODEL_DIR / "SR_Script" / "super_res"
SAVE_PATH = MODEL_DIR / "tiffen_segmenter_best.pth"

BENCHMARK_DIR = BASE_DIR / "benchmark"
BENCHMARK_IMAGES_DIR = BENCHMARK_DIR / "images"
BENCHMARK_INPUT_IMAGES_DIR = CONSERVATORIO_DIR / "B001" / "B001.001"
CSV_BENCHMARK_LOG_PATH = BENCHMARK_DIR / "benchmark_log.csv"
JSON_BENCHMARK_BEST_CONFIG_PATH = BENCHMARK_DIR / "benchmark_results.json"

DATASET_DIR = REPO_DIR / "dataset"
DATASET_TRAIN_IMAGES_DIR = DATASET_DIR / "train_images"
DATASET_TRAIN_MASK_DIR = DATASET_DIR / "train_masks"
DATASET_VAL_IMAGES_DIR = DATASET_DIR / "val_images"
DATASET_VAL_MASK_DIR = DATASET_DIR / "val_masks"
DATASET_TEST_IMAGES_DIR = DATASET_DIR / "test_images"
DATASET_TEST_MASK_DIR = DATASET_DIR / "test_masks"

DATA_TO_SPLIT_DIR = REPO_DIR / "data"
PROVA_DIR = DESKTOP_DIR / "Conservatorio"

SRC_IMAGES = DATA_TO_SPLIT_DIR / "images"               # with masks
SRC_MASKS = DATA_TO_SPLIT_DIR / "masks"
SRC_NO_MASK = DATA_TO_SPLIT_DIR / "images_no_mask"  # no masks
SPLIT_DIR = DATASET_DIR

TMP_SEGMENTATION_DIR = OUTPUT_TMP_DIR / "unet_segmentation"
TMP_SEGMENTATION_MASK_DIR = TMP_SEGMENTATION_DIR / "masks"
TMP_SEGMENTATION_BBOX_DIR = TMP_SEGMENTATION_DIR / "bbox"