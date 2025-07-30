from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGES_DIR = BASE_DIR / "images"
INPUT_IMAGES_DIR = IMAGES_DIR / "input"
OUTPUT_IMAGES_DIR = IMAGES_DIR / "output"

CSV_LOG_DIR = BASE_DIR / "logs"
CSV_LOG_PATH = CSV_LOG_DIR / "processing_log.csv"

MODEL_DIR = BASE_DIR / "model"
SR_SCRIPT_MODEL_DIR = MODEL_DIR / "SR_Script" / "super_res"

BENCHMARK_DIR = BASE_DIR / "benchmark"
BENCHMARK_IMAGES_DIR = BENCHMARK_DIR / "images"
CSV_BENCHMARK_LOG_PATH = BENCHMARK_DIR / "benchmark_log.csv"
JSON_BENCHMARK_BEST_CONFIG_PATH = BENCHMARK_DIR / "benchmark_results.json"

