import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR = os.path.join(BASE_DIR, "images")
INPUT_IMAGES_DIR = os.path.join(IMAGES_DIR, "input")
OUTPUT_IMAGES_DIR = os.path.join(IMAGES_DIR, "output")

CSV_LOG_DIR = os.path.join(BASE_DIR, "logs")
CSV_LOG_PATH = os.path.join(CSV_LOG_DIR, "processing_log.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model")
SR_SCRIPT_MODEL_DIR = os.path.join(MODEL_DIR, "SR_Script", "super_res")
