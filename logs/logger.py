import csv
import threading
import time
from pathlib import Path
from datetime import datetime

class CSVLogger:
    def __init__(self, csv_path: str, autosave_interval: int = 30):
        self.csv_path = Path(csv_path)
        self.autosave_interval = autosave_interval
        self.lock = threading.Lock()
        self.running = True

        # Structure: {filename: row_dict}
        self.rows = {}

        # Define consistent fieldnames
        self.fieldnames = ["timestamp", "directory", "filename", "step", "status", "error"]

        if self.csv_path.exists():
            self._load_existing()

        # Start autosave thread
        self.autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self.autosave_thread.start()

    def _load_existing(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows[row["filename"]] = row

    def log(self, directory: str, filename: str, step: str, success: bool, error: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self.lock:
            self.rows[filename] = {
                "timestamp": timestamp,
                "directory": directory,
                "filename": filename,
                "step": "completed" if success and step == "completed" else (step if success else f"FAILED_AT_{step}"),
                "status": "true" if success else "false",
                "error": "" if success else error,
            }

    def log_crash(self, error: str):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        crash_id = f"CRASH_{timestamp}"
        with self.lock:
            self.rows[crash_id] = {
                "timestamp": timestamp,
                "directory": "",
                "filename": "CRASH",
                "step": "CRASH",
                "status": "false",
                "error": error,
            }

    def _autosave_loop(self):
        while self.running:
            time.sleep(self.autosave_interval)
            self._flush()

    def _flush(self):
        with self.lock:
            tmp_path = self.csv_path.with_suffix(".tmp")
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows.values())
            tmp_path.replace(self.csv_path)

    def stop(self):
        self.running = False
        self.autosave_thread.join()
        self._flush()
