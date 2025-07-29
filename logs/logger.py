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

        # mappa filename -> last error/failure info (una riga per immagine con errore)
        self.rows = {}
        self.fieldnames = ["timestamp", "filename", "step", "status", "error", "full_path"]

        if self.csv_path.exists():
            self._load_existing()

        # thread di autosalvataggio
        self.autosave_thread = threading.Thread(target=self._autosave_loop, daemon=True)
        self.autosave_thread.start()

    def _load_existing(self):
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] == "false":
                    self.rows[row["filename"]] = row

    def log_failure(self, filename: str, step: str, error: str, full_path: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self.lock:
            self.rows[filename] = {
                "timestamp": timestamp,
                "filename": filename,
                "step": step,
                "status": "false",
                "error": error,
                "full_path": full_path,
            }

    def log_crash(self, error: str, full_path: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        with self.lock:
            crash_id = f"CRASH_{timestamp}"
            self.rows[crash_id] = {
                "timestamp": timestamp,
                "filename": "CRASH",
                "step": "CRASH",
                "status": "false",
                "error": error,
                "full_path": full_path,
            }

    def log(self, filename: str, step: str, success: bool, error: str = "", full_path: str = ""):
        """
        Metodo generico di logging.
        - Se success=True, non fa nulla (perché vuoi salvare solo errori).
        - Se success=False, registra un failure.
        """
        if success:
            return
        else:
            self.log_failure(filename, step, error, full_path)

    def print_status(self, total_images: int):
        with self.lock:
            failed = len(self.rows)
            success = total_images - failed
        print(f"[CSVLogger] Immagini processate con successo: {success}, immagini con errore: {failed}")

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
