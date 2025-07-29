import csv
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime

class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.fieldnames = ["timestamp", "filename", "step", "status", "error", "full_path"]

        # Queue multiprocessing-safe
        self.queue = multiprocessing.Queue()

        # Flag per gestire il thread di scrittura
        self.running = threading.Event()
        self.running.set()

        # Thread dedicato che consuma la coda e scrive su file CSV
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

        # File aperto in modalità append, si scrive header se file nuovo
        self._init_csv_file()

    def _init_csv_file(self):
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_failure(self, filename: str, step: str, error: str, full_path: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        entry = {
            "timestamp": timestamp,
            "filename": filename,
            "step": step,
            "status": "false",
            "error": error,
            "full_path": full_path,
        }
        self.queue.put(entry)

    def log_crash(self, error: str, full_path: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        entry = {
            "timestamp": timestamp,
            "filename": "CRASH",
            "step": "CRASH",
            "status": "false",
            "error": error,
            "full_path": full_path,
        }
        self.queue.put(entry)

    def log(self, filename: str, step: str, success: bool, error: str = "", full_path: str = ""):
        if not success:
            self.log_failure(filename, step, error, full_path)

    def _writer_loop(self):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            while self.running.is_set() or not self.queue.empty():
                try:
                    entry = self.queue.get(timeout=0.5)
                    writer.writerow(entry)
                    f.flush()
                except Exception:
                    # Timeout o queue vuota: passa
                    pass

    def stop(self):
        self.running.clear()
        self.writer_thread.join()