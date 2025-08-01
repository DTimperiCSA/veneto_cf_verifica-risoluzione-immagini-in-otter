import csv
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Union


class CSVLogger:
    def __init__(self, csv_path: Union[str, Path], reset: bool = True):
        self.csv_path = Path(csv_path)
        self.fieldnames = ["timestamp", "filename", "step", "status", "error", "full_path"]

        # Se voglio resettare il file (nuova run), lo elimino all'inizio
        if reset and self.csv_path.exists():
            self.csv_path.unlink()

        # Creo il file e scrivo header solo se non esiste (oppure è stato appena eliminato)
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

        # Queue multiprocessing-safe
        self.queue = multiprocessing.Queue()

        # Flag per gestire il thread di scrittura
        self.running = threading.Event()
        self.running.set()

        # Thread dedicato che consuma la coda e scrive su file CSV
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def log_failure(self, filename: str, step: str, error: str, full_path: str = ""):
        timestamp = datetime.now().isoformat(sep=" ", timespec="seconds")
        entry = {
            "timestamp": timestamp,
            "filename": str(filename),
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

    def log(self, file_path: Union[str, Path], step: str, success: bool, error: str = "", full_path: str = ""):
        if not success:
            self.log_failure(file_path, step, error, full_path)

    def _writer_loop(self):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            while self.running.is_set() or not self.queue.empty():
                try:
                    entry = self.queue.get(timeout=0.5)
                    writer.writerow(entry)
                    f.flush()
                except Exception:
                    pass

    def stop(self):
        self.running.clear()
        self.writer_thread.join()
