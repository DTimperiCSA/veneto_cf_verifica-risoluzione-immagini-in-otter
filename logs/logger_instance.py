# logger_instance.py
from logger import CSVLogger
from src.paths import CSV_LOG_PATH

logger = None

def init_logger(reset: bool = False):
    """
    Inizializza il logger globale. 
    Se reset=True, cancella il file esistente PRIMA di creare il logger.
    """
    global logger
    if logger is not None:
        return logger  # già inizializzato, non fare nulla

    if reset and CSV_LOG_PATH.exists():
        print(f"📜 Log esistente trovato: {CSV_LOG_PATH}. Rimuovo per una nuova esecuzione.")
        CSV_LOG_PATH.unlink()

    logger = CSVLogger(CSV_LOG_PATH)
    return logger
