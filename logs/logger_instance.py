from logs.logger import CSVLogger
from src.paths import CSV_LOG_PATH

logger = None

def init_logger(reset=False):
    global logger
    if logger is not None:
        return logger
    if reset and CSV_LOG_PATH.exists():
        CSV_LOG_PATH.unlink()
    logger = CSVLogger(CSV_LOG_PATH)
    return logger

def get_logger():
    """
    Restituisce sempre un logger valido. 
    Se non è stato inizializzato, lo crea senza resettare il CSV.
    """
    global logger
    if logger is None:
        init_logger(reset=False)
    return logger