import time
import random

def simulate_processing(logger, images, crash_at=None):
    try:
        for i, img in enumerate(images, start=1):
            # Simula step super_resolution
            logger.log(img, "super_resolution", success=True)

            # Simula step verify_sr
            logger.log(img, "verify_sr", success=True)

            # Se crash_at è raggiunto, solleva eccezione
            if crash_at is not None and i == crash_at:
                raise RuntimeError(f"Simulated crash at image {img}")

            # Simula step completed
            logger.log(img, "completed", success=True)

            time.sleep(0.1)  # simula lavoro

    except Exception as e:
        logger.log_crash(str(e))

    finally:
        logger.stop()


if __name__ == "__main__":
    from logs.logger import CSVLogger  # importa la classe CSVLogger che hai salvato

    images_run_1 = [f"image_{i}.png" for i in range(1, 11)]
    images_run_2 = [f"image_{i}.png" for i in range(1, 21)]

    print("=== Run 1: con crash alla 6ª immagine ===", len(images_run_1))
    logger1 = CSVLogger("logs/test_log.csv")
    simulate_processing(logger1, images_run_1, crash_at=6)

    print("=== Run 2: senza crash ===", len(images_run_2))
    logger2 = CSVLogger("logs/test_log.csv")
    simulate_processing(logger2, images_run_2)

    print("Test completato. Controlla il file logs/test_log.csv")
