import time

def simulate_processing(logger, images, crash_at=None):
    try:
        for i, img in enumerate(images, start=1):
            # Skip immagini già processate con successo
            if logger.is_processed(img):
                print(f"Skipping already processed image: {img}")
                continue

            # Simula step super_resolution (qui potresti aggiungere verifiche reali)
            # Qui simuliamo successo, ma se vuoi simulare fallimento puoi aggiungere condizioni
            success_sr = True

            if not success_sr:
                logger.log_failure(img, "super_resolution", error="Simulated SR failure")
                continue  # Skip altri step se fallito

            # Simula step verify_sr
            success_verify = True
            if not success_verify:
                logger.log_failure(img, "verify_sr", error="Simulated verify failure")
                continue

            # Se crash_at è raggiunto, solleva eccezione per simulare crash
            if crash_at is not None and i == crash_at:
                raise RuntimeError(f"Simulated crash at image {img}")

            # Se tutti step sono OK, non logghiamo perché il logger salva solo errori

            # Simula tempo lavoro
            time.sleep(0.1)

    except Exception as e:
        logger.log_crash(str(e))

    finally:
        logger.stop()


if __name__ == "__main__":
    from logs.logger import CSVLogger  # Assumi che CSVLogger sia nella cartella logs/logger.py

    images_run_1 = [f"image_{i}.png" for i in range(1, 11)]
    images_run_2 = [f"image_{i}.png" for i in range(1, 21)]

    print("=== Run 1: con crash alla 6ª immagine ===")
    logger1 = CSVLogger("logs/test_log.csv")
    simulate_processing(logger1, images_run_1, crash_at=6)

    print("\n=== Run 2: senza crash ===")
    logger2 = CSVLogger("logs/test_log.csv")
    simulate_processing(logger2, images_run_2)

    print("\nTest completato. Controlla il file logs/test_log.csv")
