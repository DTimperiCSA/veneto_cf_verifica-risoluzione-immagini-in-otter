import os
import sys
import argparse
import cv2

from super_resolution import SA_SuperResolution

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Script per applicare la super risoluzione"
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path dell'immagine di input oppure della cartella contenente le immagini."
    )
    parser.add_argument(
        "-s", "--scale", type=str, required=True,
        help="Fattore di scala (es. 2, 3 o 4)."
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path di destinazione per salvare l'immagine processata. Se l'input è una cartella, "
             "l'output deve essere una directory."
    )
    args = parser.parse_args()

    # Verifica che l'input esista
    if not os.path.exists(args.input):
        sys.exit(1)

    # Verifica il fattore di scala
    if args.scale not in ["2", "3", "4"]:        
        sys.exit(1)

    # Definisce le estensioni supportate
    allowed_ext = ['.jpg', '.jpeg', '.png', '.tiff']
    image_paths = []

    # Determina se l'input è un una cartella
    if os.path.isdir(args.input):
        # Input è una directory: itera sui file con estensione valida
        for file in os.listdir(args.input):
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_ext:
                image_paths.append(os.path.join(args.input, file))
        if len(image_paths) == 0:
            sys.exit(1)
        # Per input multipli, l'output deve essere una directory
        if not os.path.isdir(args.output):
            os.makedirs(args.output)

    # Controlla che la directory dei modelli esista
    models_dir = "./SA_OCR_Models/super_res"
    if not os.path.exists(models_dir):
        sys.exit(1)
    try:
        # SA_SuperResolution decripta internamente il modello e inizializza l'inferenza
        sr_model = SA_SuperResolution(models_dir, int(args.scale), gpu_id=0)
    except Exception as e:
        sys.exit(1)

    # Processa ogni immagine
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        try:
            output_img = sr_model.run(image)
        except Exception as e:
            continue

        # Determina il percorso di output
      
        if os.path.isdir(args.output):
            input_basename = os.path.basename(image_path)
            name, ext = os.path.splitext(input_basename)
            # Se l'estensione non è riconosciuta, forziamo ".png"
            if ext.lower() not in allowed_ext:
                sys.exit(1)
            output_path = os.path.join(args.output, f"{name}_x{args.scale}{ext}")
        else:
            output_path = args.output

        # Salva l'immagine processata
        cv2.imwrite(output_path, output_img)

if __name__ == "__main__":
    main()

