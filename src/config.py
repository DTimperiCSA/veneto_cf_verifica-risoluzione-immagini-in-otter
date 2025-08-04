# Scala per la super risoluzione (cambiare se volete diversa da x2)
SUPER_RESOLUTION_PAR = 2

# Proporzioni (sono empiriche, cioè misurate dalle foto)
# NON TOCCARE
# Le misure sono in px o in mm
CHROMATIC_BAND_MM = 200 #mm
INCH_CONVERSION = 25.4
CHROMATIC_BAND_INCH = CHROMATIC_BAND_MM / INCH_CONVERSION
TARGET_RULER_MM_OUTPUT = 200  # Desired ruler size in mm
TARGET_RULER_INCH_OUTPUT = TARGET_RULER_MM_OUTPUT / INCH_CONVERSION

# 400 PPI
TARGET_RULER_PX_400_PPI = 400 * TARGET_RULER_INCH_OUTPUT
CHROMATIC_BAND_400_PPI = {
    "ppi": 400,
    "width_pixel": 2095.8,
    "width_mm": 133.1,
    "height_pixel": 627,
    "height_mm": 39.8,
    "correction_factor": 400 / 393 
}

# 600 PPI
TARGET_RULER_PX_600_PPI = 600 * CHROMATIC_BAND_INCH  # cioè 190mm a 600ppi

CHROMATIC_BAND_600_PPI = {
    "ppi": 600,
    "width_pixel": 2970,
    "width_mm": 125.7,
    "height_pixel": 881.8,
    "height_mm": 37.3,
    "correction_factor": 600 / 586
}

# A4 AREA
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
TOLERANCE_MM = 2  # Tolleranza per considerare un'immagine A4

# Threashold for binarization
BINARY_THRESHOLD = 50