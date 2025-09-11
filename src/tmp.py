INCH_CONVERSION = 25.4

img_mm = [266.6315, 203.6581]
ppi = 400

img_long_side_mm, img_short_side_mm = max(img_mm), min(img_mm)

img_long_side_inch = img_long_side_mm / INCH_CONVERSION
img_short_side_inch = img_short_side_mm / INCH_CONVERSION

img_long_side_px = img_long_side_inch * ppi
img_short_side_px = img_short_side_inch * ppi

print(f"Lato lungo (px): {img_long_side_px:.2f}")
print(f"Lato corto (px): {img_short_side_px:.2f}")
