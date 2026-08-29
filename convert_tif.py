import rasterio
import numpy as np
from PIL import Image

def convert_tif_to_png(tif_path, png_path):
    with rasterio.open(tif_path) as src:
        print(f"Bands available: {src.count}")
        print(f"Image size: {src.width} x {src.height}")

        if src.count >= 3:
            # Read first 3 bands as RGB (adjust order if colors look wrong)
            r = src.read(1)
            g = src.read(2)
            b = src.read(3)
        else:
            # Single band - use it for all 3 channels (grayscale-style)
            band = src.read(1)
            r = g = b = band

        def normalize(band):
            band = band.astype(np.float32)
            band_min, band_max = band.min(), band.max()
            if band_max - band_min == 0:
                return np.zeros_like(band, dtype=np.uint8)
            normalized = (band - band_min) / (band_max - band_min) * 255
            return normalized.astype(np.uint8)

        r, g, b = normalize(r), normalize(g), normalize(b)
        rgb = np.dstack((r, g, b))

        Image.fromarray(rgb).save(png_path)
        print(f"Saved: {png_path}")

# UPDATE THIS FILENAME to match your actual file
convert_tif_to_png("real_images/sample.tif", "real_images/converted.png")