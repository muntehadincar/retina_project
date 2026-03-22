import os
import cv2
from PIL import Image
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_IMG_DIR = os.path.join(BASE_DIR, "data/raw/images_tif")
RAW_MASK_DIR = os.path.join(BASE_DIR, "data/raw/masks_gif")

OUT_IMG_DIR = os.path.join(BASE_DIR, "data/processed/drive/images_png")
OUT_MASK_DIR = os.path.join(BASE_DIR, "data/processed/drive/masks_png")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("RAW_MASK_DIR:", RAW_MASK_DIR)
print("OUT_MASK_DIR:", OUT_MASK_DIR)
print("RAW_MASK_DIR exists?:", os.path.exists(RAW_MASK_DIR))
if os.path.exists(RAW_MASK_DIR):
    print("RAW masks count:", len(os.listdir(RAW_MASK_DIR)))


def convert_images():
    print("Images converting...")
    for fname in os.listdir(RAW_IMG_DIR):
        if fname.lower().endswith((".tif", ".tiff")):
            path = os.path.join(RAW_IMG_DIR, fname)
            img = cv2.imread(path)

            if img is None:
                print("Image okunamadı:", fname)
                continue

            new_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, new_name), img)

def convert_masks():
    print("Masks converting...")
    for fname in os.listdir(RAW_MASK_DIR):
        if fname.lower().endswith(".gif"):
            path = os.path.join(RAW_MASK_DIR, fname)

            try:
                mask_pil = Image.open(path).convert("L")
                mask = np.array(mask_pil)
            except Exception as e:
                print("Mask okunamadı:", fname, e)
                continue

            new_name = os.path.splitext(fname)[0] + ".png"
            cv2.imwrite(os.path.join(OUT_MASK_DIR, new_name), mask)

if __name__ == "__main__":
    convert_images()
    convert_masks()
    print("DONE ✅")