import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, "..", "models", "resunet_best.pth")
UPLOAD_DIR  = os.path.join(BASE_DIR, "..", "uploads")
OUTPUT_DIR  = os.path.join(BASE_DIR, "..", "outputs")

IMG_SIZE    = 256
THRESHOLD   = 0.3
DEVICE      = "cpu"   # cuda varsa otomatik tespit edilecek

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
