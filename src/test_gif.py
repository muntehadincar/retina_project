from PIL import Image, ImageSequence
import numpy as np

path = r"data/raw/masks_gif/21_training_mask.gif"  # bu dosya gerçekten varsa

im = Image.open(path)
print("mode:", im.mode, "size:", im.size, "frames:", getattr(im, "n_frames", 1))

# ilk frame'i güvenli al
frame0 = next(ImageSequence.Iterator(im)).convert("L")
arr = np.array(frame0)
print("array:", arr.shape, arr.dtype, "min/max:", arr.min(), arr.max())