"""
split_test.py
-------------
Tüm işlenmiş görüntülerin %20'sini rastgele test seti olarak ayırır.
Seçilen görüntüler:
  data/test_im_png/
  data/test_mask_png/
klasörlerine kopyalanır ve seçilen kimlikler
  data/test_ids.txt
dosyasına kaydedilir; böylece train.py onları eğitim/doğrulama bölümünden çıkarır.
"""

import os
import random
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SRC_IMG_DIR  = os.path.join(BASE_DIR, "data", "processed", "drive", "images_png")
SRC_MASK_DIR = os.path.join(BASE_DIR, "data", "processed", "drive", "masks_png")

DST_IMG_DIR  = os.path.join(BASE_DIR, "data", "test_im_png")
DST_MASK_DIR = os.path.join(BASE_DIR, "data", "test_mask_png")
TEST_IDS_FILE = os.path.join(BASE_DIR, "data", "test_ids.txt")

TEST_RATIO = 0.20
SEED = 42


def get_id(fname):
    return fname.split("_")[0]


def main():
    random.seed(SEED)

    # Tüm benzersiz ID'leri topla
    img_files = {get_id(f): f for f in os.listdir(SRC_IMG_DIR) if f.lower().endswith(".png")}
    mask_files = {get_id(f): f for f in os.listdir(SRC_MASK_DIR) if f.lower().endswith(".png")}

    # Hem görüntüsü hem maskesi olan ID'ler
    common_ids = sorted(set(img_files.keys()) & set(mask_files.keys()))
    random.shuffle(common_ids)

    n_test = max(1, int(len(common_ids) * TEST_RATIO))
    test_ids = sorted(common_ids[:n_test])

    print(f"Toplam ID: {len(common_ids)}")
    print(f"Test ID sayısı (%{int(TEST_RATIO*100)}): {n_test}")

    # Hedef klasörleri oluştur
    os.makedirs(DST_IMG_DIR, exist_ok=True)
    os.makedirs(DST_MASK_DIR, exist_ok=True)

    # Görüntüleri ve maskeleri kopyala
    for id_ in test_ids:
        src_img  = os.path.join(SRC_IMG_DIR,  img_files[id_])
        dst_img  = os.path.join(DST_IMG_DIR,  img_files[id_])
        src_mask = os.path.join(SRC_MASK_DIR, mask_files[id_])
        dst_mask = os.path.join(DST_MASK_DIR, mask_files[id_])
        shutil.copy2(src_img,  dst_img)
        shutil.copy2(src_mask, dst_mask)

    # Test ID'lerini kaydet
    with open(TEST_IDS_FILE, "w") as f:
        f.write("\n".join(test_ids))

    print(f"\nTest görüntüleri kopyalandı:")
    print(f"  {DST_IMG_DIR}")
    print(f"  {DST_MASK_DIR}")
    print(f"Test ID listesi kaydedildi: {TEST_IDS_FILE}")
    print("TAMAM - Islem tamamlandi.")


if __name__ == "__main__":
    main()
