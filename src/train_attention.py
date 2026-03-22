print(">>> train_attention.py DOSYASI YÜKLENDİ <<<")

import os, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import AttentionUNet
from utils import dice_score

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
IMG_DIR   = os.path.join(BASE_DIR, "data", "processed", "drive", "images_png")
MASK_DIR  = os.path.join(BASE_DIR, "data", "processed", "drive", "masks_png")
TEST_IDS_FILE = os.path.join(BASE_DIR, "data", "test_ids.txt")


def get_id(fname):
    return fname.split("_")[0]


class DriveVesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, augment=False, size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.ids = ids
        self.augment = augment
        self.size = size

        self.img_files = {get_id(f): f for f in os.listdir(img_dir) if f.lower().endswith(".png")}
        self.mask_files = {get_id(f): f for f in os.listdir(mask_dir) if f.lower().endswith(".png")}
        self.ids = [i for i in self.ids if (i in self.img_files and i in self.mask_files)]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        img_path = os.path.join(self.img_dir, self.img_files[id_])
        mask_path = os.path.join(self.mask_dir, self.mask_files[id_])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.float32)
        img = img.astype(np.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask


def run_epoch(model, loader, loss_fn, optimizer, device, train=True):

    model.train() if train else model.eval()

    total_loss = 0.0
    total_dice = 0.0

    with torch.set_grad_enabled(train):

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_fn(logits, y)

            if train:

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score(logits.detach(), y.detach())

    return total_loss / len(loader), total_dice / len(loader)


def main():

    print("NEW ATTENTION UNET TRAINING")

    random.seed(42)

    # test id yükle
    test_ids = set()

    if os.path.exists(TEST_IDS_FILE):

        with open(TEST_IDS_FILE) as f:
            test_ids = set(line.strip() for line in f if line.strip())

        print(f"Test ID'leri yüklendi ({len(test_ids)})")

    # split
    all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids  = sorted(list({get_id(f) for f in all_imgs} - test_ids))

    random.shuffle(all_ids)

    n = len(all_ids)

    n_train = int(0.8 * n)

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:]

    print(f"Split → train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True, size=256)
    val_ds   = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids, augment=False, size=256)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    model = AttentionUNet().to(device)

    pos_weight = torch.tensor([5.0], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 40

    best_val_dice = -1

    out_dir = os.path.join(BASE_DIR, "results", "models")

    os.makedirs(out_dir, exist_ok=True)

    last_path = os.path.join(out_dir, "attention_unet_last.pth")
    best_path = os.path.join(out_dir, "attention_unet_best.pth")

    for epoch in range(1, EPOCHS + 1):

        tr_loss, tr_dice = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)

        va_loss, va_dice = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}")

        torch.save(model.state_dict(), last_path)

        if va_dice > best_val_dice:

            best_val_dice = va_dice

            torch.save(model.state_dict(), best_path)

            print(f"  ✅ Best updated → {best_val_dice:.4f}")

    print("Training done")
    print("Best model:", best_path)


if __name__ == "__main__":
    main()