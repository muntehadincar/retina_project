print(">>> train.py DOSYASI YÜKLENDİ <<<")
import os, random, csv
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from model import UNet
from utils import dice_score, CombinedLoss

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
IMG_DIR   = os.path.join(BASE_DIR, "data", "processed", "drive", "images_png")
MASK_DIR  = os.path.join(BASE_DIR, "data", "processed", "drive", "masks_png")
TEST_IDS_FILE = os.path.join(BASE_DIR, "data", "test_ids.txt")

def get_id(fname):
    return fname.split("_")[0]

class DriveVesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, augment=False, size=512):
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

        # CLAHE (kontrast artırma)
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
    total_loss, total_dice = 0.0, 0.0

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
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
    print("NEW TRAINING LOOP RUNNING")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- test ID'lerini yükle (varsa) ve hariç bırak
    test_ids = set()
    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = set(line.strip() for line in f if line.strip())
        print(f"Test ID'leri yüklendi ({len(test_ids)} adet): {TEST_IDS_FILE}")
    else:
        print("test_ids.txt bulunamadı — split_test.py'yi çalıştırın.")

    # --- train/val split (test görüntüleri hariç)
    all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids  = sorted(list({get_id(f) for f in all_imgs} - test_ids))
    random.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(0.8 * n)
    train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]

    print(f"Split → train: {len(train_ids)}, val: {len(val_ids)}, test (hariç): {len(test_ids)}")

    train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True, size=256)
    val_ds   = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids, augment=False, size=256)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0)

    # --- device/model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet().to(device)

    # --- loss/optimizer
    loss_fn = CombinedLoss(pos_weight=5.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # --- training config
    EPOCHS = 40
    best_val_dice = -1.0

    # --- output paths
    out_dir = os.path.join(BASE_DIR, "results", "models")
    last_path = os.path.join(out_dir, "unet_last.pth")
    best_path = os.path.join(out_dir, "unet_best.pth")

    os.makedirs(out_dir, exist_ok=True)
    print("Model save dir:", out_dir)
    print("Last path:", last_path)
    print("Best path:", best_path)

    # --- CSV history log
    log_dir = os.path.join(BASE_DIR, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    history_path = os.path.join(log_dir, "unet_history.csv")
    csv_fields = ["epoch", "train_loss", "train_dice", "val_loss", "val_dice"]
    # Yeni eğitim başlıyor: mevcut log'u sıfırla
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()
    print("History log:", history_path)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_dice = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va_loss, va_dice = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} dice {tr_dice:.4f} | val loss {va_loss:.4f} dice {va_dice:.4f}")

        # CSV'ye epoch kaydı ekle
        with open(history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "train_dice": round(tr_dice, 6),
                "val_loss":   round(va_loss, 6),
                "val_dice":   round(va_dice, 6),
            })

        # save last
        torch.save(model.state_dict(), last_path)

        # save best
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Best updated: val dice {best_val_dice:.4f} -> saved {best_path}")

    print("Training done.")
    print("Saved last:", last_path)
    print("Saved best:", best_path)
    print("History log:", history_path)

if __name__ == "__main__":
    main()