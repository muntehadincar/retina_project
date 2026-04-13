"""
train_resunet.py
----------------
ResUNet'i 30 epoch boyunca eğitir.
Tam metrik tablosu her epoch sonunda yazdırılır.

Kaydedilen dosyalar:
  results/models/resunet_best.pth
  results/models/resunet_last.pth
  results/logs/resunet_history.csv
"""

import os
import random
import csv

import torch
from torch.utils.data import DataLoader

from model import ResUNet
from utils import CombinedLoss, compute_metrics
from train import DriveVesselDataset, IMG_DIR, MASK_DIR, TEST_IDS_FILE, get_id

# ─── Ayarlar ────────────────────────────────────────────────────────────────
EPOCHS     = 30
BATCH_SIZE = 2
LR         = 1e-4
POS_WEIGHT = 5.0
SEED       = 42
IMG_SIZE   = 256
MODEL_NAME = "resunet"

# ─── Dizinler ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "models")
LOG_DIR  = os.path.join(BASE_DIR, "results", "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

METRIC_KEYS = ["dice", "iou", "precision", "recall",
               "accuracy", "sensitivity", "specificity"]

CSV_FIELDS  = (["epoch", "train_loss", "train_dice", "val_loss", "val_dice"] +
               [f"train_{k}" for k in METRIC_KEYS if k != "dice"] +
               [f"val_{k}"   for k in METRIC_KEYS if k != "dice"])


# ─── Tek Epoch ──────────────────────────────────────────────────────────────
def run_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    sum_m = {k: 0.0 for k in METRIC_KEYS}

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            bm = compute_metrics(logits.detach(), y.detach())
            for k in METRIC_KEYS:
                sum_m[k] += bm[k]

    n = len(loader)
    return total_loss / n, {k: v / n for k, v in sum_m.items()}


# ─── Ana Fonksiyon ──────────────────────────────────────────────────────────
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Test ID'leri yükle
    test_ids = set()
    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = {line.strip() for line in f if line.strip()}
        print(f"Test ID hariç tutuldu: {len(test_ids)} adet")
    else:
        print("UYARI: test_ids.txt bulunamadı — önce split_test.py çalıştırın.")

    # Train / Val split
    all_imgs  = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids   = sorted({get_id(f) for f in all_imgs} - test_ids)
    random.shuffle(all_ids)
    n_train   = int(0.8 * len(all_ids))
    train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]
    print(f"Split → train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True,  size=IMG_SIZE)
    val_ds   = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids,   augment=False, size=IMG_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model     = ResUNet().to(device)
    loss_fn   = CombinedLoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_dice    = -1.0
    best_path    = os.path.join(OUT_DIR, f"{MODEL_NAME}_best.pth")
    last_path    = os.path.join(OUT_DIR, f"{MODEL_NAME}_last.pth")
    history_path = os.path.join(LOG_DIR, f"{MODEL_NAME}_history.csv")

    # CSV başlık yaz
    with open(history_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    print(f"{'='*62}")
    print(f"  Model: {MODEL_NAME.upper()}   |   Epochs: {EPOCHS}   |   Device: {device}")
    print(f"{'='*62}")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
        va_loss, va_m = run_epoch(model, val_loader,   loss_fn, optimizer, device, train=False)

        print(f"\nEpoch {epoch:02d}/{EPOCHS}  "
              f"| train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        print(f"  {'Metrik':<14} {'Train':>9} {'Val':>9}")
        print("  " + "─" * 34)
        for k in METRIC_KEYS:
            print(f"  {k:<14} {tr_m[k]:>9.4f} {va_m[k]:>9.4f}")

        # CSV kayıt
        row = {
            "epoch":      epoch,
            "train_loss": round(tr_loss,          6),
            "train_dice": round(tr_m["dice"],      6),
            "val_loss":   round(va_loss,           6),
            "val_dice":   round(va_m["dice"],      6),
        }
        for k in METRIC_KEYS:
            if k != "dice":
                row[f"train_{k}"] = round(tr_m[k], 6)
                row[f"val_{k}"]   = round(va_m[k], 6)
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)

        # Checkpoint
        torch.save(model.state_dict(), last_path)
        if va_m["dice"] > best_dice:
            best_dice = va_m["dice"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ Best güncellendi: val_dice={best_dice:.4f}")

    print(f"\n{'='*62}")
    print(f"  {MODEL_NAME.upper()} eğitimi tamamlandı.")
    print(f"  En iyi val Dice : {best_dice:.4f}")
    print(f"  Model           : {best_path}")
    print(f"  Log             : {history_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
