"""
train_compare.py
----------------
UNet ve AttentionUNet'i ayni veri bolumu, kayip fonksiyonu ve
augmentation ayarlari altinda ard ardina egitir.

Her epoch sonunda iki modelin metriklerini yan yana yazar:

  Epoch 01  |  loss: UNet train=0.42  val=0.50   |  AttUNet train=0.40  val=0.48
    Metric         UNet-Tr  UNet-Val  Att-Tr   Att-Val
    ─────────────────────────────────────────────────
    dice           0.7234   0.6891   0.7412   0.7100
    ...

Kaydedilen modeller:
  results/models/unet_best.pth
  results/models/attention_unet_best.pth
"""

import os
import random

import torch
from torch.utils.data import DataLoader

from model import UNet, AttentionUNet
from utils import CombinedLoss, compute_metrics
from train import DriveVesselDataset, IMG_DIR, MASK_DIR, TEST_IDS_FILE, get_id

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
EPOCHS     = 30
BATCH_SIZE = 2
LR         = 1e-4
POS_WEIGHT = 5.0
SEED       = 42

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
OUT_DIR    = os.path.join(BASE_DIR, "results", "models")
os.makedirs(OUT_DIR, exist_ok=True)

METRIC_KEYS = ["dice", "iou", "precision", "recall",
               "accuracy", "sensitivity", "specificity"]


# ---------------------------------------------------------------------------
# Yardimci: tek epoch
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # --- test ID'lerini yukle
    test_ids = set()
    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = set(line.strip() for line in f if line.strip())
        print(f"Test ID haric: {len(test_ids)} adet")
    else:
        print("UYARI: test_ids.txt bulunamadi. Once split_test.py calistirin.")

    # --- train / val split
    all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids  = sorted(list({get_id(f) for f in all_imgs} - test_ids))
    random.shuffle(all_ids)
    n        = len(all_ids)
    n_train  = int(0.8 * n)
    train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]
    print(f"Split -> train: {len(train_ids)}, val: {len(val_ids)}")

    train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True)
    val_ds   = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- modeller
    unet     = UNet().to(device)
    att_unet = AttentionUNet().to(device)

    # --- kayip & optimizer (her model icin ayri)
    loss_fn   = CombinedLoss(pos_weight=POS_WEIGHT)
    opt_unet  = torch.optim.Adam(unet.parameters(),     lr=LR)
    opt_att   = torch.optim.Adam(att_unet.parameters(), lr=LR)

    best_unet_dice = -1.0
    best_att_dice  = -1.0

    unet_best_path = os.path.join(OUT_DIR, "unet_best.pth")
    att_best_path  = os.path.join(OUT_DIR, "attention_unet_best.pth")
    unet_last_path = os.path.join(OUT_DIR, "unet_last.pth")
    att_last_path  = os.path.join(OUT_DIR, "attention_unet_last.pth")

    # --- egitim dongusu
    for epoch in range(1, EPOCHS + 1):
        u_tr_loss, u_tr_m = run_epoch(unet,     train_loader, loss_fn, opt_unet, device, train=True)
        u_va_loss, u_va_m = run_epoch(unet,     val_loader,   loss_fn, opt_unet, device, train=False)
        a_tr_loss, a_tr_m = run_epoch(att_unet, train_loader, loss_fn, opt_att,  device, train=True)
        a_va_loss, a_va_m = run_epoch(att_unet, val_loader,   loss_fn, opt_att,  device, train=False)

        # --- ozet baslik
        print(f"\nEpoch {epoch:02d}"
              f"  |  UNet loss: train={u_tr_loss:.4f} val={u_va_loss:.4f}"
              f"  |  AttUNet loss: train={a_tr_loss:.4f} val={a_va_loss:.4f}")

        # --- metrik tablosu
        hdr = f"  {'Metric':<14} {'UNet-Tr':>9} {'UNet-Val':>9} {'Att-Tr':>9} {'Att-Val':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in METRIC_KEYS:
            print(f"  {k:<14} {u_tr_m[k]:>9.4f} {u_va_m[k]:>9.4f}"
                  f" {a_tr_m[k]:>9.4f} {a_va_m[k]:>9.4f}")

        # --- son kontrol noktalari
        torch.save(unet.state_dict(),     unet_last_path)
        torch.save(att_unet.state_dict(), att_last_path)

        # --- en iyi UNet
        if u_va_m["dice"] > best_unet_dice:
            best_unet_dice = u_va_m["dice"]
            torch.save(unet.state_dict(), unet_best_path)
            print(f"  >> UNet best guncellendi: val dice={best_unet_dice:.4f}")

        # --- en iyi AttentionUNet
        if a_va_m["dice"] > best_att_dice:
            best_att_dice = a_va_m["dice"]
            torch.save(att_unet.state_dict(), att_best_path)
            print(f"  >> AttUNet best guncellendi: val dice={best_att_dice:.4f}")

    # --- sonuc karsilastirmasi
    print("\n" + "=" * 55)
    print("EGITIM TAMAMLANDI")
    print(f"  UNet      en iyi val dice : {best_unet_dice:.4f}  -> {unet_best_path}")
    print(f"  AttUNet   en iyi val dice : {best_att_dice:.4f}  -> {att_best_path}")
    winner = "AttentionUNet" if best_att_dice > best_unet_dice else "UNet"
    print(f"  Kazanan model            : {winner}")
    print("=" * 55)


if __name__ == "__main__":
    main()
  