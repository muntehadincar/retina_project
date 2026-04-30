"""
train_kfold.py
==============
Veri bölme stratejisi: 5-Fold Cross Validation

Akis:
  1. Test seti (test_ids.txt) tamamen disarda tutulur.
  2. Kalan tum goruntular 5 esit parca yapilir.
  3. Her fold icin: %80 egitim, %20 dogrulama
  4. Model her foldda sifirdan egitilir.
  5. 5 foldin ortalama Dice +/- Std raporlanir.
  6. Ardindan evaluate.py ile test seti uzerinde final degerlendirilir.

Kullanim:
    python src/train_kfold.py --model resunet
    python src/train_kfold.py --model unet
    python src/train_kfold.py --model attention_unet
    python src/train_kfold.py --model segformer
    python src/train_kfold.py --model swinunet

Ciktilar:
    results/models/kfold/<model>_fold<N>_best.pth
    results/logs/kfold/<model>_fold<N>_history.csv
    results/logs/kfold_summary.csv
"""

import os
import csv
import json
import argparse
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from model import ResUNet, UNet, AttentionUNet, SegFormerLite, SwinUNet
from utils import CombinedLoss, compute_metrics
from train import DriveVesselDataset, IMG_DIR, MASK_DIR, TEST_IDS_FILE, get_id


# ─── Eğitim Ayarları ──────────────────────────────────────────────────────────
K_FOLDS = 5
EPOCHS  = 25
SEED    = 42

# ─── Klasörler ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "models", "kfold")
LOG_DIR  = os.path.join(BASE_DIR, "results", "logs",   "kfold")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Model Tanımları ──────────────────────────────────────────────────────────
# Her model için hiperparametreler literatürde yaygın kullanılan değerlerdir.
# Veri bölme yöntemi olarak K-Fold CV kullanılmaktadır.
MODEL_CONFIGS = {
    "unet": {
        "class":      UNet,
        "img_size":   256,
        "lr":         1e-4,
        "batch_size": 2,
        "pos_weight": 5.0,
    },
    "attention_unet": {
        "class":      AttentionUNet,
        "img_size":   256,
        "lr":         1e-4,
        "batch_size": 2,
        "pos_weight": 5.0,
    },
    "resunet": {
        "class":      ResUNet,
        "img_size":   256,
        "lr":         1e-4,
        "batch_size": 2,
        "pos_weight": 5.0,
    },
    "segformer": {
        "class":      SegFormerLite,
        "img_size":   256,
        "lr":         1e-4,
        "batch_size": 2,
        "pos_weight": 5.0,
    },
    "swinunet": {
        "class":      SwinUNet,
        "img_size":   224,
        "lr":         1e-4,
        "batch_size": 2,
        "pos_weight": 5.0,
    },
}

# ─── best_hyperparams.json varsa, bulunan en iyi parametreleri otomatik yükle ─
HYPERPARAM_JSON = os.path.join(BASE_DIR, "results", "logs", "best_hyperparams.json")

def _load_best_hyperparams():
    """Hiperparametre aramasından bulunan en iyi değerleri MODEL_CONFIGS'e uygular."""
    if not os.path.exists(HYPERPARAM_JSON):
        return False
    with open(HYPERPARAM_JSON) as f:
        best = json.load(f)
    updated = []
    for model_name, params in best.items():
        if model_name in MODEL_CONFIGS:
            if "lr" in params:
                MODEL_CONFIGS[model_name]["lr"] = params["lr"]
            if "batch_size" in params:
                MODEL_CONFIGS[model_name]["batch_size"] = params["batch_size"]
            if "pos_weight" in params:
                MODEL_CONFIGS[model_name]["pos_weight"] = params["pos_weight"]
            updated.append(model_name)
    return updated

_hp_updated = _load_best_hyperparams()

METRIC_KEYS = ["dice", "iou", "precision", "accuracy", "sensitivity", "specificity"]


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def kfold_split(items, k=5, seed=42):
    """Deterministik K-Fold bolme — ayni seed her calisirmada ayni bolumleri uretir."""
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    fold_size = len(items) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end   = start + fold_size if i < k - 1 else len(items)
        val   = items[start:end]
        train = items[:start] + items[end:]
        folds.append((train, val))
    return folds


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

    n = max(len(loader), 1)
    return total_loss / n, {k: v / n for k, v in sum_m.items()}


# ─── Ana Eğitim Fonksiyonu ────────────────────────────────────────────────────

def train_kfold_model(model_name, cfg, all_ids, device):
    model_class = cfg["class"]
    img_size    = cfg["img_size"]
    lr          = cfg["lr"]
    bs          = cfg["batch_size"]
    pw          = cfg["pos_weight"]

    folds = kfold_split(all_ids, k=K_FOLDS, seed=SEED)

    print(f"\n{'='*62}")
    print(f"  {model_name.upper()}  |  {K_FOLDS}-Fold Cross Validation")
    print(f"  Parametreler: LR={lr}  Batch={bs}  PosWeight={pw}")
    print(f"  Goruntu boyutu: {img_size}x{img_size}  |  Epoch/fold: {EPOCHS}")
    print(f"{'='*62}")

    csv_fields    = ["epoch", "train_loss", "train_dice", "val_loss", "val_dice"]
    fold_best_dice = []

    for fold_idx, (train_ids, val_ids) in enumerate(folds, 1):
        print(f"\n  ── Fold {fold_idx}/{K_FOLDS}  "
              f"(train={len(train_ids)}, val={len(val_ids)}) ──")

        train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True,  size=img_size)
        val_ds   = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids,   augment=False, size=img_size)

        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)

        model     = model_class().to(device)
        loss_fn   = CombinedLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_dice = -1.0
        best_path = os.path.join(OUT_DIR, f"{model_name}_fold{fold_idx}_best.pth")
        hist_path = os.path.join(LOG_DIR, f"{model_name}_fold{fold_idx}_history.csv")

        with open(hist_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_m = run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
            va_loss, va_m = run_epoch(model, val_loader,   loss_fn, optimizer, device, train=False)

            print(f"    Epoch {epoch:02d}/{EPOCHS} | "
                  f"train_dice={tr_m['dice']:.4f}  val_dice={va_m['dice']:.4f}", flush=True)

            with open(hist_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow({
                    "epoch":      epoch,
                    "train_loss": round(tr_loss,       6),
                    "train_dice": round(tr_m["dice"],  6),
                    "val_loss":   round(va_loss,       6),
                    "val_dice":   round(va_m["dice"],  6),
                })

            if va_m["dice"] > best_dice:
                best_dice = va_m["dice"]
                torch.save(model.state_dict(), best_path)

        print(f"  Fold {fold_idx} tamamlandi. En iyi Val Dice: {best_dice:.4f}")
        fold_best_dice.append(best_dice)

    mean_d = float(np.mean(fold_best_dice))
    std_d  = float(np.std(fold_best_dice))

    print(f"\n  {'─'*50}")
    print(f"  {model_name.upper()} 5-Fold Sonucu:")
    print(f"  Ortalama Val Dice : {mean_d:.4f}  ±  {std_d:.4f}")
    for i, d in enumerate(fold_best_dice, 1):
        print(f"    Fold {i}: {d:.4f}")
    print(f"  {'─'*50}\n")

    return mean_d, std_d, fold_best_dice


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="5-Fold Cross Validation ile model egitimi"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Hangi modeli egitmek istiyorsun? Bos birakirsan hepsi calisir."
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Test setini tamamen disarda tut
    test_ids = set()
    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = {line.strip() for line in f if line.strip()}
        print(f"Test seti harici tutuldu: {len(test_ids)} goruntu")

    # Tum egitim aday goruntuleri
    all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids  = sorted(list({get_id(f) for f in all_imgs} - test_ids))
    print(f"K-Fold icin toplam goruntu: {len(all_ids)}"
          f"  (her fold: ~{len(all_ids)*4//5} train, ~{len(all_ids)//5} val)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Hiperparametre kaynağını bildir
    if _hp_updated:
        print(f"\n  [INFO] best_hyperparams.json bulundu — optimize edilmis parametreler yuklendi:")
        for m in _hp_updated:
            cfg = MODEL_CONFIGS[m]
            print(f"    {m}: LR={cfg['lr']}  Batch={cfg['batch_size']}  PosW={cfg['pos_weight']}")
    else:
        print(f"\n  [INFO] best_hyperparams.json bulunamadi — varsayilan parametreler kullaniliyor.")
        print(f"         Once calistir: python src/hyperparameter_search.py")

    # Hangi modeller calisacak?
    models_to_run = (
        {args.model: MODEL_CONFIGS[args.model]} if args.model
        else MODEL_CONFIGS
    )

    # Ozet CSV
    summary_path = os.path.join(BASE_DIR, "results", "logs", "kfold_summary.csv")
    sum_fields   = ["model", "mean_dice", "std_dice"] + [f"fold{i}_dice" for i in range(1, K_FOLDS + 1)]
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=sum_fields).writeheader()

    # Egitim dongüsü
    for model_name, cfg in models_to_run.items():
        mean_d, std_d, fold_dices = train_kfold_model(model_name, cfg, all_ids, device)

        row = {"model": model_name, "mean_dice": round(mean_d, 4), "std_dice": round(std_d, 4)}
        for i, d in enumerate(fold_dices, 1):
            row[f"fold{i}_dice"] = round(d, 4)

        with open(summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=sum_fields).writerow(row)

    print(f"\n{'='*62}")
    if args.model:
        print(f"  {args.model.upper()} K-FOLD EGITIMI TAMAMLANDI")
        print(f"  Sonraki model → python src/train_kfold.py --model <diger_model>")
        print(f"  Tum modeller bitince → python src/evaluate.py")
    else:
        print("  TUM MODELLER TAMAMLANDI → python src/evaluate.py")
    print(f"  Ozet dosyasi: {summary_path}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
