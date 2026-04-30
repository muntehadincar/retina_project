"""
hyperparameter_search.py  (CPU-Uyumlu — Genisletilmis Arama)
=============================================================
3 LR x 2 Batch x 3 PosWeight = 18 kombinasyon/model, 10 epoch/kombin.

Ciktilar:
  results/logs/<model>_hyperparam_search.csv
  results/logs/best_hyperparams.json
"""

import os, csv, json, argparse, itertools, random, time
import torch, numpy as np
from torch.utils.data import DataLoader
from model import ResUNet, UNet, AttentionUNet, SegFormerLite, SwinUNet
from utils import CombinedLoss, compute_metrics
from train import DriveVesselDataset, IMG_DIR, MASK_DIR, TEST_IDS_FILE, get_id

SEARCH_EPOCHS  = 10
SEARCH_IMGSIZE = 128
SEED           = 42

LR_LIST         = [1e-3, 5e-4, 1e-4]
BATCH_SIZE_LIST = [2, 4]
POS_WEIGHT_LIST = [3.0, 5.0, 7.0]

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "models", "hyperopt")
LOG_DIR  = os.path.join(BASE_DIR, "results", "logs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

ALL_MODELS = {
    "unet":           {"class": UNet,           "img_size": SEARCH_IMGSIZE},
    "attention_unet": {"class": AttentionUNet,   "img_size": SEARCH_IMGSIZE},
    "resunet":        {"class": ResUNet,         "img_size": SEARCH_IMGSIZE},
    "segformer":      {"class": SegFormerLite,   "img_size": SEARCH_IMGSIZE},
    "swinunet":       {"class": SwinUNet,        "img_size": 128},
}

METRIC_KEYS = ["dice", "iou", "accuracy", "sensitivity", "specificity"]


def run_epoch(model, loader, loss_fn, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    sum_m = {k: 0.0 for k in METRIC_KEYS}
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
            bm = compute_metrics(logits.detach(), y.detach())
            for k in METRIC_KEYS:
                sum_m[k] += bm[k]
    n = max(len(loader), 1)
    return total_loss / n, {k: v / n for k, v in sum_m.items()}


def search_one_model(model_name, model_cfg, train_ids, val_ids, device):
    model_class = model_cfg["class"]
    img_size = model_cfg["img_size"]
    param_grid = list(itertools.product(LR_LIST, BATCH_SIZE_LIST, POS_WEIGHT_LIST))

    print(f"\n{'='*65}")
    print(f"  {model_name.upper()}  |  {len(param_grid)} kombinasyon  |  {SEARCH_EPOCHS} epoch/kombin")
    print(f"  Goruntu boyutu: {img_size}x{img_size}  |  Device: {device}")
    print(f"  LR={LR_LIST}  Batch={BATCH_SIZE_LIST}  PosW={POS_WEIGHT_LIST}")
    print(f"{'='*65}")

    csv_fields = [
        "sim_id", "lr", "batch_size", "pos_weight",
        "best_val_dice", "val_iou", "val_accuracy",
        "val_sensitivity", "val_specificity", "elapsed_sec",
    ]
    results_path = os.path.join(LOG_DIR, f"{model_name}_hyperparam_search.csv")
    with open(results_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_global_dice = -1.0
    best_global_params = {"lr": 1e-4, "batch_size": 2, "pos_weight": 5.0}

    for idx, (lr, bs, pw) in enumerate(param_grid, 1):
        print(f"\n  [{idx:02d}/{len(param_grid)}] LR={lr:<8} Batch={bs}  PosW={pw}", flush=True)
        t0 = time.time()

        train_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, train_ids, augment=True, size=img_size)
        val_ds = DriveVesselDataset(IMG_DIR, MASK_DIR, val_ids, augment=False, size=img_size)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

        model = model_class().to(device)
        loss_fn = CombinedLoss(pos_weight=pw)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_dice = -1.0
        best_val_m = {k: 0.0 for k in METRIC_KEYS}

        for epoch in range(1, SEARCH_EPOCHS + 1):
            run_epoch(model, train_loader, loss_fn, optimizer, device, train=True)
            _, va_m = run_epoch(model, val_loader, loss_fn, optimizer, device, train=False)
            if va_m["dice"] > best_val_dice:
                best_val_dice = va_m["dice"]
                best_val_m = va_m.copy()

        elapsed = time.time() - t0
        print(f"    -> Dice={best_val_dice:.4f} IoU={best_val_m['iou']:.4f} ({elapsed:.0f}s)")

        row = {
            "sim_id": idx, "lr": lr, "batch_size": bs, "pos_weight": pw,
            "best_val_dice": round(best_val_dice, 4),
            "val_iou": round(best_val_m["iou"], 4),
            "val_accuracy": round(best_val_m["accuracy"], 4),
            "val_sensitivity": round(best_val_m["sensitivity"], 4),
            "val_specificity": round(best_val_m["specificity"], 4),
            "elapsed_sec": round(elapsed, 1),
        }
        with open(results_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if best_val_dice > best_global_dice:
            best_global_dice = best_val_dice
            best_global_params = {"lr": lr, "batch_size": bs, "pos_weight": pw}
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, f"{model_name}_best_hyper_model.pth"))

    print(f"\n  >> {model_name.upper()} En iyi Dice: {best_global_dice:.4f}")
    print(f"     Optimum: {best_global_params}")
    return best_global_params, best_global_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        choices=list(ALL_MODELS.keys()))
    args = parser.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    test_ids = set()
    if os.path.exists(TEST_IDS_FILE):
        with open(TEST_IDS_FILE) as f:
            test_ids = {line.strip() for line in f if line.strip()}

    all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png")])
    all_ids = sorted(list({get_id(f) for f in all_imgs} - test_ids))
    random.shuffle(all_ids)
    n_train = int(0.8 * len(all_ids))
    train_ids, val_ids = all_ids[:n_train], all_ids[n_train:]
    print(f"Veri: {len(train_ids)} train / {len(val_ids)} val  |  Test seti harici")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = {args.model: ALL_MODELS[args.model]} if args.model else ALL_MODELS

    json_path = os.path.join(LOG_DIR, "best_hyperparams.json")
    all_best = {}
    if os.path.exists(json_path):
        with open(json_path) as f:
            all_best = json.load(f)

    t_total = time.time()
    for mname, mcfg in models_to_run.items():
        bp, bd = search_one_model(mname, mcfg, train_ids, val_ids, device)
        all_best[mname] = {
            **bp, "best_dice_search": round(bd, 4),
            "search_config": {
                "lr_list": LR_LIST, "batch_size_list": BATCH_SIZE_LIST,
                "pos_weight_list": POS_WEIGHT_LIST,
                "search_epochs": SEARCH_EPOCHS,
                "search_img_size": SEARCH_IMGSIZE,
                "total_combinations": len(LR_LIST)*len(BATCH_SIZE_LIST)*len(POS_WEIGHT_LIST),
                "seed": SEED,
            },
        }
        with open(json_path, "w") as f:
            json.dump(all_best, f, indent=2)
        print(f"  Kaydedildi: {json_path}\n")

    elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  TAMAMLANDI  ({int(elapsed//3600)}h {int((elapsed%3600)//60)}m)")
    print(f"  Sonraki: python src/plot_hyperparameter_results.py")
    print(f"           python src/train_kfold.py")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
