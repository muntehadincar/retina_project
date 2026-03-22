# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Test seti uzerinde model degerlendirilmesi yapar.

Hesaplanan metrikler (her goruntu icin ayri, sonra ortalama):
  - Dice Score
  - IoU  (Jaccard)
  - Accuracy
  - Sensitivity  (= Recall)
  - Specificity
  - Precision
  - AUC-ROC  (olasiliksiz versiyon: threshold'suz sigmoid prob ile)

Ciktilar:
  results/evaluation/unet_test_results.csv      <- per-image metrikleri
  results/evaluation/unet_test_summary.txt      <- ozet tablo
  results/evaluation/evaluation_comparison.png  <- model karsilastirma grafigi

Kullanim:
  $env:PYTHONUTF8=1; python src/evaluate.py
"""

import os
import csv
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from train import DriveVesselDataset, get_id
from model import UNet, AttentionUNet

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
TEST_IMG_DIR  = os.path.join(BASE_DIR, "data", "test_im_png")
TEST_MASK_DIR = os.path.join(BASE_DIR, "data", "test_mask_png")
EVAL_DIR      = os.path.join(BASE_DIR, "results", "evaluation")
IMG_SIZE      = 256
THR           = 0.3   # sigmoid esik degeri (train.py ile ayni)

MODELS = {
    "unet": {
        "class": UNet,
        "path":  os.path.join(BASE_DIR, "results", "models", "unet_best.pth"),
        "color": "#4fc3f7",
    },
    "attention_unet": {
        "class": AttentionUNet,
        "path":  os.path.join(BASE_DIR, "results", "models", "attention_unet_best.pth"),
        "color": "#ff7043",
    },
}


# ---------------------------------------------------------------------------
# Yardimci: metrik hesapla (tek goruntu, numpy uzerinde)
# ---------------------------------------------------------------------------
EPS = 1e-7

def compute_metrics_numpy(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """
    pred_bin, gt_bin: boolean veya 0/1 float numpy dizisi (H, W)
    """
    pred = pred_bin.astype(np.float32).ravel()
    gt   = gt_bin.astype(np.float32).ravel()

    TP = float((pred * gt).sum())
    FP = float((pred * (1 - gt)).sum())
    FN = float(((1 - pred) * gt).sum())
    TN = float(((1 - pred) * (1 - gt)).sum())

    dice        = (2 * TP + EPS) / (2 * TP + FP + FN + EPS)
    iou         = (TP + EPS) / (TP + FP + FN + EPS)
    accuracy    = (TP + TN + EPS) / (TP + TN + FP + FN + EPS)
    sensitivity = (TP + EPS) / (TP + FN + EPS)   # recall
    specificity = (TN + EPS) / (TN + FP + EPS)
    precision   = (TP + EPS) / (TP + FP + EPS)
    f1          = dice   # ayni formul

    return {
        "dice":        dice,
        "iou":         iou,
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision":   precision,
    }


# ---------------------------------------------------------------------------
# Tek modeli degerlendir
# ---------------------------------------------------------------------------
def evaluate_model(model_name: str, model_cfg: dict, device: torch.device,
                   test_ids: list, ds) -> dict:
    """Modeli yukleer, her test ornegi uzerinde tahminde bulunur ve metrikleri dondurur."""

    model_path = model_cfg["path"]

    if not os.path.exists(model_path):
        print(f"  [ATLANDI] Model bulunamadi: {model_path}")
        return None

    model = model_cfg["class"]().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Yuklendi: {model_path}")

    per_image = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            logits = model(x.unsqueeze(0).to(device))
            prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred   = (prob > THR).astype(np.float32)
            gt     = y[0].numpy()

            m = compute_metrics_numpy(pred, gt)
            m["sample"] = i + 1
            per_image.append(m)

    return per_image


# ---------------------------------------------------------------------------
# Ozet tablosu yazdir ve kaydet
# ---------------------------------------------------------------------------
METRIC_KEYS = ["dice", "iou", "accuracy", "sensitivity", "specificity", "precision"]
METRIC_LABELS = {
    "dice":        "Dice Score",
    "iou":         "IoU (Jaccard)",
    "accuracy":    "Accuracy",
    "sensitivity": "Sensitivity (Recall)",
    "specificity": "Specificity",
    "precision":   "Precision",
}


def summarize(per_image: list) -> dict:
    summary = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in per_image]
        summary[k] = {
            "mean":  float(np.mean(vals)),
            "std":   float(np.std(vals)),
            "min":   float(np.min(vals)),
            "max":   float(np.max(vals)),
        }
    return summary


def save_csv(per_image: list, path: str, model_name: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["sample"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in per_image:
            w.writerow({k: (f"{row[k]:.6f}" if k != "sample" else row[k]) for k in fields})
    print(f"  Kaydedildi: {path}")


def print_summary_table(model_name: str, summary: dict):
    col = 22
    print(f"\n  {'Metrik':<{col}}  {'Ortalama':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'-'*col}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for k in METRIC_KEYS:
        s = summary[k]
        print(f"  {METRIC_LABELS[k]:<{col}}  {s['mean']:>8.4f}  {s['std']:>7.4f}  {s['min']:>7.4f}  {s['max']:>7.4f}")


def save_summary_txt(all_summaries: dict, path: str, n_samples: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"  TEST SET EVALUATION RESULTS  (n={n_samples}, threshold={THR})\n")
        f.write("=" * 70 + "\n\n")
        for model_name, summary in all_summaries.items():
            f.write(f"--- {model_name.upper()} ---\n")
            f.write(f"  {'Metrik':<24}  {'Ort':>7}  {'Std':>7}  {'Min':>7}  {'Max':>7}\n")
            f.write(f"  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}\n")
            for k in METRIC_KEYS:
                s = summary[k]
                f.write(f"  {METRIC_LABELS[k]:<24}  {s['mean']:>7.4f}  {s['std']:>7.4f}  {s['min']:>7.4f}  {s['max']:>7.4f}\n")
            f.write("\n")
    print(f"  Ozet kaydedildi: {path}")


# ---------------------------------------------------------------------------
# Karsilastirma grafigi
# ---------------------------------------------------------------------------
def plot_comparison(all_summaries: dict, out_dir: str):
    if len(all_summaries) < 1:
        return

    plt.rcParams.update({
        "figure.facecolor":  "#0f1117",
        "axes.facecolor":    "#1a1d27",
        "axes.edgecolor":    "#3a3d4d",
        "axes.labelcolor":   "#e0e0e0",
        "xtick.color":       "#cccccc",
        "ytick.color":       "#cccccc",
        "text.color":        "#e0e0e0",
        "grid.color":        "#2a2d3d",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "font.family":       "DejaVu Sans",
    })

    labels = [METRIC_LABELS[k] for k in METRIC_KEYS]
    x = np.arange(len(METRIC_KEYS))
    bar_w = 0.35 / max(len(all_summaries), 1)
    offsets = np.linspace(-(len(all_summaries)-1)*bar_w/2,
                           (len(all_summaries)-1)*bar_w/2,
                           len(all_summaries))

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    patches = []
    for i, (model_name, summary) in enumerate(all_summaries.items()):
        means  = [summary[k]["mean"] for k in METRIC_KEYS]
        stds   = [summary[k]["std"]  for k in METRIC_KEYS]
        color  = MODELS[model_name]["color"] if model_name in MODELS else "#aaaaaa"
        bars = ax.bar(x + offsets[i], means, width=bar_w * 0.85,
                      color=color, alpha=0.85, zorder=3,
                      yerr=stds, capsize=4, error_kw={"ecolor": "#ffffff88", "lw": 1.2})
        # Ortalama degeri cubuk ustune yaz
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color="#dddddd")
        patches.append(mpatches.Patch(color=color,
                                      label=model_name.replace("_", " ").title()))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Test Set Evaluation — Model Karsilastirmasi",
                 fontsize=13, fontweight="bold", color="#ffffff", pad=10)
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=10)
    ax.grid(True, axis="y", zorder=0)

    out_path = os.path.join(out_dir, "evaluation_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Grafik kaydedildi: {out_path}")


# ---------------------------------------------------------------------------
# Ana Akis
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  EVALUATE.PY  --  Test Set Degerlendirmesi")
    print("=" * 60)
    print(f"  Esik (threshold): {THR}")
    print(f"  Goruntu boyutu  : {IMG_SIZE}x{IMG_SIZE}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test ID'lerini bul
    if not os.path.isdir(TEST_IMG_DIR) or not os.listdir(TEST_IMG_DIR):
        print(f"[HATA] Test klasoru bos veya bulunamadi: {TEST_IMG_DIR}")
        sys.exit(1)

    test_img_files  = {get_id(f): f for f in os.listdir(TEST_IMG_DIR)
                       if f.lower().endswith(".png")}
    test_mask_files = {get_id(f): f for f in os.listdir(TEST_MASK_DIR)
                       if f.lower().endswith(".png")}
    test_ids = sorted(set(test_img_files) & set(test_mask_files))

    if not test_ids:
        print("[HATA] Test klasorunde eslesme bulunamadi.")
        sys.exit(1)

    print(f"Test ornekleri: {len(test_ids)}\n")

    ds = DriveVesselDataset(
        TEST_IMG_DIR, TEST_MASK_DIR,
        ids=test_ids, augment=False, size=IMG_SIZE
    )

    os.makedirs(EVAL_DIR, exist_ok=True)
    all_summaries = {}

    # Her modeli degerlendir
    for model_name, model_cfg in MODELS.items():
        print(f"--- Model: {model_name.upper()} ---")
        per_image = evaluate_model(model_name, model_cfg, device, test_ids, ds)

        if per_image is None:
            continue

        summary = summarize(per_image)
        all_summaries[model_name] = summary

        # Konsola yazdir
        print_summary_table(model_name, summary)

        # Per-image CSV kaydet
        csv_path = os.path.join(EVAL_DIR, f"{model_name}_test_results.csv")
        save_csv(per_image, csv_path, model_name)
        print()

    if not all_summaries:
        print("[HATA] Hicbir model bulunamadi veya degerlendirilemedi.")
        sys.exit(1)

    # Ozet TXT
    summary_path = os.path.join(EVAL_DIR, "test_summary.txt")
    save_summary_txt(all_summaries, summary_path, len(test_ids))

    # Karsilastirma grafigi
    plot_comparison(all_summaries, EVAL_DIR)

    print("\n" + "=" * 60)
    print("  Tamamlandi!")
    print(f"  Cikti klasoru: {EVAL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
