"""
plot_multi_compare.py
---------------------
Tüm modellerin CSV loglarını okuyarak karşılaştırma grafikleri oluşturur.

Okunan CSV'ler (eğer mevcutsa):
  results/logs/unet_history.csv
  results/logs/attention_unet_history.csv   (henüz yoksa atlanır)
  results/logs/resunet_history.csv
  results/logs/segformer_history.csv
  results/logs/swinunet_history.csv

Kaydedilen grafikler:
  results/plots/val_dice_curves.png     — Her modelin val Dice eğrisi
  results/plots/best_dice_bar.png       — En iyi val Dice karşılaştırması
  results/plots/all_metrics_bar.png     — Tüm metrikler (en iyi epoch)
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")           # ekransız ortamlarda da çalışır
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
LOG_DIR   = os.path.join(BASE_DIR, "results", "logs")
PLOT_DIR  = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Model → CSV dosya adı eşlemesi (sıra önemli)
MODEL_FILES = {
    "UNet":          "unet_history.csv",
    "AttentionUNet": "attention_unet_history.csv",
    "ResUNet":       "resunet_history.csv",
    "SegFormer":     "segformer_history.csv",
    "SwinUNet":      "swinunet_history.csv",
}

# Renk paleti
COLORS = {
    "UNet":          "#4A90D9",
    "AttentionUNet": "#E8694A",
    "ResUNet":       "#57B894",
    "SegFormer":     "#A066CC",
    "SwinUNet":      "#E8B84B",
}

ALL_METRICS = ["dice", "iou", "precision", "recall", "accuracy", "specificity"]
METRIC_LABELS = {
    "dice":        "Dice",
    "iou":         "IoU",
    "precision":   "Precision",
    "recall":      "Recall",
    "accuracy":    "Accuracy",
    "specificity": "Specificity",
}



def load_history(path: str) -> list[dict]:
    """CSV'yi satır listesi olarak döndürür (float değerli)."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def best_row(rows: list[dict]) -> dict:
    """En yüksek val_dice'a sahip satırı döndürür."""
    return max(rows, key=lambda r: r["val_dice"])


#Grafik 1: Val Dice Eğrileri
def plot_val_dice_curves(histories: dict[str, list[dict]]):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for name, rows in histories.items():
        epochs    = [r["epoch"]   for r in rows]
        val_dice  = [r["val_dice"] for r in rows]
        ax.plot(epochs, val_dice,
                color=COLORS[name], linewidth=2.2,
                marker="o", markersize=3.5,
                label=name)

    ax.set_title("Validation Dice — Epoch Karşılaştırması",
                 color="white", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch", color="#CCCCCC", fontsize=11)
    ax.set_ylabel("Val Dice", color="#CCCCCC", fontsize=11)
    ax.tick_params(colors="#AAAAAA")
    ax.spines[:].set_color("#444466")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color="#333355", linewidth=0.6, linestyle="--")
    ax.legend(facecolor="#222244", edgecolor="#555577",
              labelcolor="white", fontsize=10)

    out = os.path.join(PLOT_DIR, "val_dice_curves.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {out}")


# Grafik 2: En İyi Val Dice Bar Grafiği
def plot_best_dice_bar(histories: dict[str, list[dict]]):
    names  = list(histories.keys())
    scores = [best_row(histories[n])["val_dice"] for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    bars = ax.bar(names, scores, color=colors, edgecolor="#333355",
                  linewidth=0.8, width=0.55)

    # Değerleri bar üstüne yaz
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{score:.4f}",
                ha="center", va="bottom",
                color="white", fontsize=9.5, fontweight="bold")

    ax.set_title("En İyi Validation Dice — Model Karşılaştırması",
                 color="white", fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Val Dice", color="#CCCCCC", fontsize=11)
    ax.set_ylim(0, min(scores) * 0.85,)
    # y eksenini anlamlı bir alt sınırdan başlat
    ymin = max(0, min(scores) - 0.10)
    ymax = min(1.0, max(scores) + 0.05)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(colors="#AAAAAA")
    ax.spines[:].set_color("#444466")
    ax.grid(axis="y", color="#333355", linewidth=0.6, linestyle="--")

    out = os.path.join(PLOT_DIR, "best_dice_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {out}")


# Grafik 3: Tüm Metrikler Yan Yana
def plot_all_metrics_bar(histories: dict[str, list[dict]]):
    """
    Her model için en iyi val_dice epoch'undaki val metriklerini gösterir.
    Sadece mevcut metrikleri çizer (unet_history.csv daha az sütun taşıyabilir).
    """
    names    = list(histories.keys())
    n_models = len(names)

    # Hangi metrikler tüm modellerde mevcut?
    available = []
    for m in ALL_METRICS:
        if all(f"val_{m}" in best_row(histories[n]) for n in names):
            available.append(m)

    if not available:
        # Sadece Dice varsa basit bar çiz
        available = ["dice"]

    n_metrics = len(available)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2), 5.5))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    for i, name in enumerate(names):
        br  = best_row(histories[name])
        vals = []
        for m in available:
            key = f"val_{m}" if m != "dice" else "val_dice"
            vals.append(br.get(key, br.get(f"val_{m}", 0.0)))

        offsets = x + (i - n_models / 2 + 0.5) * width
        ax.bar(offsets, vals, width=width * 0.9,
               color=COLORS[name], edgecolor="#222244",
               linewidth=0.6, label=name)

    ax.set_title("Tüm Metrikler — En İyi Epoch'ta Model Karşılaştırması",
                 color="white", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in available],
                       color="#CCCCCC", fontsize=10)
    ax.set_ylabel("Değer", color="#CCCCCC", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors="#AAAAAA")
    ax.spines[:].set_color("#444466")
    ax.grid(axis="y", color="#333355", linewidth=0.6, linestyle="--")
    ax.legend(facecolor="#222244", edgecolor="#555577",
              labelcolor="white", fontsize=9, loc="lower right")

    out = os.path.join(PLOT_DIR, "all_metrics_bar.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ {out}")


# 
def main():
    print(f"\n{'='*60}")
    print("  plot_multi_compare.py — Model Karşılaştırma Grafikleri")
    print(f"{'='*60}\n")

    # Mevcut CSV'leri yükle
    histories = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(LOG_DIR, fname)
        if os.path.exists(path):
            rows = load_history(path)
            if rows:
                histories[name] = rows
                best = best_row(rows)
                print(f"  {name:<16} → {len(rows):2d} epoch yüklendi | "
                      f"best val_dice = {best['val_dice']:.4f}")
            else:
                print(f"  {name:<16} → CSV boş, atlandı: {path}")
        else:
            print(f"  {name:<16} → CSV bulunamadı, atlandı: {path}")

    if not histories:
        print("\nHiçbir CSV bulunamadı. Önce modelleri eğitin.")
        return

    print(f"\nGrafikler {PLOT_DIR} klasörüne kaydediliyor...")

    plot_val_dice_curves(histories)
    plot_best_dice_bar(histories)
    plot_all_metrics_bar(histories)

    # Özet tablo
    print("\n" + "─" * 55)
    print(f"  {'Model':<16} {'Best Val Dice':>14} {'Best Epoch':>11}")
    print("─" * 55)
    for name, rows in histories.items():
        br = best_row(rows)
        print(f"  {name:<16} {br['val_dice']:>14.4f} {int(br['epoch']):>11d}")
    print("─" * 55)
    print("\nTamamlandı.\n")


if __name__ == "__main__":
    main()
