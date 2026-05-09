# -*- coding: utf-8 -*-
"""
plot_eval_comparison_white.py
==============================
Mevcut per-image CSV'lerden evaluation_comparison.png'yi
BEYAZ ARKA PLAN ile yeniden uretir.

Kaynak CSV'ler: results/evaluation/<model>_test_results.csv
Cikti        : results/evaluation/evaluation_comparison_white.png
"""

import os, csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "results", "evaluation")

MODEL_ORDER = ["unet", "attention_unet", "resunet", "segformer", "swinunet"]
MODEL_LABELS = {
    "unet":           "U-Net",
    "attention_unet": "Attention U-Net",
    "resunet":        "ResUNet",
    "segformer":      "SegFormer-Lite",
    "swinunet":       "Swin-UNet",
}
MODEL_COLORS = {
    "unet":           "#2196F3",
    "attention_unet": "#FF5722",
    "resunet":        "#4CAF50",
    "segformer":      "#9C27B0",
    "swinunet":       "#FF9800",
}

METRIC_KEYS   = ["dice", "iou", "accuracy", "sensitivity", "specificity", "precision"]
METRIC_LABELS = {
    "dice":        "Dice Score",
    "iou":         "IoU (Jaccard)",
    "accuracy":    "Accuracy",
    "sensitivity": "Sensitivity (Recall)",
    "specificity": "Specificity",
    "precision":   "Precision",
}


def load_csv(model_name):
    path = os.path.join(EVAL_DIR, f"{model_name}_test_results.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items() if k != "sample"})
    return rows


def summarize(rows):
    summary = {}
    for k in METRIC_KEYS:
        vals = [r[k] for r in rows if k in r]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
        }
    return summary


def plot_comparison_white(all_summaries):
    models    = [m for m in MODEL_ORDER if m in all_summaries]
    n_models  = len(models)
    n_metrics = len(METRIC_KEYS)

    x     = np.arange(n_metrics)
    bar_w = 0.75 / n_models
    offsets = np.linspace(
        -(n_models - 1) * bar_w / 2,
         (n_models - 1) * bar_w / 2,
        n_models
    )

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="white")
    ax.set_facecolor("white")

    patches = []
    for i, m in enumerate(models):
        summary = all_summaries[m]
        means = [summary[k]["mean"] for k in METRIC_KEYS]
        stds  = [summary[k]["std"]  for k in METRIC_KEYS]
        color = MODEL_COLORS.get(m, "#888888")

        bars = ax.bar(
            x + offsets[i], means,
            width=bar_w * 0.88,
            color=color,
            alpha=0.87,
            zorder=3,
            edgecolor="white",
            linewidth=0.5,
            yerr=stds,
            capsize=3.5,
            error_kw={"ecolor": "#555555", "lw": 1.2, "capthick": 1.2},
        )

        # Deger etiketi — cubuk ustune
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{mean:.3f}",
                ha="center", va="bottom",
                fontsize=7, color="#222222",
                fontweight="bold",
            )

        patches.append(mpatches.Patch(color=color, label=MODEL_LABELS.get(m, m)))

    # Eksen ayarlari
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_LABELS[k] for k in METRIC_KEYS],
        rotation=18, ha="right", fontsize=10, color="#222222"
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", fontsize=12, color="#222222", labelpad=8)
    ax.set_title(
        "Test Set Evaluation — Model Karsilastirmasi",
        fontsize=14, fontweight="bold", color="#111111", pad=14
    )

    # Izgara
    ax.grid(True, axis="y", zorder=0, color="#e0e0e0", linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Kenarlıklar
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#aaaaaa")
        ax.spines[spine].set_linewidth(0.8)

    ax.tick_params(colors="#444444")

    # Lejant
    ax.legend(
        handles=patches,
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
    )

    fig.tight_layout()

    out = os.path.join(EVAL_DIR, "evaluation_comparison_white.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[OK] Kaydedildi: {out}")
    return out


def main():
    print("=" * 55)
    print("  Evaluation Comparison -- BEYAZ ARKA PLAN")
    print("=" * 55)

    all_summaries = {}
    for m in MODEL_ORDER:
        rows = load_csv(m)
        if rows:
            all_summaries[m] = summarize(rows)
            mean_dice = all_summaries[m]["dice"]["mean"]
            print(f"  {MODEL_LABELS.get(m, m):<18} yüklendi  |  mean Dice = {mean_dice:.4f}")
        else:
            print(f"  {MODEL_LABELS.get(m, m):<18} CSV bulunamadi -- atlandi")

    if not all_summaries:
        print("\n[HATA] Hicbir CSV bulunamadi.")
        sys.exit(1)

    print()
    plot_comparison_white(all_summaries)
    print("=" * 55)


if __name__ == "__main__":
    main()
