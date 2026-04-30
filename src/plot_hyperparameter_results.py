"""
plot_hyperparameter_results.py
==============================
Hiperparametre arama sonuclarini gorsellestir ve LaTeX tablosu uret.

Ciktilar:
  results/plots/hyperparam_heatmap_<model>.png    — LR x Batch x PosW isi haritasi
  results/plots/hyperparam_best_comparison.png    — En iyi kombinasyon karsilastirmasi
  results/plots/hyperparam_sensitivity_lr.png     — LR duyarlilik analizi
  results/logs/hyperparameter_summary_table.tex   — LaTeX tablosu
  results/logs/hyperparameter_summary_table.csv   — CSV tablosu

Kullanim:
    python src/plot_hyperparameter_results.py
"""

import os, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR  = os.path.dirname(os.path.dirname(__file__))
LOG_DIR   = os.path.join(BASE_DIR, "results", "logs")
PLOT_DIR  = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_NAMES  = ["unet", "attention_unet", "resunet", "segformer", "swinunet"]
MODEL_LABELS = {
    "unet": "U-Net", "attention_unet": "Attention U-Net",
    "resunet": "ResUNet", "segformer": "SegFormer-Lite",
    "swinunet": "Swin-UNet",
}
MODEL_COLORS = {
    "unet": "#4fc3f7", "attention_unet": "#ff7043",
    "resunet": "#57B894", "segformer": "#A066CC",
    "swinunet": "#E8B84B",
}

DARK_STYLE = {
    "figure.facecolor": "#0f1117", "axes.facecolor": "#1a1d27",
    "axes.edgecolor": "#3a3d4d", "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#cccccc", "ytick.color": "#cccccc",
    "text.color": "#e0e0e0", "grid.color": "#2a2d3d",
    "grid.linestyle": "--", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
}


def load_search_csv(model_name):
    path = os.path.join(LOG_DIR, f"{model_name}_hyperparam_search.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "lr": float(r["lr"]),
                "batch_size": int(r["batch_size"]),
                "pos_weight": float(r["pos_weight"]),
                "dice": float(r["best_val_dice"]),
                "iou": float(r["val_iou"]),
                "accuracy": float(r.get("val_accuracy", 0)),
                "sensitivity": float(r.get("val_sensitivity", 0)),
                "specificity": float(r.get("val_specificity", 0)),
                "elapsed": float(r.get("elapsed_sec", 0)),
            })
    return rows


# ─── 1. Heatmap: her pos_weight icin LR x Batch ──────────────

def plot_heatmaps(model_name, rows):
    pw_vals = sorted(set(r["pos_weight"] for r in rows))
    lr_vals = sorted(set(r["lr"] for r in rows), reverse=True)
    bs_vals = sorted(set(r["batch_size"] for r in rows))

    fig, axes = plt.subplots(1, len(pw_vals), figsize=(5*len(pw_vals), 4),
                             facecolor="#0f1117")
    if len(pw_vals) == 1:
        axes = [axes]

    for ax, pw in zip(axes, pw_vals):
        ax.set_facecolor("#1a1d27")
        grid = np.zeros((len(lr_vals), len(bs_vals)))
        for r in rows:
            if r["pos_weight"] == pw:
                i = lr_vals.index(r["lr"])
                j = bs_vals.index(r["batch_size"])
                grid[i, j] = r["dice"]

        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto",
                       vmin=max(0, grid.min()-0.05), vmax=min(1, grid.max()+0.02))
        for i in range(len(lr_vals)):
            for j in range(len(bs_vals)):
                ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color="black" if grid[i,j] > grid.mean() else "white")

        ax.set_xticks(range(len(bs_vals)))
        ax.set_xticklabels([str(b) for b in bs_vals], color="#ccc")
        ax.set_yticks(range(len(lr_vals)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lr_vals], color="#ccc")
        ax.set_xlabel("Batch Size", color="#e0e0e0")
        ax.set_ylabel("Learning Rate", color="#e0e0e0")
        ax.set_title(f"pos_weight = {pw}", color="#fff", fontsize=11, pad=8)
        ax.tick_params(colors="#ccc")

    fig.suptitle(f"{MODEL_LABELS.get(model_name, model_name)} — Hiperparametre Isı Haritası (Dice)",
                 fontsize=13, fontweight="bold", color="#fff", y=1.02)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f"hyperparam_heatmap_{model_name}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Heatmap: {out}")


# ─── 2. En iyi kombinasyon karsilastirmasi ────────────────────

def plot_best_comparison(all_data):
    plt.rcParams.update(DARK_STYLE)
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    best_dices = []
    best_labels = []
    colors = []
    for m in models:
        rows = all_data[m]
        best = max(rows, key=lambda r: r["dice"])
        best_dices.append(best["dice"])
        best_labels.append(
            f"LR={best['lr']:.0e}\nBS={best['batch_size']}\nPW={best['pos_weight']}"
        )
        colors.append(MODEL_COLORS.get(m, "#aaa"))

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    x = np.arange(len(models))
    bars = ax.bar(x, best_dices, color=colors, alpha=0.85, width=0.55, zorder=3)

    for bar, dice, lbl in zip(bars, best_dices, best_labels):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{dice:.4f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#fff")
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                lbl, ha="center", va="center", fontsize=8, color="#000",
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
    ax.set_ylabel("Best Validation Dice", fontsize=11)
    ax.set_ylim(0, max(best_dices) + 0.08)
    ax.set_title("Hiperparametre Araması — En İyi Kombinasyonlar",
                 fontsize=13, fontweight="bold", color="#fff", pad=10)
    ax.grid(True, axis="y", zorder=0)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_best_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Karsilastirma: {out}")


# ─── 3. LR duyarlilik analizi ─────────────────────────────────

def plot_sensitivity(all_data):
    plt.rcParams.update(DARK_STYLE)
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")

    for m in models:
        rows = all_data[m]
        lr_vals = sorted(set(r["lr"] for r in rows))
        means = []
        stds = []
        for lr in lr_vals:
            dices = [r["dice"] for r in rows if r["lr"] == lr]
            means.append(np.mean(dices))
            stds.append(np.std(dices))
        c = MODEL_COLORS.get(m, "#aaa")
        ax.errorbar(range(len(lr_vals)), means, yerr=stds,
                    marker="o", label=MODEL_LABELS.get(m, m),
                    color=c, capsize=4, linewidth=2, markersize=6)

    ax.set_xticks(range(len(lr_vals)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in sorted(set(r["lr"] for r in rows))],
                       fontsize=10)
    ax.set_xlabel("Learning Rate", fontsize=11)
    ax.set_ylabel("Mean Validation Dice", fontsize=11)
    ax.set_title("Learning Rate Duyarlılık Analizi",
                 fontsize=13, fontweight="bold", color="#fff", pad=10)
    ax.legend(facecolor="#1a1d27", edgecolor="#3a3d4d", fontsize=9)
    ax.grid(True, axis="y", zorder=0)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_sensitivity_lr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Duyarlilik: {out}")


# ─── 4. LaTeX + CSV ozet tablosu ──────────────────────────────

def generate_tables(all_data):
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    # ── 4a. Tum kombinasyonlar LaTeX (her model icin ayri tablo)
    full_tex_path = os.path.join(LOG_DIR, "hyperparameter_full_results.tex")
    with open(full_tex_path, "w", encoding="utf-8") as f:
        for m in models:
            rows = all_data[m]
            f.write(f"% {MODEL_LABELS.get(m, m)}\n")
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write(f"\\caption{{{MODEL_LABELS.get(m,m)} Hiperparametre Arama Sonuçları}}\n")
            f.write(f"\\label{{tab:hyperparam_{m}}}\n")
            f.write("\\begin{tabular}{cccccccc}\n\\hline\n")
            f.write("\\textbf{\\#} & \\textbf{LR} & \\textbf{Batch} & "
                    "\\textbf{Pos Weight} & \\textbf{Dice} & \\textbf{IoU} & "
                    "\\textbf{Acc} & \\textbf{Sens} \\\\\n\\hline\n")
            for i, r in enumerate(rows, 1):
                best_r = max(rows, key=lambda x: x["dice"])
                bold = r["dice"] == best_r["dice"]
                dice_str = f"\\textbf{{{r['dice']:.4f}}}" if bold else f"{r['dice']:.4f}"
                f.write(f"{i} & {r['lr']:.0e} & {r['batch_size']} & "
                        f"{r['pos_weight']:.1f} & {dice_str} & "
                        f"{r['iou']:.4f} & {r['accuracy']:.4f} & "
                        f"{r['sensitivity']:.4f} \\\\\n")
            f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    print(f"  LaTeX (detay): {full_tex_path}")

    # ── 4b. Ozet tablosu — sadece en iyi parametreler
    summary_tex = os.path.join(LOG_DIR, "hyperparameter_summary_table.tex")
    summary_csv = os.path.join(LOG_DIR, "hyperparameter_summary_table.csv")

    csv_fields = ["Model", "Best LR", "Best Batch", "Best PosWeight",
                  "Best Dice", "Best IoU", "Combinations"]

    csv_rows = []
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Hiperparametre Arama Sonuçları — En İyi Kombinasyonlar}\n")
        f.write("\\label{tab:hyperparam_summary}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("\\textbf{Model} & \\textbf{LR} & \\textbf{Batch} & "
                "\\textbf{Pos Wt.} & \\textbf{Dice} & \\textbf{IoU} & "
                "\\textbf{\\# Komb.} \\\\\n\\hline\n")

        for m in models:
            rows = all_data[m]
            best = max(rows, key=lambda r: r["dice"])
            label = MODEL_LABELS.get(m, m)
            f.write(f"{label} & {best['lr']:.0e} & {best['batch_size']} & "
                    f"{best['pos_weight']:.1f} & {best['dice']:.4f} & "
                    f"{best['iou']:.4f} & {len(rows)} \\\\\n")
            csv_rows.append({
                "Model": label, "Best LR": best["lr"],
                "Best Batch": best["batch_size"],
                "Best PosWeight": best["pos_weight"],
                "Best Dice": best["dice"], "Best IoU": best["iou"],
                "Combinations": len(rows),
            })

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX (ozet): {summary_tex}")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  CSV  (ozet): {summary_csv}")


# ─── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Hiperparametre Sonuc Gorsellestirme")
    print("=" * 60)

    all_data = {}
    for m in MODEL_NAMES:
        rows = load_search_csv(m)
        if rows:
            all_data[m] = rows
            print(f"  {m}: {len(rows)} kombinasyon yuklendi")
        else:
            print(f"  {m}: CSV bulunamadi — atlandi")

    if not all_data:
        print("\n[HATA] Hicbir model icin arama sonucu bulunamadi.")
        print("  Once calistir: python src/hyperparameter_search.py")
        return

    print()
    for m in all_data:
        plot_heatmaps(m, all_data[m])

    plot_best_comparison(all_data)
    plot_sensitivity(all_data)
    generate_tables(all_data)

    print(f"\n{'='*60}")
    print("  TAMAMLANDI")
    print(f"  Grafikler : {PLOT_DIR}")
    print(f"  Tablolar  : {LOG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
