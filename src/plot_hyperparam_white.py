# -*- coding: utf-8 -*-
"""
plot_hyperparam_white.py
========================
Hiperparametre isi haritalari ve LR duyarlilik analiz tablolarini
BEYAZ arka plan ile uretir (tez / yayin kalitesi).

Ciktilar:
  results/plots/white/hyperparam_heatmap_<model>.png
  results/plots/white/hyperparam_sensitivity_lr.png
  results/plots/white/hyperparam_sensitivity_table.png
  results/logs/hyperparameter_summary_table_white.csv
"""

import os, csv, sys
import numpy as np
import matplotlib
# Windows terminal icin UTF-8 zorla
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import ticker

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR  = os.path.join(BASE_DIR, "results", "logs")
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots", "white")
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_NAMES  = ["unet", "attention_unet", "resunet", "segformer", "swinunet"]
MODEL_LABELS = {
    "unet": "U-Net",
    "attention_unet": "Attention U-Net",
    "resunet": "ResUNet",
    "segformer": "SegFormer-Lite",
    "swinunet": "Swin-UNet",
}
MODEL_COLORS = {
    "unet":           "#2196F3",
    "attention_unet": "#FF5722",
    "resunet":        "#4CAF50",
    "segformer":      "#9C27B0",
    "swinunet":       "#FF9800",
}

# Akademik beyaz stil
WHITE_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.labelcolor":   "#111111",
    "xtick.color":       "#333333",
    "ytick.color":       "#333333",
    "text.color":        "#111111",
    "grid.color":        "#cccccc",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.7,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}
plt.rcParams.update(WHITE_STYLE)


# ── Yardımcı: CSV yükle ────────────────────────────────────────
def load_search_csv(model_name):
    path = os.path.join(LOG_DIR, f"{model_name}_hyperparam_search.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "lr":          float(r["lr"]),
                "batch_size":  int(r["batch_size"]),
                "pos_weight":  float(r["pos_weight"]),
                "dice":        float(r["best_val_dice"]),
                "iou":         float(r["val_iou"]),
                "accuracy":    float(r.get("val_accuracy", 0)),
                "sensitivity": float(r.get("val_sensitivity", 0)),
                "specificity": float(r.get("val_specificity", 0)),
            })
    return rows


# ── 1. Isı Haritaları (LR × Batch, pos_weight paneli) ─────────
def plot_heatmaps(model_name, rows):
    pw_vals = sorted(set(r["pos_weight"] for r in rows))
    lr_vals = sorted(set(r["lr"] for r in rows), reverse=True)
    bs_vals = sorted(set(r["batch_size"] for r in rows))

    n_panels = len(pw_vals)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(4.5 * n_panels + 1.2, 4.2),
        facecolor="white"
    )
    if n_panels == 1:
        axes = [axes]

    # Özel renk haritası: beyaz → turuncu → koyu kırmızı
    cmap = LinearSegmentedColormap.from_list(
        "dice_cmap", ["#ffffff", "#fde68a", "#f59e0b", "#b45309", "#7c2d12"]
    )

    global_min = min(r["dice"] for r in rows)
    global_max = max(r["dice"] for r in rows)
    vmin = max(0.0, global_min - 0.01)
    vmax = min(1.0, global_max + 0.01)

    for ax, pw in zip(axes, pw_vals):
        ax.set_facecolor("white")
        grid = np.zeros((len(lr_vals), len(bs_vals)))
        for r in rows:
            if r["pos_weight"] == pw:
                i = lr_vals.index(r["lr"])
                j = bs_vals.index(r["batch_size"])
                grid[i, j] = r["dice"]

        im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Hücre değerleri
        for i in range(len(lr_vals)):
            for j in range(len(bs_vals)):
                val = grid[i, j]
                # Koyu hücrelerde beyaz, açık hücrelerde siyah yaz
                brightness = (val - vmin) / max(vmax - vmin, 1e-9)
                txt_color = "white" if brightness > 0.55 else "#111111"
                ax.text(j, i, f"{val:.4f}",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold",
                        color=txt_color)

        ax.set_xticks(range(len(bs_vals)))
        ax.set_xticklabels([str(b) for b in bs_vals], fontsize=10)
        ax.set_yticks(range(len(lr_vals)))
        ax.set_yticklabels([f"{lr:.0e}" for lr in lr_vals], fontsize=10)
        ax.set_xlabel("Batch Size", fontsize=11, labelpad=6)
        ax.set_ylabel("Learning Rate", fontsize=11, labelpad=6)
        ax.set_title(f"pos_weight = {pw:.1f}", fontsize=11,
                     fontweight="bold", color="#333333", pad=8)

        # İnce kenarlık
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.8)
            spine.set_visible(True)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=9, colors="#333333")
        cb.set_label("Dice Score", fontsize=9, color="#333333")

    label = MODEL_LABELS.get(model_name, model_name)
    fig.suptitle(f"{label} — Hiperparametre Isı Haritası (Dice Skoru)",
                 fontsize=13, fontweight="bold", color="#111111", y=1.03)
    fig.tight_layout()

    out = os.path.join(PLOT_DIR, f"hyperparam_heatmap_{model_name}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] Heatmap: {out}")


# ── 2. LR Duyarlılık Analizi (Grafik) ──────────────────────────
def plot_sensitivity(all_data):
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    # Tüm veri setindeki LR değerlerini al
    all_lr = sorted(set(r["lr"] for m in models for r in all_data[m]))

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    for m in models:
        rows = all_data[m]
        means, stds = [], []
        for lr in all_lr:
            dices = [r["dice"] for r in rows if r["lr"] == lr]
            if dices:
                means.append(np.mean(dices))
                stds.append(np.std(dices))
            else:
                means.append(np.nan)
                stds.append(0.0)

        means = np.array(means)
        stds  = np.array(stds)
        c = MODEL_COLORS.get(m, "#555555")
        x = np.arange(len(all_lr))

        ax.plot(x, means, marker="o", color=c,
                linewidth=2.2, markersize=7,
                label=MODEL_LABELS.get(m, m))
        ax.fill_between(x, means - stds, means + stds,
                        color=c, alpha=0.12)
        # Hata çubukları
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    ecolor=c, elinewidth=1.5, capsize=5, capthick=1.5)

    ax.set_xticks(range(len(all_lr)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in all_lr], fontsize=11)
    ax.set_xlabel("Learning Rate", fontsize=12, labelpad=8)
    ax.set_ylabel("Ortalama Doğrulama Dice Skoru", fontsize=12, labelpad=8)
    ax.set_title("Learning Rate Duyarlılık Analizi",
                 fontsize=13, fontweight="bold", color="#111111", pad=12)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
              fontsize=10, loc="best")
    ax.grid(True, axis="y", zorder=0, color="#dddddd", linestyle="--", linewidth=0.8)
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 0.02))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#555555")
        ax.spines[spine].set_linewidth(0.8)

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_sensitivity_lr.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] LR Duyarlilik Grafigi: {out}")


# ── 3. LR Duyarlılık Analizi — Tablo Görseli ──────────────────
def plot_sensitivity_table(all_data):
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    all_lr = sorted(set(r["lr"] for m in models for r in all_data[m]))

    # Tablo verisi hazırla
    col_headers = ["Model"] + [f"LR={lr:.0e}" for lr in all_lr] + ["En İyi LR"]
    table_data = []
    highlight_cols = []   # (satır_idx, sütun_idx) best değer pozisyonları

    for row_i, m in enumerate(models):
        rows = all_data[m]
        means = []
        for lr in all_lr:
            dices = [r["dice"] for r in rows if r["lr"] == lr]
            means.append(np.mean(dices) if dices else np.nan)

        best_idx = int(np.nanargmax(means))
        best_lr  = all_lr[best_idx]
        highlight_cols.append((row_i, best_idx + 1))  # +1 for "Model" col

        row_vals = [MODEL_LABELS.get(m, m)]
        for v in means:
            row_vals.append(f"{v:.4f}" if not np.isnan(v) else "—")
        row_vals.append(f"{best_lr:.0e}")
        table_data.append(row_vals)

    n_rows = len(table_data)
    n_cols = len(col_headers)

    fig_w = max(10, n_cols * 1.8)
    fig_h = n_rows * 0.65 + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 1.8)

    # Başlık satırı stili
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#cccccc")

    # Veri satırları — alternatif renk
    for i in range(n_rows):
        bg = "#f5f8ff" if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="#111111")
            cell.set_edgecolor("#d0d7e3")

    # En iyi değerleri vurgula
    for (ri, ci) in highlight_cols:
        cell = tbl[ri + 1, ci]
        cell.set_facecolor("#d4edda")
        cell.set_text_props(color="#155724", fontweight="bold")

    # Model adı sütunu kalın
    for i in range(n_rows):
        tbl[i + 1, 0].set_text_props(fontweight="bold", color="#1e3a5f")

    ax.set_title("Learning Rate Duyarlılık Analizi — Model Karşılaştırma Tablosu",
                 fontsize=12, fontweight="bold", color="#111111",
                 pad=14, loc="center")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_sensitivity_table.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] LR Duyarlilik Tablosu: {out}")


# ── 4. Özet Tablo — PNG ────────────────────────────────────────
def plot_summary_table(all_data):
    models = [m for m in MODEL_NAMES if m in all_data]
    if not models:
        return

    col_headers = ["Model", "En İyi LR", "Batch Size",
                   "Pos Weight", "Dice ↑", "IoU ↑", "# Kombinasyon"]
    table_data = []
    best_dice_idx = None
    best_dice_val = -1.0

    for ri, m in enumerate(models):
        rows = all_data[m]
        best = max(rows, key=lambda r: r["dice"])
        if best["dice"] > best_dice_val:
            best_dice_val = best["dice"]
            best_dice_idx = ri
        table_data.append([
            MODEL_LABELS.get(m, m),
            f"{best['lr']:.0e}",
            str(best["batch_size"]),
            f"{best['pos_weight']:.1f}",
            f"{best['dice']:.4f}",
            f"{best['iou']:.4f}",
            str(len(rows)),
        ])

    n_rows = len(table_data)
    n_cols = len(col_headers)

    fig, ax = plt.subplots(figsize=(12, n_rows * 0.65 + 1.4), facecolor="white")
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 1.9)

    # Başlık satırı
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#1e3a5f")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#c0c8d8")

    # Veri satırları
    for i in range(n_rows):
        bg = "#f5f8ff" if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(bg)
            cell.set_text_props(color="#111111")
            cell.set_edgecolor("#d0d7e3")

    # En iyi Dice satırını vurgula
    if best_dice_idx is not None:
        for j in range(n_cols):
            cell = tbl[best_dice_idx + 1, j]
            cell.set_facecolor("#fff3cd")
            cell.set_text_props(fontweight="bold", color="#856404")

    # Model sütunu kalın
    for i in range(n_rows):
        tbl[i + 1, 0].set_text_props(fontweight="bold", color="#1e3a5f")

    ax.set_title("Hiperparametre Arama Özeti — En İyi Kombinasyonlar",
                 fontsize=13, fontweight="bold", color="#111111",
                 pad=14, loc="center")

    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_best_summary_table.png")
    fig.savefig(out, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  [OK] Ozet Tablo: {out}")


# ── Main ───────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Hiperparametre Görselleştirme — BEYAZ ARKA PLAN")
    print("=" * 62)

    all_data = {}
    for m in MODEL_NAMES:
        rows = load_search_csv(m)
        if rows:
            all_data[m] = rows
            print(f"  {m}: {len(rows)} kombinasyon yüklendi")
        else:
            print(f"  {m}: CSV bulunamadı — atlandı")

    if not all_data:
        print("\n[HATA] Hiçbir model için arama sonucu bulunamadı.")
        print("  Önce çalıştır: python src/hyperparameter_search.py")
        return

    print()
    for m, rows in all_data.items():
        print(f"  >> {MODEL_LABELS.get(m, m)} isi haritasi olusturuluyor...")
        plot_heatmaps(m, rows)

    print()
    print("  >> LR duyarlilik grafigi...")
    plot_sensitivity(all_data)

    print()
    print("  >> LR duyarlilik tablosu...")
    plot_sensitivity_table(all_data)

    print()
    print("  >> Ozet tablo...")
    plot_summary_table(all_data)

    print(f"\n{'=' * 62}")
    print("  TAMAMLANDI")
    print(f"  Çıktılar: {PLOT_DIR}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
