# -*- coding: utf-8 -*-
"""
visualize.py
============
İki görev:
  1. Eğitim eğrilerini çiz (train/val loss & dice) — results/logs/*_history.csv'den
  2. Test seti görsel karşılaştırması (4 sütun):
       Input  |  Ground Truth  |  UNet Pred  |  AttentionUNet Pred

Çıktılar:
  results/plots/unet_training_curves.png
  results/plots/attention_unet_training_curves.png          (varsa)
  results/preds/comparison/sample_XX.png    ← her test görüntüsü için
  results/preds/comparison/strip_all.png    ← tüm örnekler tek şerit

Kullanım:
    $env:PYTHONUTF8=1; python src/visualize.py
"""

import os
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from train import DriveVesselDataset, get_id
from model import UNet, AttentionUNet

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATHS = {
    "unet":           os.path.join(BASE_DIR, "results", "models", "unet_best.pth"),
    "attention_unet": os.path.join(BASE_DIR, "results", "models", "attention_unet_best.pth"),
}

HISTORY_PATHS = {
    "unet":           os.path.join(BASE_DIR, "results", "logs", "unet_history.csv"),
    "attention_unet": os.path.join(BASE_DIR, "results", "logs", "attention_unet_history.csv"),
}

TEST_IMG_DIR  = os.path.join(BASE_DIR, "data", "test_im_png")
TEST_MASK_DIR = os.path.join(BASE_DIR, "data", "test_mask_png")

PRED_THRESHOLD = 0.3
IMG_SIZE       = 256
MAX_SAMPLES    = 8

PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")
CMP_DIR  = os.path.join(BASE_DIR, "results", "preds", "comparison")


# ---------------------------------------------------------------------------
# Bölüm 1 — Eğitim eğrileri
# ---------------------------------------------------------------------------
def plot_training_curves(history_csv, model_name, out_dir):
    if not os.path.exists(history_csv):
        print(f"[UYARI] Eğitim geçmişi bulunamadı: {history_csv} — atlanıyor.")
        return

    epochs, tr_loss, va_loss, tr_dice, va_dice = [], [], [], [], []
    with open(history_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tr_loss.append(float(row["train_loss"]))
            va_loss.append(float(row["val_loss"]))
            tr_dice.append(float(row["train_dice"]))
            va_dice.append(float(row["val_dice"]))

    if not epochs:
        print("[UYARI] CSV boş — atlanıyor.")
        return

    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#3a3d4d",
        "axes.labelcolor":  "#e0e0e0",
        "xtick.color":      "#a0a0b0",
        "ytick.color":      "#a0a0b0",
        "text.color":       "#e0e0e0",
        "grid.color":       "#2a2d3d",
        "grid.linestyle":   "--",
        "grid.linewidth":   0.6,
        "legend.facecolor": "#1a1d27",
        "legend.edgecolor": "#3a3d4d",
        "font.family":      "DejaVu Sans",
    })

    C_TR   = "#4fc3f7"
    C_VA   = "#ff7043"
    C_BEST = "#ffd54f"

    fig = plt.figure(figsize=(14, 9), facecolor="#0f1117")
    fig.suptitle(f"{model_name.replace('_', ' ').title()}  —  Eğitim Eğrileri",
                 fontsize=16, fontweight="bold", color="#ffffff", y=0.97)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    # Loss
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, tr_loss, color=C_TR, lw=2.2, label="Train Loss", zorder=3)
    ax1.plot(epochs, va_loss, color=C_VA, lw=2.2, label="Val Loss", zorder=3, linestyle="--")
    bv_ep = epochs[int(np.argmin(va_loss))]
    bv_lo = min(va_loss)
    ax1.axvline(x=bv_ep, color=C_BEST, lw=1.2, linestyle=":", alpha=0.8,
                label=f"En İyi Val (epoch {bv_ep})")
    ax1.scatter([bv_ep], [bv_lo], color=C_BEST, s=80, zorder=5)
    ax1.set_title("Loss", fontsize=12, pad=6)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(framealpha=0.9, fontsize=9); ax1.grid(True)

    # Dice
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, tr_dice, color=C_TR, lw=2.2, label="Train Dice", zorder=3)
    ax2.plot(epochs, va_dice, color=C_VA, lw=2.2, label="Val Dice", zorder=3, linestyle="--")
    bd_ep = epochs[int(np.argmax(va_dice))]
    bd_va = max(va_dice)
    ax2.axvline(x=bd_ep, color=C_BEST, lw=1.2, linestyle=":", alpha=0.8,
                label=f"En İyi Val (epoch {bd_ep})")
    ax2.scatter([bd_ep], [bd_va], color=C_BEST, s=80, zorder=5)
    ax2.set_title("Dice Score", fontsize=12, pad=6)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
    ax2.set_ylim(0, 1.0)
    ax2.legend(framealpha=0.9, fontsize=9); ax2.grid(True)

    out_path = os.path.join(out_dir, f"{model_name}_training_curves.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PLOT] Eğitim eğrileri → {out_path}")


# ---------------------------------------------------------------------------
# Yardımcı: model yükle
# ---------------------------------------------------------------------------
def load_model(name, device):
    path = MODEL_PATHS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model ağırlığı bulunamadı: {path}")
    cls = UNet if name == "unet" else AttentionUNet
    m = cls().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    print(f"  Yüklendi: {path}")
    return m


# ---------------------------------------------------------------------------
# Bölüm 2 — 4 sütunlu karşılaştırma görseli
#   Sütunlar: Orijinal | Ground Truth | UNet Pred | AttUNet Pred
# ---------------------------------------------------------------------------
def visualize_comparison(unet, att_unet, device, out_dir, max_samples=MAX_SAMPLES):
    if not os.path.isdir(TEST_IMG_DIR) or not os.listdir(TEST_IMG_DIR):
        print(f"[UYARI] Test klasörü boş: {TEST_IMG_DIR}")
        return

    test_img_files  = {get_id(f): f for f in os.listdir(TEST_IMG_DIR)
                       if f.lower().endswith(".png")}
    test_mask_files = {get_id(f): f for f in os.listdir(TEST_MASK_DIR)
                       if f.lower().endswith(".png")}
    common_ids = sorted(set(test_img_files) & set(test_mask_files))

    if not common_ids:
        print("[UYARI] Test klasöründe eşleşen görüntü+maske çifti yok.")
        return

    ds = DriveVesselDataset(TEST_IMG_DIR, TEST_MASK_DIR,
                            ids=common_ids, augment=False, size=IMG_SIZE)
    os.makedirs(out_dir, exist_ok=True)

    n = min(max_samples, len(ds))
    print(f"\n[CMP] {n} test örneği işleniyor...")

    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#0f1117",
        "text.color":       "#e0e0e0",
    })

    COLS     = ["Orijinal Görüntü", "Ground Truth", "UNet Tahmin", "AttentionUNet Tahmin"]
    CLR_UNET = "#4fc3f7"   # açık mavi başlık rengi
    CLR_ATT  = "#ff7043"   # turuncu başlık rengi

    all_panels = []   # strip için

    with torch.no_grad():
        for i in range(n):
            x, y = ds[i]
            x_gpu = x.unsqueeze(0).to(device)

            # UNet tahmini
            u_logits  = unet(x_gpu)
            u_prob    = torch.sigmoid(u_logits)[0, 0].cpu().numpy()
            u_pred    = (u_prob > PRED_THRESHOLD).astype(np.uint8) * 255

            # AttentionUNet tahmini
            a_logits  = att_unet(x_gpu)
            a_prob    = torch.sigmoid(a_logits)[0, 0].cpu().numpy()
            a_pred    = (a_prob > PRED_THRESHOLD).astype(np.uint8) * 255

            img_np = (x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_np  = (y[0].numpy() * 255).astype(np.uint8)

            all_panels.append((img_np, gt_np, u_pred, a_pred))

            # ── Bireysel PNG ─────────────────────────────────────────────────
            fig, axes = plt.subplots(1, 4, figsize=(20, 5),
                                     facecolor="#0f1117",
                                     gridspec_kw={"wspace": 0.04})
            fig.suptitle(f"Test Örneği #{i+1:02d}  |  threshold = {PRED_THRESHOLD}",
                         color="#ffffff", fontsize=13, fontweight="bold", y=1.01)

            data_cmap = [
                (img_np, None),
                (gt_np,  "gray"),
                (u_pred, "gray"),
                (a_pred, "gray"),
            ]
            title_colors = ["#e0e0e0", "#e0e0e0", CLR_UNET, CLR_ATT]

            for ax, (data, cmap), title, tc in zip(axes, data_cmap, COLS, title_colors):
                ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
                ax.set_title(title, color=tc, fontsize=9, fontweight="bold", pad=5)
                ax.axis("off")

            out_path = os.path.join(out_dir, f"sample_{i+1:02d}.png")
            fig.savefig(out_path, dpi=140, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Kaydedildi: {out_path}")

    # ── Toplu şerit ──────────────────────────────────────────────────────────
    print("\n[CMP] Toplu şerit oluşturuluyor...")
    rows = 4   # Orijinal / GT / UNet / AttUNet
    fig, axes = plt.subplots(rows, n,
                             figsize=(3.5 * n, 14),
                             facecolor="#0f1117",
                             gridspec_kw={"hspace": 0.04, "wspace": 0.03})

    row_labels  = ["Orijinal", "Ground Truth", "UNet", "AttentionUNet"]
    row_colors  = ["#e0e0e0",  "#e0e0e0",      CLR_UNET, CLR_ATT]

    for row in range(rows):
        for col in range(n):
            ax   = axes[row][col] if n > 1 else axes[row]
            data = all_panels[col][row]
            cmap = None if row == 0 else "gray"
            ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], color=row_colors[row],
                              fontsize=10, fontweight="bold",
                              rotation=90, labelpad=8)

    fig.suptitle(f"Test Seti — Model Karşılaştırması  (thr={PRED_THRESHOLD})",
                 color="#ffffff", fontsize=14, fontweight="bold", y=1.002)

    strip_path = os.path.join(out_dir, "strip_all.png")
    fig.savefig(strip_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[CMP] Toplu şerit → {strip_path}")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  VISUALIZE.PY  —  4-Sütun Model Karşılaştırması")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. Eğitim eğrileri ──────────────────────────────────────────────────
    print("--- Bölüm 1: Eğitim Eğrileri ---")
    for name in ("unet", "attention_unet"):
        plot_training_curves(HISTORY_PATHS[name], name, PLOT_DIR)

    # ── 2. Model yükle ──────────────────────────────────────────────────────
    print("\n--- Bölüm 2: Modeller Yükleniyor ---")
    try:
        unet     = load_model("unet",           device)
        att_unet = load_model("attention_unet", device)
    except FileNotFoundError as e:
        print(f"[HATA] {e}")
        print("  → Önce train_compare.py ile her iki modeli eğitin.")
        return

    # ── 3. 4 sütunlu karşılaştırma görseli ──────────────────────────────────
    print("\n--- Bölüm 3: 4-Sütun Karşılaştırma Görseli ---")
    visualize_comparison(unet, att_unet, device, out_dir=CMP_DIR,
                         max_samples=MAX_SAMPLES)

    print("\n" + "=" * 65)
    print("  ✅ Tamamlandı!")
    print(f"     Grafikler    → {PLOT_DIR}")
    print(f"     Karşılaştır. → {CMP_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()