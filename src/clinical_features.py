# -*- coding: utf-8 -*-
"""
clinical_features.py
====================
Her test görüntüsü için klinik damar özellikleri çıkarır:

  - vessel_pixel_count  : binary maskede beyaz (damar) piksel sayısı
  - vessel_area_ratio   : damar pikseli / toplam piksel
  - vessel_density      : vessel_area_ratio ile aynı (alan yoğunluğu)

Her iki model (UNet, AttentionUNet) için ayrı CSV üretir;
ayrıca iki modeli yan yana gösteren bir karşılaştırma CSV'si kaydeder.

Çıktılar:
  results/clinical/unet_clinical.csv
  results/clinical/attention_unet_clinical.csv
  results/clinical/clinical_comparison.csv

Kullanım:
  $env:PYTHONUTF8=1; python src/clinical_features.py
"""

import os
import csv
import numpy as np
import torch

from train import DriveVesselDataset, get_id
from model import UNet, AttentionUNet

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(__file__))
TEST_IMG_DIR  = os.path.join(BASE_DIR, "data", "test_im_png")
TEST_MASK_DIR = os.path.join(BASE_DIR, "data", "test_mask_png")
OUT_DIR       = os.path.join(BASE_DIR, "results", "clinical")

IMG_SIZE  = 256
THR       = 0.3      # sigmoid eşik değeri (evaluate.py ile aynı)
TOTAL_PIX = IMG_SIZE * IMG_SIZE   # 65 536

MODELS = {
    "unet": {
        "class": UNet,
        "path":  os.path.join(BASE_DIR, "results", "models", "unet_best.pth"),
    },
    "attention_unet": {
        "class": AttentionUNet,
        "path":  os.path.join(BASE_DIR, "results", "models", "attention_unet_best.pth"),
    },
}

FIELDS = ["image_name", "vessel_pixel_count", "vessel_area_ratio", "vessel_density"]


# ---------------------------------------------------------------------------
# Yardımcı
# ---------------------------------------------------------------------------
def load_model(cfg, device):
    path = cfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model bulunamadi: {path}")
    m = cfg["class"]().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m


def extract_features(model, ds, img_names, device):
    """Her görüntü için klinik özellikleri hesapla. Liste döner."""
    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            logits = model(x.unsqueeze(0).to(device))
            prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred   = (prob > THR).astype(np.uint8)   # binary (0 / 1)

            vessel_px    = int(pred.sum())
            area_ratio   = vessel_px / TOTAL_PIX
            vessel_dens  = area_ratio          # aynı formül, ayrı sütun

            rows.append({
                "image_name":        img_names[i],
                "vessel_pixel_count": vessel_px,
                "vessel_area_ratio":  round(area_ratio,  4),
                "vessel_density":     round(vessel_dens, 4),
            })
    return rows


def save_csv(rows, path, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Kaydedildi: {path}")


def print_table(model_name, rows):
    col = max(len(r["image_name"]) for r in rows)
    col = max(col, 12)
    hdr = f"  {'image_name':<{col}}  {'vessel_pixel_count':>18}  {'vessel_area_ratio':>17}  {'vessel_density':>14}"
    print(f"\n--- {model_name.upper()} ---")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['image_name']:<{col}}  {r['vessel_pixel_count']:>18}  "
              f"{r['vessel_area_ratio']:>17.4f}  {r['vessel_density']:>14.4f}")


# ---------------------------------------------------------------------------
# Ana akış
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  CLINICAL_FEATURES.PY  —  Klinik Özellik Çıkarma")
    print("=" * 60)
    print(f"  Eşik (threshold) : {THR}")
    print(f"  Görüntü boyutu   : {IMG_SIZE}x{IMG_SIZE}  ({TOTAL_PIX} piksel)\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Test görüntülerini bul
    if not os.path.isdir(TEST_IMG_DIR) or not os.listdir(TEST_IMG_DIR):
        print(f"[HATA] Test klasörü boş veya bulunamadı: {TEST_IMG_DIR}")
        return

    img_files  = {get_id(f): f for f in os.listdir(TEST_IMG_DIR)
                  if f.lower().endswith(".png")}
    mask_files = {get_id(f): f for f in os.listdir(TEST_MASK_DIR)
                  if f.lower().endswith(".png")}
    test_ids   = sorted(set(img_files) & set(mask_files))

    if not test_ids:
        print("[HATA] Eşleşen test görüntüsü bulunamadı.")
        return

    print(f"Test örnek sayısı: {len(test_ids)}\n")

    ds         = DriveVesselDataset(TEST_IMG_DIR, TEST_MASK_DIR,
                                    ids=test_ids, augment=False, size=IMG_SIZE)
    img_names  = [img_files[tid] for tid in test_ids]   # gerçek dosya adları

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}   # model_name → rows

    # Her model için özellik çıkar
    for model_name, cfg in MODELS.items():
        print(f"[{model_name.upper()}] Tahminler hesaplanıyor...")
        try:
            model = load_model(cfg, device)
        except FileNotFoundError as e:
            print(f"  [ATLANDI] {e}")
            continue

        rows = extract_features(model, ds, img_names, device)
        all_results[model_name] = rows

        # Konsola tablo
        print_table(model_name, rows)

        # Per-model CSV
        csv_path = os.path.join(OUT_DIR, f"{model_name}_clinical.csv")
        save_csv(rows, csv_path, FIELDS)
        print()

    # Karşılaştırma CSV (iki model yan yana)
    if len(all_results) == 2:
        u_rows = all_results["unet"]
        a_rows = all_results["attention_unet"]

        cmp_fields = (
            ["image_name"]
            + [f"unet_{f}"    for f in FIELDS[1:]]
            + [f"att_{f}"     for f in FIELDS[1:]]
        )
        cmp_rows = []
        for u, a in zip(u_rows, a_rows):
            cmp_rows.append({
                "image_name":             u["image_name"],
                "unet_vessel_pixel_count": u["vessel_pixel_count"],
                "unet_vessel_area_ratio":  u["vessel_area_ratio"],
                "unet_vessel_density":     u["vessel_density"],
                "att_vessel_pixel_count":  a["vessel_pixel_count"],
                "att_vessel_area_ratio":   a["vessel_area_ratio"],
                "att_vessel_density":      a["vessel_density"],
            })

        cmp_path = os.path.join(OUT_DIR, "clinical_comparison.csv")
        save_csv(cmp_rows, cmp_path, cmp_fields)

        # Özet istatistikler
        print("\n[ÖZET] Ortalama Klinik Özellikler")
        print(f"  {'Model':<20}  {'Ort. Damar Piksel':>18}  {'Ort. Alan Oranı':>16}")
        print("  " + "-" * 58)
        for model_name, rows in all_results.items():
            avg_px    = np.mean([r["vessel_pixel_count"] for r in rows])
            avg_ratio = np.mean([r["vessel_area_ratio"]  for r in rows])
            print(f"  {model_name:<20}  {avg_px:>18.1f}  {avg_ratio:>16.4f}")

    print("\n" + "=" * 60)
    print(f"  ✅ Tamamlandı!  →  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
