"""
model_service.py — Attention U-Net inference servisi.
Uygulama başlarken model yüklenir, predict() çağrıldığında çıkarım yapılır.
"""

import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from core.config import MODEL_PATH, IMG_SIZE, THRESHOLD

# ---------------------------------------------------------------------------
# Model tanımı (model.py'den kopyalandı — bağımlılıksız)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_x, F_int, 1, bias=False), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g_up  = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        alpha = self.psi(self.relu(self.W_g(g_up) + self.W_x(x)))
        return x * alpha


class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downs = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f)); ch = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.up_convs, self.att_gates, self.dec_convs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.att_gates.append(AttentionGate(F_g=f, F_x=f, F_int=f // 2))
            self.dec_convs.append(DoubleConv(f * 2, f))
        self.final = nn.Conv2d(features[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for up, ag, dc, skip in zip(self.up_convs, self.att_gates, self.dec_convs, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = dc(torch.cat([ag(g=x, x=skip), x], dim=1))
        return self.final(x)


# ---------------------------------------------------------------------------
# Servis
# ---------------------------------------------------------------------------

_model  = None
_device = None


def load_model():
    """Uygulama başlarken bir kez çağrılır."""
    global _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model  = AttentionUNet().to(_device)
    state   = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.eval()
    print(f"[model_service] Attention U-Net yuklendi -> {_device} | {MODEL_PATH}")


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    """Ham byte → (1, 3, H, W) float tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(_device)


def _to_base64_png(mask_np: np.ndarray) -> str:
    """(H, W) uint8 array → base64 PNG string."""
    pil_img = Image.fromarray(mask_np * 255)
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@torch.no_grad()
def predict(image_bytes: bytes) -> dict:
    """
    Görüntüyü işler, Attention U-Net ile çıkarım yapar.
    Döndürür:
        mask_base64      : base64 PNG maskesi
        vessel_pixel_count, vessel_area_ratio, vessel_density
    """
    tensor = _preprocess(image_bytes)
    logits = _model(tensor)
    prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask   = (prob > THRESHOLD).astype(np.uint8)

    total_px         = IMG_SIZE * IMG_SIZE
    vessel_px        = int(mask.sum())
    vessel_area_ratio = round(vessel_px / total_px, 4)

    return {
        "mask_base64":       _to_base64_png(mask),
        "vessel_pixel_count": vessel_px,
        "vessel_area_ratio":  vessel_area_ratio,
        "vessel_density":     vessel_area_ratio,
    }
