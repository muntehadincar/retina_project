"""
model_service.py — ResUNet inference servisi.
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

class ResidualBlock(nn.Module):
    """Artık bağlantılı çift konvolüsyon bloğu (in_ch ≠ out_ch olsa bile shortcut)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch))
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.shortcut(x))


class ResUNet(nn.Module):
    """
    U-Net encoder bloklarının yerine ResidualBlock kullanan mimari.
    Genel U-Net yapısı korunur; skip connection'lar aynen aktarılır.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 features: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.downs = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.downs.append(ResidualBlock(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.dec_convs.append(ResidualBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for up, dc, skip in zip(self.up_convs, self.dec_convs, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = dc(x)

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
    _model  = ResUNet().to(_device)
    state   = torch.load(MODEL_PATH, map_location=_device)
    _model.load_state_dict(state)
    _model.eval()
    print(f"[model_service] ResUNet yuklendi -> {_device} | {MODEL_PATH}")


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
    Görüntüyü işler, ResUNet ile çıkarım yapar.
    Döndürür:
        mask_base64      : base64 PNG maskesi
        vessel_pixel_count, vessel_area_ratio, vessel_density
    """
    tensor = _preprocess(image_bytes)
    logits = _model(tensor)
    prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()
    mask   = (prob > THRESHOLD).astype(np.uint8)

    total_px          = IMG_SIZE * IMG_SIZE
    vessel_px         = int(mask.sum())
    vessel_area_ratio = round(vessel_px / total_px, 4)

    return {
        "mask_base64":        _to_base64_png(mask),
        "vessel_pixel_count": vessel_px,
        "vessel_area_ratio":  vessel_area_ratio,
        "vessel_density":     vessel_area_ratio,
    }
