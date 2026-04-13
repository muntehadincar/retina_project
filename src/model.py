import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)


# ---------------------------------------------------------------------------
# Attention Gate
# ---------------------------------------------------------------------------

# Attention Gate: decoder'dan gelen g sinyali ile encoder skip feature'ı x'i karşılaştırıp
# her piksel için 0-1 arası bir ağırlık (alpha) üretir, sonra x * alpha döner.
class AttentionGate(nn.Module):

    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # g'yi x'in uzamsal boyutuna getir
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1   = self.W_g(g_up)
        x1   = self.W_x(x)
        psi  = self.relu(g1 + x1)
        alpha = self.psi(psi)          # dikkat maskesi [0,1]
        return x * alpha               # ağırlıklı skip feature


# ---------------------------------------------------------------------------
# Attention U-Net
# ---------------------------------------------------------------------------

# Attention U-Net: standart U-Net'ten farkı decoder kısmında
# skip connection'lar doğrudan gelmez, önce AttentionGate'ten geçer.
class AttentionUNet(nn.Module):

    def __init__(self, in_ch: int = 3, out_ch: int = 1,
                 features: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.downs = nn.ModuleList()
        ch = in_ch
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: ConvTranspose + AttentionGate + DoubleConv
        self.up_convs   = nn.ModuleList()
        self.att_gates  = nn.ModuleList()
        self.dec_convs  = nn.ModuleList()

        for f in reversed(features):
            self.up_convs.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.att_gates.append(
                AttentionGate(F_g=f, F_x=f, F_int=f // 2)
            )
            self.dec_convs.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        # Decoder
        for up, ag, dc, skip in zip(
                self.up_convs, self.att_gates, self.dec_convs, skips):
            x    = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:],
                                  mode="bilinear", align_corners=False)
            skip = ag(g=x, x=skip)     # attention-gated skip
            x    = torch.cat([skip, x], dim=1)
            x    = dc(x)

        return self.final(x)


# ─────────────────────────────────────────────────────────────────────────────
# ResUNet
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# SegFormer-Lite   (gereksinim: pip install timm>=0.9)
# ─────────────────────────────────────────────────────────────────────────────

class _DWSConv(nn.Module):
    """Depthwise Separable Conv + BN + ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class _MixTransformerBlock(nn.Module):
    """
    SegFormer Mix Transformer bloğunu taklit eden hafif blok:
    Depthwise Conv (lokal dikkat yerine) + Channel MLP (FFN).
    Saf PyTorch — dış bağımlılık yok.
    """
    def __init__(self, ch, expand=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(ch)
        self.dw    = _DWSConv(ch, ch)
        self.norm2 = nn.BatchNorm2d(ch)
        self.ffn   = nn.Sequential(
            nn.Conv2d(ch, ch * expand, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(ch * expand, ch, 1, bias=False),
        )

    def forward(self, x):
        x = x + self.dw(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SegFormerLite(nn.Module):
    """
    SegFormer'dan ilham alan hafif segmentasyon modeli.
    MiT-B0 yerine Depthwise Separable Conv + Mix Transformer blok
    kullanır — saf PyTorch, dış bağımlılık yok.

    Referans: Xie et al., 'SegFormer: Simple and Efficient Design for
              Semantic Segmentation with Transformers' (NeurIPS 2021).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Encoder: 4 aşama, her biri stride-2 ile aşağı örnekler
        # Çıkış boyutları: H/4, H/8, H/16, H/32
        chs = [32, 64, 160, 256]

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, chs[0], 7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(chs[0]), nn.ReLU(inplace=True),
            _MixTransformerBlock(chs[0]),
        )
        self.stage2 = nn.Sequential(
            _DWSConv(chs[0], chs[1], stride=2),
            _MixTransformerBlock(chs[1]),
        )
        self.stage3 = nn.Sequential(
            _DWSConv(chs[1], chs[2], stride=2),
            _MixTransformerBlock(chs[2]),
        )
        self.stage4 = nn.Sequential(
            _DWSConv(chs[2], chs[3], stride=2),
            _MixTransformerBlock(chs[3]),
        )

        # Decoder: her aşama 1/4 çözünürlüğe getirilip birleştiriliyor
        embed = 256
        self.projs = nn.ModuleList([nn.Conv2d(c, embed, 1) for c in chs])
        self.fuse  = nn.Sequential(
            nn.Conv2d(embed * 4, embed, 1, bias=False),
            nn.BatchNorm2d(embed),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(embed, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W   = x.shape[2:]
        target = (H // 4, W // 4)

        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        outs = []
        for feat, proj in zip([f1, f2, f3, f4], self.projs):
            out = proj(feat)
            out = F.interpolate(out, size=target, mode='bilinear', align_corners=False)
            outs.append(out)

        x = self.fuse(torch.cat(outs, dim=1))
        x = self.drop(x)
        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Swin-UNet   (gereksinim: pip install timm>=0.9)
# Girdi boyutu: 224×224 (Swin-Tiny pencere boyutu 7 ile tam uyumlu)
# ─────────────────────────────────────────────────────────────────────────────

class SwinUNet(nn.Module):
    """
    Swin Transformer (Swin-Tiny) encoder + CNN U-Net decoder.
    Referans: Cao et al., 'Swin-Unet: Unet-like Pure Transformer
              for Medical Image Segmentation' (ECCV 2022).

    Decoder CNN tabanlıdır (sınırlı GPU için optimize edilmiştir).
    Eğitimde IMG_SIZE=224 kullanın.

    Bağımlılık: pip install timm>=0.9
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        try:
            import timm  # noqa: F401
        except ImportError:
            raise ImportError(
                "SwinUNet için timm gereklidir.\n"
                "  Kurmak için: pip install timm>=0.9"
            )
        import timm

        self.encoder = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            features_only=True,
            in_chans=in_ch,
        )
        # Swin-Tiny kanal boyutları (stride 4,8,16,32):
        enc_ch = self.encoder.feature_info.channels()   # [96, 192, 384, 768]

        # Decoder: en derin feature'dan başlayarak upsample
        self.dec4 = DoubleConv(enc_ch[3], 384)
        self.up3  = nn.ConvTranspose2d(384, 384, 2, 2)
        self.dec3 = DoubleConv(384 + enc_ch[2], 192)
        self.up2  = nn.ConvTranspose2d(192, 192, 2, 2)
        self.dec2 = DoubleConv(192 + enc_ch[1],  96)
        self.up1  = nn.ConvTranspose2d(96,  96, 2, 2)
        self.dec1 = DoubleConv(96  + enc_ch[0],  48)

        # Patch embed stride 4'ten tam çözünürlüğe ×4 upsample
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(24, 16, 2, 2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm swin: özellikler (B, H, W, C) formatında gelir → (B, C, H, W) çevir
        f1, f2, f3, f4 = [t.permute(0, 3, 1, 2).contiguous()
                           for t in self.encoder(x)]

        d = self.dec4(f4)
        d = self.up3(d)
        if d.shape[2:] != f3.shape[2:]:
            d = F.interpolate(d, size=f3.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec3(torch.cat([d, f3], dim=1))

        d = self.up2(d)
        if d.shape[2:] != f2.shape[2:]:
            d = F.interpolate(d, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([d, f2], dim=1))

        d = self.up1(d)
        if d.shape[2:] != f1.shape[2:]:
            d = F.interpolate(d, size=f1.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec1(torch.cat([d, f1], dim=1))

        d = self.final_up(d)
        return self.head(d)