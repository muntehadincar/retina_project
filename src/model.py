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