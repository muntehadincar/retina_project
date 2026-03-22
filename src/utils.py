import torch
import torch.nn as nn
import os


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Differansiyellenebilir Dice kayıp fonksiyonu (binary segmentasyon için)."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        target = (target > 0.5).float()
        inter = (prob * target).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + self.eps) / (union + self.eps)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """0.5 × BCEWithLogitsLoss  +  0.5 × DiceLoss"""

    def __init__(self, pos_weight: float = 5.0, eps: float = 1e-7):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.bce  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.dice = DiceLoss(eps=eps)
        self.alpha = 0.5

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # BCEWithLogitsLoss pos_weight'i otomatik doğru cihaza taşıyabilmek için
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        bce_val  = self.bce(logits, target)
        dice_val = self.dice(logits, target)
        return self.alpha * bce_val + (1.0 - self.alpha) * dice_val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits: torch.Tensor,
                    target: torch.Tensor,
                    thr: float = 0.3,
                    eps: float = 1e-7) -> dict:
    """
    Logit tensörlerinden tüm segmentasyon metriklerini hesaplar.

    Döndürülen anahtarlar:
        dice, iou, precision, recall, accuracy, sensitivity, specificity
    """
    pred   = (torch.sigmoid(logits) > thr).float()
    target = (target > 0.5).float()

    # Batchleri düzleştir, her örnek için hesapla, sonra ortala
    pred_f   = pred.view(pred.size(0), -1)
    target_f = target.view(target.size(0), -1)

    TP = (pred_f * target_f).sum(dim=1)
    FP = (pred_f * (1 - target_f)).sum(dim=1)
    FN = ((1 - pred_f) * target_f).sum(dim=1)
    TN = ((1 - pred_f) * (1 - target_f)).sum(dim=1)

    dice        = ((2 * TP + eps) / (2 * TP + FP + FN + eps)).mean().item()
    iou         = ((TP + eps) / (TP + FP + FN + eps)).mean().item()
    precision   = ((TP + eps) / (TP + FP + eps)).mean().item()
    recall      = ((TP + eps) / (TP + FN + eps)).mean().item()   # == sensitivity
    accuracy    = ((TP + TN + eps) / (TP + TN + FP + FN + eps)).mean().item()
    sensitivity = recall                                           # alias
    specificity = ((TN + eps) / (TN + FP + eps)).mean().item()

    return {
        "dice":        dice,
        "iou":         iou,
        "precision":   precision,
        "recall":      recall,
        "accuracy":    accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# Eski arayüz — geriye dönük uyumluluk
# ---------------------------------------------------------------------------

def dice_score(logits: torch.Tensor,
               target: torch.Tensor,
               eps: float = 1e-7,
               thr: float = 0.3) -> float:
    """Geriye dönük uyumluluk için korundu. compute_metrics kullanımı önerilir."""
    return compute_metrics(logits, target, thr=thr, eps=eps)["dice"]


# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def save_checkpoint(state_dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)


@torch.no_grad()
def predict_batch(model, x, device, thr: float = 0.3):
    model.eval()
    x = x.to(device)
    logits = model(x)
    prob   = torch.sigmoid(logits)
    pred   = (prob > thr).float()
    return prob.cpu(), pred.cpu()