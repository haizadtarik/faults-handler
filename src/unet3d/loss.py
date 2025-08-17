import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELossWrapper(nn.Module):
    """Standard BCE Loss."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, preds, targets):
        return self.loss_fn(preds, targets)


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class BalancedBCELoss(nn.Module):
    """Balanced BCE that re-weights positive and negative pixels."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.eps = epsilon

    def forward(self, preds, targets):
        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)
        pos = targets.sum()
        neg = targets.numel() - pos
        pos = pos + self.eps
        neg = neg + self.eps
        pos_weight = neg / pos
        pos_weight_tensor = torch.tensor([pos_weight], device=preds.device)
        return F.binary_cross_entropy(preds, targets, reduction="mean", weight=pos_weight_tensor)


class BalancedDiceLoss(nn.Module):
    """Balanced Dice giving equal importance to foreground and background."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Foreground dice
        intersection_fg = (preds * targets).sum()
        union_fg = preds.sum() + targets.sum()
        dice_fg = (2.0 * intersection_fg + self.smooth) / (union_fg + self.smooth)

        # Background dice
        preds_bg = 1.0 - preds
        targets_bg = 1.0 - targets
        intersection_bg = (preds_bg * targets_bg).sum()
        union_bg = preds_bg.sum() + targets_bg.sum()
        dice_bg = (2.0 * intersection_bg + self.smooth) / (union_bg + self.smooth)

        # Weighted average
        pos = targets.sum() + self.smooth
        neg = targets.numel() - pos + self.smooth
        total = pos + neg
        weight_fg = neg / total
        weight_bg = pos / total
        combined_dice = weight_fg * dice_fg + weight_bg * dice_bg
        return 1.0 - combined_dice


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification/segmentation.
    alpha: weighting factor [0..1]
    gamma: focusing parameter
    """

    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        preds = preds.clamp(min=self.eps, max=1.0 - self.eps)
        pt = preds * targets + (1 - preds) * (1 - targets)
        w = self.alpha * targets + (1.0 - self.alpha) * (1 - targets)
        focal = -w * (1 - pt).pow(self.gamma) * pt.log()
        return focal.mean()
