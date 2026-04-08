"""
Loss functions for tooth & root canal segmentation.
Combines Dice Loss + Focal Loss to handle class imbalance (tiny canal).
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Computes per-class dice and returns 1 - mean(dice).
    """

    def __init__(self, num_classes: int = 3, smooth: float = 1e-5, ignore_bg: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W) raw model output
            targets: (B, 1, D, H, W) or (B, D, H, W) integer labels
        """
        probs = F.softmax(logits, dim=1)

        if targets.dim() == 5:
            targets = targets.squeeze(1)
        targets_one_hot = F.one_hot(targets.long(), self.num_classes)  # (B, D, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)

        start_class = 1 if self.ignore_bg else 0
        dice_scores = []

        for c in range(start_class, self.num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.
    Down-weights well-classified examples, focusing on hard examples.
    Particularly useful for the tiny canal class.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        num_classes: int = 3,
    ):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, D, H, W)
            targets: (B, 1, D, H, W) or (B, D, H, W)
        """
        if targets.dim() == 5:
            targets = targets.squeeze(1)

        ce_loss = F.cross_entropy(logits, targets.long(), reduction="none")
        probs = F.softmax(logits, dim=1)
        targets_expanded = targets.long().unsqueeze(1)  # (B, 1, D, H, W)
        pt = probs.gather(1, targets_expanded).squeeze(1)  # (B, D, H, W)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]
            focal_weight = focal_weight * alpha_t

        loss = (focal_weight * ce_loss).mean()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Dice + Focal loss.
    loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    """

    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[List[float]] = None,
    ):
        super().__init__()
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.focal_loss = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, num_classes=num_classes
        )
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        d_loss = self.dice_loss(logits, targets)
        f_loss = self.focal_loss(logits, targets)
        return self.dice_weight * d_loss + self.focal_weight * f_loss


class DeepSupervisionLoss(nn.Module):
    """
    Wraps CombinedLoss with deep supervision support for UNet3D.
    Applies loss at multiple decoder scales with decreasing weights.
    """

    def __init__(self, base_loss: CombinedLoss, weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights or [1.0, 0.5, 0.25]

    def forward(
        self,
        output: torch.Tensor,
        deep_outputs: list,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        loss = self.base_loss(output, targets)

        for i, deep_out in enumerate(deep_outputs):
            if i >= len(self.weights):
                break
            # Resize deep output to match target size
            if deep_out.shape[2:] != targets.shape[1:] and targets.dim() == 4:
                deep_out = F.interpolate(
                    deep_out, size=targets.shape[1:], mode="trilinear", align_corners=False
                )
            elif deep_out.shape[2:] != targets.shape[2:] and targets.dim() == 5:
                deep_out = F.interpolate(
                    deep_out, size=targets.shape[2:], mode="trilinear", align_corners=False
                )
            loss += self.weights[i] * self.base_loss(deep_out, targets)

        return loss
