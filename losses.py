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
    Soft Dice Loss for multi-class segmentation with optional per-class weights.

    Tính (1 - dice) cho từng class rồi lấy trung bình có trọng số.
    Trọng số [1, 1, 5] ưu tiên class canal gấp 5 lần → model bị phạt
    mạnh hơn khi bỏ sót canal.
    """

    def __init__(
        self,
        num_classes: int = 3,
        smooth: float = 1e-5,
        ignore_bg: bool = False,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

        if class_weights is not None:
            assert len(class_weights) == num_classes, (
                f"class_weights phải có độ dài {num_classes}, "
                f"nhận được {len(class_weights)}"
            )
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

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
        per_class_losses = []
        per_class_weights = []

        for c in range(start_class, self.num_classes):
            pred_c = probs[:, c]
            target_c = targets_one_hot[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            per_class_losses.append(1.0 - dice)

            if self.class_weights is not None:
                per_class_weights.append(self.class_weights[c])
            else:
                per_class_weights.append(torch.tensor(1.0, device=logits.device))

        losses = torch.stack(per_class_losses)
        weights = torch.stack(per_class_weights).to(losses.device)
        # Weighted mean (chuẩn hóa theo tổng trọng số)
        return (losses * weights).sum() / weights.sum()


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
    Combined Dice + Focal loss với class weights.

        loss = dice_weight * WeightedDiceLoss + focal_weight * FocalLoss

    class_weights được áp dụng cho CẢ dice và focal để model bị phạt
    nhất quán khi sai ở class canal. Mặc định [1, 1, 5] → canal x5.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        # Dùng cùng class_weights cho cả dice và focal
        if class_weights is None:
            class_weights = [1.0, 1.0, 5.0]  # default: canal x5

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
        )
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,  # reuse cho focal
            num_classes=num_classes,
        )
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.class_weights = class_weights

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
