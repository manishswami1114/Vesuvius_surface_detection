"""
Vesuvius Challenge Experiment Lab - Loss Functions
==================================================
Comprehensive loss functions for 3D surface segmentation.
Handles class imbalance and unlabeled regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Handles unlabeled regions via mask.
    """
    
    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted logits [B, 1, D, H, W]
            target: Ground truth [B, 1, D, H, W] or [B, D, H, W]
            mask: Valid region mask (1 = valid, 0 = ignore)
        """
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        
        # Ensure consistent shapes
        if target.dim() == 4:
            target = target.unsqueeze(1)
        if pred.dim() != target.dim():
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(mask.size(0), -1)
            pred_flat = pred_flat * mask_flat
            target_flat = target_flat * mask_flat
        
        # Compute Dice
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice.mean()


class BCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with optional class weighting and masking.
    """
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        # Compute BCE
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=pred.device)
            loss = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pos_weight, reduction='none'
            )
        else:
            loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Apply mask
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)
            mask = mask.float()
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Downweights easy examples to focus on hard ones.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        sigmoid: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = sigmoid
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        target = target.float()
        
        if self.sigmoid:
            p = torch.sigmoid(pred)
        else:
            p = pred
        
        # Focal weight
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        
        loss = alpha_weight * focal_weight * ce_loss
        
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)
            mask = mask.float()
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice that controls FP/FN trade-off.
    alpha > beta penalizes FP more (precision-focused)
    alpha < beta penalizes FN more (recall-focused)
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        smooth: float = 1e-5,
        sigmoid: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.sigmoid = sigmoid
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.sigmoid:
            pred = torch.sigmoid(pred)
        
        if target.dim() == 4:
            target = target.unsqueeze(1)
        
        # Flatten
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()
        
        if mask is not None:
            mask_flat = mask.view(mask.size(0), -1)
            pred_flat = pred_flat * mask_flat
            target_flat = target_flat * mask_flat
        
        # True positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = ((1 - target_flat) * pred_flat).sum(dim=1)
        fn = (target_flat * (1 - pred_flat)).sum(dim=1)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        return 1.0 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of multiple losses.
    Best practice: Dice + BCE/Focal for stable training.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.3,
        focal_weight: float = 0.2,
        tversky_weight: float = 0.0,
        dice_smooth: float = 1e-5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3
    ):
        super().__init__()
        
        self.weights = {
            'dice': dice_weight,
            'bce': bce_weight,
            'focal': focal_weight,
            'tversky': tversky_weight
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        # Initialize loss components
        self.dice_loss = DiceLoss(smooth=dice_smooth) if dice_weight > 0 else None
        self.bce_loss = BCEWithLogitsLoss() if bce_weight > 0 else None
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma) if focal_weight > 0 else None
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta) if tversky_weight > 0 else None
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Returns total loss and component breakdown."""
        
        total_loss = 0.0
        loss_dict = {}
        
        if self.dice_loss is not None and self.weights['dice'] > 0:
            dice_l = self.dice_loss(pred, target, mask)
            loss_dict['dice'] = dice_l.item()
            total_loss = total_loss + self.weights['dice'] * dice_l
        
        if self.bce_loss is not None and self.weights['bce'] > 0:
            bce_l = self.bce_loss(pred, target, mask)
            loss_dict['bce'] = bce_l.item()
            total_loss = total_loss + self.weights['bce'] * bce_l
        
        if self.focal_loss is not None and self.weights['focal'] > 0:
            focal_l = self.focal_loss(pred, target, mask)
            loss_dict['focal'] = focal_l.item()
            total_loss = total_loss + self.weights['focal'] * focal_l
        
        if self.tversky_loss is not None and self.weights['tversky'] > 0:
            tversky_l = self.tversky_loss(pred, target, mask)
            loss_dict['tversky'] = tversky_l.item()
            total_loss = total_loss + self.weights['tversky'] * tversky_l
        
        loss_dict['total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        
        return total_loss, loss_dict


class DeepSupervisionLoss(nn.Module):
    """
    Loss wrapper for deep supervision (multi-scale outputs).
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        weights: List[float] = [1.0, 0.5, 0.25, 0.125]
    ):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights
    
    def forward(
        self,
        preds: List[torch.Tensor],
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            preds: List of predictions at different scales
            target: Ground truth (single scale, will be resized)
            mask: Valid region mask
        """
        total_loss = 0.0
        
        for i, (pred, weight) in enumerate(zip(preds, self.weights[:len(preds)])):
            # Resize target to match prediction if needed
            if pred.shape[2:] != target.shape[1:]:
                target_resized = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=pred.shape[2:],
                    mode='nearest'
                ).squeeze(1).long()
                
                if mask is not None:
                    mask_resized = F.interpolate(
                        mask.unsqueeze(1).float(),
                        size=pred.shape[2:],
                        mode='nearest'
                    ).squeeze(1)
                else:
                    mask_resized = None
            else:
                target_resized = target
                mask_resized = mask
            
            loss = self.base_loss(pred, target_resized, mask_resized)
            if isinstance(loss, tuple):
                loss = loss[0]
            total_loss = total_loss + weight * loss
        
        return total_loss


def build_loss(config) -> nn.Module:
    """Factory function to build loss from config."""
    
    loss_type = config.loss.loss_type.lower()
    
    if loss_type == "dice":
        base_loss = DiceLoss(smooth=config.loss.dice_smooth)
    
    elif loss_type == "bce":
        base_loss = BCEWithLogitsLoss()
    
    elif loss_type == "focal":
        base_loss = FocalLoss(
            alpha=config.loss.focal_alpha,
            gamma=config.loss.focal_gamma
        )
    
    elif loss_type == "tversky":
        base_loss = TverskyLoss(
            alpha=config.loss.tversky_alpha,
            beta=config.loss.tversky_beta
        )
    
    elif loss_type == "combined":
        base_loss = CombinedLoss(
            dice_weight=config.loss.dice_weight,
            bce_weight=config.loss.bce_weight,
            focal_weight=config.loss.focal_weight,
            dice_smooth=config.loss.dice_smooth,
            focal_alpha=config.loss.focal_alpha,
            focal_gamma=config.loss.focal_gamma
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Wrap with deep supervision if enabled
    if config.model.use_deep_supervision:
        return DeepSupervisionLoss(base_loss, config.loss.ds_weights)
    
    return base_loss


if __name__ == "__main__":
    # Test losses
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    pred = torch.randn(2, 1, 32, 32, 32, device=device)
    target = torch.randint(0, 2, (2, 32, 32, 32), device=device)
    mask = torch.ones_like(target)
    
    losses = [
        ("Dice", DiceLoss()),
        ("BCE", BCEWithLogitsLoss()),
        ("Focal", FocalLoss()),
        ("Tversky", TverskyLoss()),
        ("Combined", CombinedLoss())
    ]
    
    for name, loss_fn in losses:
        loss_fn = loss_fn.to(device)
        result = loss_fn(pred, target, mask)
        if isinstance(result, tuple):
            print(f"{name} Loss: {result[0].item():.4f} | Components: {result[1]}")
        else:
            print(f"{name} Loss: {result.item():.4f}")