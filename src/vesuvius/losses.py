import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, eps=1e-6, smooth=1.0):
        super().__init__()
        self.eps = eps
        self.smooth = smooth

    def forward(self, logits, targets):
        n_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(targets, num_classes=n_classes).permute(0,4,1,2,3).float()
        dims = (2,3,4)
        intersection = (probs * target_onehot).sum(dims)
        cardinality = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.eps + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        l_ce = self.ce(logits, targets)
        l_dice = self.dice(logits, targets)
        return self.ce_weight * l_ce + self.dice_weight * l_dice
