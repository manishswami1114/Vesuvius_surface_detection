import numpy as np
from scipy import ndimage
import cc3d

def voxel_accuracy(preds, targets):
    return (preds == targets).mean()

def dice_score_per_class(preds, targets, num_classes=3):
    out = {}
    for c in range(num_classes):
        p = (preds == c)
        t = (targets == c)
        intersection = (p & t).sum()
        denom = p.sum() + t.sum()
        dice = (2*intersection) / (denom + 1e-8) if denom > 0 else 1.0
        out[f'dice_c{c}'] = dice
    return out

def surface_dice_approx(pred, target, tol=2):
    from skimage import morphology
    b_pred = pred ^ ndimage.binary_erosion(pred)
    b_tgt = target ^ ndimage.binary_erosion(target)
    se = morphology.ball(tol)
    b_pred_d = ndimage.binary_dilation(b_pred, structure=se)
    b_tgt_d = ndimage.binary_dilation(b_tgt, structure=se)
    t_overlap = (b_tgt & b_pred_d).sum() / (b_tgt.sum() + 1e-8)
    p_overlap = (b_pred & b_tgt_d).sum() / (b_pred.sum() + 1e-8)
    return 0.5 * (t_overlap + p_overlap)
