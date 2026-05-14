import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# CELL 3: LOSS FUNCTIONS (ALL from epoch 0)
# =============================================================================

def soft_skeletonize(x, num_iter=5):
    """Differentiable soft skeletonization at half resolution"""
    orig_shape = x.shape[2:]
    # Half resolution - good balance of quality vs speed on H100
    x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)
    for _ in range(num_iter):
        min_pool = -F.max_pool3d(-x, 3, stride=1, padding=1)
        max_min_pool = F.max_pool3d(min_pool, 3, stride=1, padding=1)
        x = F.relu(x - max_min_pool)
    # Upscale back
    x = F.interpolate(x, size=orig_shape, mode='trilinear', align_corners=False)
    return x


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target, mask=None):
        pred = torch.sigmoid(pred)
        if mask is not None:
            pred = pred * (1 - mask)
            target = target * (1 - mask)
        inter = (pred * target).sum()
        union = pred.sum() + target.sum()
        return 1 - (2 * inter + self.smooth) / (union + self.smooth)


class clDiceLoss(nn.Module):
    """Centerline Dice — encourages continuity without bridges."""
    def __init__(self, num_iter=5, smooth=1e-5):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth
    
    def forward(self, pred, target, mask=None):
        pred_sig = torch.sigmoid(pred)
        if mask is not None:
            pred_sig = pred_sig * (1 - mask)
            target = target * (1 - mask)
        
        skel_pred = soft_skeletonize(pred_sig, self.num_iter)
        # Target skeleton doesn't need gradients
        with torch.no_grad():
            skel_target = soft_skeletonize(target, self.num_iter)
        
        tprec = ((skel_pred * target).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((skel_target * pred_sig).sum() + self.smooth) / (skel_target.sum() + self.smooth)
        
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1 - cl_dice


def gpu_approx_signed_distance(target, num_iters=5):
    """
    GPU-native approximate signed distance map via iterative morphological dilation.
    5 iterations for better boundary precision
    """
    binary = (target > 0.5).float()
    inv_binary = 1.0 - binary
    
    # Approximate distance to foreground boundary (for background pixels)
    dist_bg = torch.zeros_like(binary)
    frontier = inv_binary.clone()
    for i in range(1, num_iters + 1):
        dilated = F.max_pool3d(binary, 3, stride=1, padding=1)
        new_frontier = (dilated > 0.5) & (frontier > 0.5)
        dist_bg = dist_bg + new_frontier.float()
        frontier = frontier * (1.0 - new_frontier.float())
        binary = dilated
    dist_bg = dist_bg + frontier * (num_iters + 1)
    
    # Approximate distance to background boundary (for foreground pixels)
    binary_fg = (target > 0.5).float()
    dist_fg = torch.zeros_like(binary_fg)
    frontier_fg = binary_fg.clone()
    inv_fg = 1.0 - binary_fg
    for i in range(1, num_iters + 1):
        dilated = F.max_pool3d(inv_fg, 3, stride=1, padding=1)
        new_frontier = (dilated > 0.5) & (frontier_fg > 0.5)
        dist_fg = dist_fg + new_frontier.float()
        frontier_fg = frontier_fg * (1.0 - new_frontier.float())
        inv_fg = dilated
    dist_fg = dist_fg + frontier_fg * (num_iters + 1)
    
    # Signed distance: positive outside, negative inside
    is_fg = (target > 0.5).float()
    signed_dist = dist_bg * (1.0 - is_fg) - dist_fg * is_fg
    max_val = signed_dist.abs().amax(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
    signed_dist = signed_dist / max_val
    return signed_dist


class SurfaceLoss(nn.Module):
    """Surface Loss — GPU-native boundary alignment via approximate distance maps."""
    def forward(self, pred, target, mask=None):
        pred_sig = torch.sigmoid(pred)
        
        with torch.no_grad():
            dist_map = gpu_approx_signed_distance(target, num_iters=5)
        
        if mask is not None:
            pred_sig = pred_sig * (1 - mask)
            dist_map = dist_map * (1 - mask)
        
        return (pred_sig * dist_map).mean()


class TopoLoss(nn.Module):
    """Topology Loss — penalizes holes/bridges via Laplacian."""
    def __init__(self, sigma=2.0):
        super().__init__()
        self.sigma = sigma
        kernel = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=torch.float32).view(1, 1, 3, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)
    
    def forward(self, pred, target, mask=None):
        pred_sig = torch.sigmoid(pred)
        
        if mask is not None:
            pred_sig = pred_sig * (1 - mask)
            target = target * (1 - mask)
        
        kernel = self.laplacian_kernel.to(dtype=pred.dtype, device=pred.device)
        lap_pred = F.conv3d(pred_sig, kernel, padding=1)
        lap_target = F.conv3d(target, kernel, padding=1)
        
        topo_diff = (lap_pred - lap_target).abs()
        weight = torch.exp(-self.sigma * target)
        
        return (topo_diff * weight).mean()


class CombinedLossV11(nn.Module):
    """V11 Combined Loss — ALL losses from epoch 0."""
    def __init__(self, dice_w=0.25, bce_w=0.10, cldice_w=0.30, surface_w=0.15, topo_w=0.20):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.cldice_w = cldice_w
        self.surface_w = surface_w
        self.topo_w = topo_w
        
        self.dice_loss = DiceLoss()
        self.cldice_loss = clDiceLoss()
        self.surface_loss = SurfaceLoss()
        self.topo_loss = TopoLoss()
        self.ds_weights = [0.5, 0.25, 0.125]
    
    def forward(self, output, target, ignore_mask, epoch=None):
        if isinstance(output, dict):
            pred = output['output']
            deep = output.get('deep', [])
        else:
            pred = output
            deep = []
        
        dice = self.dice_loss(pred, target, ignore_mask)
        bce = F.binary_cross_entropy_with_logits(
            pred * (1 - ignore_mask) if ignore_mask is not None else pred,
            target * (1 - ignore_mask) if ignore_mask is not None else target,
        )
        cldice = self.cldice_loss(pred, target, ignore_mask)
        surface = self.surface_loss(pred, target, ignore_mask)
        topo = self.topo_loss(pred, target, ignore_mask)
        
        total = (self.dice_w * dice + self.bce_w * bce + self.cldice_w * cldice +
                 self.surface_w * surface + self.topo_w * topo)
        
        for i, ds in enumerate(deep):
            if i >= len(self.ds_weights): break
            ds_target = F.interpolate(target, size=ds.shape[2:], mode='nearest')
            ds_mask = F.interpolate(ignore_mask, size=ds.shape[2:], mode='nearest') if ignore_mask is not None else None
            total = total + self.ds_weights[i] * self.dice_loss(ds, ds_target, ds_mask)
        
        return {'total': total, 'dice': dice, 'bce': bce,
                'cldice': cldice, 'surface': surface, 'topo': topo}

print("Loss Stack ready (ALL from epoch 0)")