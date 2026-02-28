#!/usr/bin/env python3
"""Test loss functions produce valid outputs on CPU with random tensors."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- Paste the loss functions from the notebook ---

def soft_skeletonize(x, num_iter=5):
    orig_shape = x.shape[2:]
    x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)
    for _ in range(num_iter):
        min_pool = -F.max_pool3d(-x, 3, stride=1, padding=1)
        max_min_pool = F.max_pool3d(min_pool, 3, stride=1, padding=1)
        x = F.relu(x - max_min_pool)
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
        with torch.no_grad():
            skel_target = soft_skeletonize(target, self.num_iter)
        tprec = ((skel_pred * target).sum() + self.smooth) / (skel_pred.sum() + self.smooth)
        tsens = ((skel_target * pred_sig).sum() + self.smooth) / (skel_target.sum() + self.smooth)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1 - cl_dice


def gpu_approx_signed_distance(target, num_iters=8):
    binary = (target > 0.5).float()
    inv_binary = 1.0 - binary
    dist_bg = torch.zeros_like(binary)
    frontier = inv_binary.clone()
    for i in range(1, num_iters + 1):
        dilated = F.max_pool3d(binary, 3, stride=1, padding=1)
        new_frontier = (dilated > 0.5) & (frontier > 0.5)
        dist_bg = dist_bg + new_frontier.float()
        frontier = frontier * (1.0 - new_frontier.float())
        binary = dilated
    dist_bg = dist_bg + frontier * (num_iters + 1)
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
    is_fg = (target > 0.5).float()
    signed_dist = dist_bg * (1.0 - is_fg) - dist_fg * is_fg
    max_val = signed_dist.abs().amax(dim=(1, 2, 3, 4), keepdim=True) + 1e-8
    signed_dist = signed_dist / max_val
    return signed_dist


class SurfaceLoss(nn.Module):
    def forward(self, pred, target, mask=None):
        pred_sig = torch.sigmoid(pred)
        with torch.no_grad():
            dist_map = gpu_approx_signed_distance(target, num_iters=8)
        if mask is not None:
            pred_sig = pred_sig * (1 - mask)
            dist_map = dist_map * (1 - mask)
        return (pred_sig * dist_map).mean()


class TopoLoss(nn.Module):
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
        kernel = self.laplacian_kernel.to(dtype=pred.dtype)
        lap_pred = F.conv3d(pred_sig, kernel, padding=1)
        lap_target = F.conv3d(target, kernel, padding=1)
        topo_diff = (lap_pred - lap_target).abs()
        weight = torch.exp(-self.sigma * target)
        return (topo_diff * weight).mean()


# --- Test on smaller volume (64³) for CPU speed ---
SIZE = 64
print(f"Testing loss functions on {SIZE}³ patches (CPU)...")
print("=" * 60)

pred = torch.randn(1, 1, SIZE, SIZE, SIZE)
target = (torch.rand(1, 1, SIZE, SIZE, SIZE) > 0.5).float()
mask = (torch.rand(1, 1, SIZE, SIZE, SIZE) > 0.9).float()

losses = {
    'DiceLoss': DiceLoss(),
    'clDiceLoss': clDiceLoss(),
    'SurfaceLoss': SurfaceLoss(),
    'TopoLoss': TopoLoss(),
}

all_passed = True
for name, loss_fn in losses.items():
    t0 = time.time()
    try:
        val = loss_fn(pred, target, mask)
        dt = time.time() - t0
        is_finite = torch.isfinite(val).item()
        status = "✅" if is_finite else "❌ (not finite!)"
        print(f"  {name:14s}: {val.item():+.6f}  ({dt*1000:.0f}ms)  {status}")
        if not is_finite:
            all_passed = False
    except Exception as e:
        print(f"  {name:14s}: ❌ ERROR: {e}")
        all_passed = False

# Test gradient flow
print("\nGradient flow test:")
pred_g = torch.randn(1, 1, SIZE, SIZE, SIZE, requires_grad=True)
for name, loss_fn in losses.items():
    try:
        val = loss_fn(pred_g, target, mask)
        val.backward()
        grad_ok = pred_g.grad is not None and torch.isfinite(pred_g.grad).all()
        pred_g.grad = None
        status = "✅" if grad_ok else "❌ (bad grads!)"
        print(f"  {name:14s}: {status}")
        if not grad_ok:
            all_passed = False
    except Exception as e:
        print(f"  {name:14s}: ❌ ERROR: {e}")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS PASSED ✅")
else:
    print("SOME TESTS FAILED ❌")
    exit(1)
