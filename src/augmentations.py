import torch
import torch.nn.functional as F

# =============================================================================
# CELL 5: GPU AUGMENTATIONS (all ops on GPU, no CPU bottleneck)
# =============================================================================

class GPUAugmentation3D(nn.Module):
    """
    GPU-native 3D augmentation using pure PyTorch ops.
    Replaces scipy-based CPU augmentations for ~10-50x speedup.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @torch.no_grad()
    def forward(self, img, lbl, ignore):
        """
        Args:
            img: (B, 1, D, H, W) float tensor on GPU
            lbl: (B, 1, D, H, W) float tensor on GPU
            ignore: (B, 1, D, H, W) float tensor on GPU
        Returns:
            Augmented (img, lbl, ignore) tensors
        """
        B = img.shape[0]
        device = img.device
        dtype = img.dtype  # Preserve input dtype (bfloat16)

        # --- Spatial augmentations (applied to all channels equally) ---

        # Flip (per-sample random)
        if self.cfg.AUG_FLIP:
            for ax in [2, 3, 4]:  # D, H, W
                mask = torch.rand(B, device=device) > 0.5
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0]
                    img[idx] = torch.flip(img[idx], [ax])
                    lbl[idx] = torch.flip(lbl[idx], [ax])
                    ignore[idx] = torch.flip(ignore[idx], [ax])

        # Rotate 90 in HW plane (per-sample random)
        if self.cfg.AUG_ROTATE:
            for b in range(B):
                k = torch.randint(0, 4, (1,)).item()
                if k > 0:
                    img[b] = torch.rot90(img[b], k, [2, 3])     # H, W dims
                    lbl[b] = torch.rot90(lbl[b], k, [2, 3])
                    ignore[b] = torch.rot90(ignore[b], k, [2, 3])

        # Elastic + Affine via grid_sample (batched, very fast on GPU)
        if self.cfg.AUG_ELASTIC or self.cfg.AUG_AFFINE:
            img, lbl, ignore = self._elastic_affine(img, lbl, ignore)

        # --- Intensity augmentations (image only) ---

        # Gaussian noise
        if self.cfg.AUG_NOISE:
            noise_mask = torch.rand(B, device=device) > 0.5
            if noise_mask.any():
                idx = noise_mask.nonzero(as_tuple=True)[0]
                img[idx] = img[idx] + torch.randn_like(img[idx]) * 0.05

        # Contrast
        if self.cfg.AUG_CONTRAST:
            contrast_mask = torch.rand(B, device=device) > 0.5
            if contrast_mask.any():
                idx = contrast_mask.nonzero(as_tuple=True)[0]
                mean = img[idx].mean(dim=(1, 2, 3, 4), keepdim=True)
                scale = torch.empty(len(idx), 1, 1, 1, 1, device=device, dtype=dtype).uniform_(0.8, 1.2)
                img[idx] = (img[idx] - mean) * scale + mean

        # Brightness
        if self.cfg.AUG_BRIGHTNESS:
            bright_mask = torch.rand(B, device=device) > 0.5
            if bright_mask.any():
                idx = bright_mask.nonzero(as_tuple=True)[0]
                shift = torch.empty(len(idx), 1, 1, 1, 1, device=device, dtype=dtype).uniform_(-0.1, 0.1)
                img[idx] = img[idx] + shift

        # Occlusion (random cubic patches zeroed out)
        if self.cfg.AUG_OCCLUSION:
            occ_mask = torch.rand(B, device=device) > 0.7
            if occ_mask.any():
                idx = occ_mask.nonzero(as_tuple=True)[0]
                img = self._apply_occlusion(img, idx)

        return img, lbl, ignore

    def _elastic_affine(self, img, lbl, ignore):
        """Combined elastic + affine deformation using F.grid_sample."""
        B, _, D, H, W = img.shape
        device = img.device
        dtype = img.dtype  # Preserve input dtype (bfloat16)

        # Create base grid: (B, D, H, W, 3) in [-1, 1] - MUST match input dtype
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, D, device=device, dtype=dtype),
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing='ij'
        ), dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1).clone()

        # Elastic deformation
        if self.cfg.AUG_ELASTIC:
            elastic_mask = torch.rand(B, device=device) > 0.5
            if elastic_mask.any():
                idx = elastic_mask.nonzero(as_tuple=True)[0]
                n_idx = len(idx)
                # Generate smooth random displacement at low res then upsample
                low_d, low_h, low_w = max(4, D // 16), max(4, H // 16), max(4, W // 16)
                sigma = self.cfg.AUG_ELASTIC_SIGMA
                # Scale displacement: sigma controls magnitude in normalized [-1,1] space
                disp_scale = sigma * 2.0 / max(D, H, W)
                noise = torch.randn(n_idx, 3, low_d, low_h, low_w, device=device, dtype=dtype) * disp_scale
                # Upsample to full resolution (smooth interpolation = smooth deformation)
                disp = F.interpolate(noise, size=(D, H, W), mode='trilinear', align_corners=False)
                # disp: (n_idx, 3, D, H, W) -> (n_idx, D, H, W, 3)
                disp = disp.permute(0, 2, 3, 4, 1)
                grid[idx] = grid[idx] + disp

        # Affine (scale)
        if self.cfg.AUG_AFFINE:
            affine_mask = torch.rand(B, device=device) > 0.5
            if affine_mask.any():
                idx = affine_mask.nonzero(as_tuple=True)[0]
                lo, hi = self.cfg.AUG_AFFINE_SCALE
                scales = torch.empty(len(idx), 1, 1, 1, 3, device=device, dtype=dtype).uniform_(lo, hi)
                grid[idx] = grid[idx] * scales

        # Clamp grid to valid range
        grid = grid.clamp(-1, 1)

        # Apply deformation - use grid_sample (very fast on GPU)
        # grid_sample expects grid in (B, D, H, W, 3) with order (x, y, z) but we have (z, y, x)
        # PyTorch grid_sample 3D: grid[..., 0] = W dim, grid[..., 1] = H dim, grid[..., 2] = D dim
        grid_sample = grid.flip(-1)  # (z,y,x) -> (x,y,z) for grid_sample

        img = F.grid_sample(img, grid_sample, mode='bilinear', padding_mode='reflection', align_corners=False)
        # Use nearest for labels to preserve discrete values
        lbl = F.grid_sample(lbl, grid_sample, mode='nearest', padding_mode='zeros', align_corners=False)
        ignore = F.grid_sample(ignore, grid_sample, mode='nearest', padding_mode='zeros', align_corners=False)

        return img, lbl, ignore

    def _apply_occlusion(self, img, idx):
        """Apply random cubic occlusion patches."""
        _, _, D, H, W = img.shape
        for b_idx in idx:
            for _ in range(3):
                if torch.rand(1).item() > 0.5:
                    size = torch.randint(5, 15, (1,)).item()
                    z = torch.randint(0, max(1, D - size), (1,)).item()
                    y = torch.randint(0, max(1, H - size), (1,)).item()
                    x = torch.randint(0, max(1, W - size), (1,)).item()
                    patch = img[b_idx, :, z:z+size, y:y+size, x:x+size]
                    img[b_idx, :, z:z+size, y:y+size, x:x+size] = patch.mean()
        return img

print("GPUAugmentation3D ready (all ops on GPU, bfloat16 compatible)")