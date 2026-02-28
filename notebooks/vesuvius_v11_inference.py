# =============================================================================
# VESUVIUS V11 - INFERENCE WITH TOPOLOGY-AWARE POST-PROCESSING
# =============================================================================
#
# This notebook performs inference with the V11 trained model and applies
# comprehensive post-processing optimized for the VOI metric.
#
# Post-processing pipeline:
# 1. Sliding window inference with Gaussian weighting
# 2. Test-time augmentation (TTA)
# 3. Surface-aware smoothing
# 4. Slice-wise hole filling (for tunnels)
# 5. Small component removal
# 6. Topology validation
# =============================================================================

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import json
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from PIL import Image, ImageSequence
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from scipy import ndimage
from scipy.ndimage import (
    binary_fill_holes, binary_closing, binary_opening,
    binary_dilation, binary_erosion, distance_transform_edt,
    gaussian_filter, label, generate_binary_structure
)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class InferenceConfig:
    # Paths
    TEST_DATA_ROOT: Path = Path("/kaggle/input/vesuvius-challenge-2025")
    CHECKPOINT_PATH: Path = Path("/kaggle/input/v11-checkpoints/fold_0/best_model.pth")
    OUTPUT_DIR: Path = Path("/kaggle/working/predictions")

    # Model architecture (must match training)
    PATCH_SIZE: Tuple[int, int, int] = (192, 192, 192)
    FEATURES: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 320, 320])
    N_BLOCKS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 6])

    # Inference settings
    OVERLAP: float = 0.5  # 50% overlap for sliding window
    TTA_LEVEL: str = "flip"  # "none", "flip", "full"
    BATCH_SIZE: int = 1  # For inference

    # Post-processing settings
    THRESHOLD: float = 0.5
    SURFACE_SMOOTH_SIGMA: float = 0.5
    MIN_COMPONENT_SIZE: int = 100
    HOLE_FILL_AXES: List[int] = field(default_factory=lambda: [0, 1, 2])
    MAX_TUNNEL_DIAMETER: int = 2

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BFLOAT16: bool = True

    def __post_init__(self):
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = InferenceConfig()

print("="*70)
print("V11 INFERENCE WITH TOPOLOGY-AWARE POST-PROCESSING")
print("="*70)
print(f"Checkpoint: {cfg.CHECKPOINT_PATH}")
print(f"Patch size: {cfg.PATCH_SIZE}")
print(f"Overlap: {cfg.OVERLAP}")
print(f"TTA: {cfg.TTA_LEVEL}")
print(f"Threshold: {cfg.THRESHOLD}")
print("="*70)


# =============================================================================
# MODEL ARCHITECTURE (same as training)
# =============================================================================

def get_num_groups(channels, max_groups=32):
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class HybridConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 2
        self.conv_xy = nn.Conv3d(in_ch, mid_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.conv_z = nn.Conv3d(in_ch, out_ch - mid_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.norm = nn.GroupNorm(get_num_groups(out_ch), out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.norm(torch.cat([self.conv_xy(x), self.conv_z(x)], dim=1)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_hybrid=False):
        super().__init__()
        if use_hybrid:
            self.conv = HybridConv3d(in_ch, out_ch)
        else:
            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.GroupNorm(get_num_groups(out_ch), out_ch),
                nn.LeakyReLU(0.01, inplace=True),
            )

    def forward(self, x):
        return self.conv(x)


class MultiScaleResBlock(nn.Module):
    def __init__(self, channels, scales=2, use_hybrid=False):
        super().__init__()
        self.scales = scales
        self.width = channels // scales
        self.convs = nn.ModuleList([ConvBlock(self.width, self.width, use_hybrid=use_hybrid) for _ in range(scales - 1)])
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)

    def forward(self, x):
        splits = torch.chunk(x, self.scales, dim=1)
        outputs = [splits[0]]
        for i, conv in enumerate(self.convs):
            out = conv(splits[i + 1] + outputs[-1]) if i > 0 else conv(splits[i + 1])
            outputs.append(out)
        return x + self.norm(torch.cat(outputs, dim=1))


class AttentionBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, max(channels // reduction, 8))
        self.fc2 = nn.Linear(max(channels // reduction, 8), channels)
        self.conv_spatial = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c = x.shape[:2]
        ca = torch.sigmoid(self.fc2(F.relu(self.fc1(self.gap(x).view(b, c))))).view(b, c, 1, 1, 1)
        x_ca = x * ca
        sa = torch.sigmoid(self.conv_spatial(torch.cat([x_ca.mean(1, keepdim=True), x_ca.max(1, keepdim=True)[0]], 1)))
        return x_ca * sa


class SurfaceRefinementBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.edge_conv = nn.Conv3d(in_ch, in_ch, 3, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv3d(in_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(get_num_groups(out_ch), out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.refine(torch.cat([x, torch.abs(self.edge_conv(x))], dim=1))


class TopoPreservingUNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=None, n_blocks=None,
                 use_attention=True, use_hybrid_conv=True, use_surface_refinement=True, use_deep_supervision=False):
        super().__init__()
        features = features or [32, 64, 128, 256, 320, 320]
        n_blocks = n_blocks or [1, 2, 3, 4, 6, 6]
        self.features = features
        self.use_deep_supervision = use_deep_supervision

        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i, (feat, nb) in enumerate(zip(features, n_blocks)):
            in_channels = in_ch if i == 0 else features[i-1]
            layers = [ConvBlock(in_channels, feat, use_hybrid=use_hybrid_conv and i > 0)]
            layers.extend([MultiScaleResBlock(feat, scales=2, use_hybrid=use_hybrid_conv) for _ in range(nb)])
            self.encoders.append(nn.Sequential(*layers))
            self.attentions.append(AttentionBlock(feat) if use_attention and i >= 2 else nn.Identity())
            if i < len(features) - 1:
                self.pools.append(nn.Conv3d(feat, feat, 2, stride=2))

        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        for i in range(len(features)-2, -1, -1):
            self.ups.append(nn.ConvTranspose3d(features[i+1], features[i], 2, stride=2))
            if use_surface_refinement and i == 0:
                self.dec_convs.append(SurfaceRefinementBlock(features[i]*2, features[i]))
            else:
                self.dec_convs.append(nn.Sequential(
                    ConvBlock(features[i]*2, features[i], use_hybrid=use_hybrid_conv),
                    MultiScaleResBlock(features[i], scales=2, use_hybrid=use_hybrid_conv),
                ))

        self.final = nn.Conv3d(features[0], out_ch, 1)

    def forward(self, x):
        enc_features = []
        for i, (enc, att) in enumerate(zip(self.encoders, self.attentions)):
            x = att(enc(x))
            enc_features.append(x)
            if i < len(self.pools): x = self.pools[i](x)

        enc_features = enc_features[::-1]
        x = enc_features[0]

        for i, (up, dec) in enumerate(zip(self.ups, self.dec_convs)):
            x = up(x)
            skip = enc_features[i+1]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))

        return self.final(x)


# =============================================================================
# NORMALIZATION (same as training)
# =============================================================================

def robust_zscore_normalize(img, lower_percentile=0.5, upper_percentile=99.5):
    """Robust Z-score normalization with percentile clipping."""
    p_low = np.percentile(img, lower_percentile)
    p_high = np.percentile(img, upper_percentile)
    img_clipped = np.clip(img, p_low, p_high)
    mean = img_clipped.mean()
    std = img_clipped.std()
    img_norm = (img_clipped - mean) / (std + 1e-8)
    return img_norm.astype(np.float32)


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================

def create_gaussian_weight(patch_size, sigma=0.125):
    """Create 3D Gaussian weighting kernel."""
    d, h, w = patch_size
    gauss_z = np.exp(-0.5 * ((np.arange(d) - d/2) / (d * sigma)) ** 2)
    gauss_y = np.exp(-0.5 * ((np.arange(h) - h/2) / (h * sigma)) ** 2)
    gauss_x = np.exp(-0.5 * ((np.arange(w) - w/2) / (w * sigma)) ** 2)
    gauss_weight = gauss_z[:, None, None] * gauss_y[None, :, None] * gauss_x[None, None, :]
    return gauss_weight.astype(np.float32)


@torch.no_grad()
def sliding_window_inference(
    model,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    device: str = 'cuda',
    use_bf16: bool = True,
) -> np.ndarray:
    """
    Sliding window inference with Gaussian weighting.
    """
    model.eval()
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd = int(pd * (1 - overlap))
    sh = int(ph * (1 - overlap))
    sw = int(pw * (1 - overlap))

    # Pad volume if needed
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)

    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
        D, H, W = volume.shape

    # Output arrays
    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)

    # Gaussian weight
    gauss_weight = create_gaussian_weight(patch_size)

    # Generate positions
    z_positions = list(range(0, max(1, D - pd + 1), sd))
    if z_positions[-1] + pd < D:
        z_positions.append(D - pd)

    y_positions = list(range(0, max(1, H - ph + 1), sh))
    if y_positions[-1] + ph < H:
        y_positions.append(H - ph)

    x_positions = list(range(0, max(1, W - pw + 1), sw))
    if x_positions[-1] + pw < W:
        x_positions.append(W - pw)

    # Normalize volume
    vol_norm = robust_zscore_normalize(volume)

    total_patches = len(z_positions) * len(y_positions) * len(x_positions)

    with tqdm(total=total_patches, desc="Inference", leave=False) as pbar:
        for z in z_positions:
            for y in y_positions:
                for x in x_positions:
                    # Extract patch
                    patch = vol_norm[z:z+pd, y:y+ph, x:x+pw]

                    # Inference
                    inp = torch.from_numpy(patch[None, None]).to(device, dtype=dtype)

                    with autocast(device_type='cuda', dtype=dtype):
                        out = model(inp)
                        prob = torch.sigmoid(out).squeeze().float().cpu().numpy()

                    # Accumulate with Gaussian weighting
                    pred_sum[z:z+pd, y:y+ph, x:x+pw] += prob * gauss_weight
                    weight_sum[z:z+pd, y:y+ph, x:x+pw] += gauss_weight

                    pbar.update(1)

    # Normalize by weight
    pred = pred_sum / np.maximum(weight_sum, 1e-8)

    # Remove padding
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        pred = pred[:D-pad_d if pad_d > 0 else D,
                    :H-pad_h if pad_h > 0 else H,
                    :W-pad_w if pad_w > 0 else W]

    return pred


@torch.no_grad()
def inference_with_tta(
    model,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    overlap: float = 0.5,
    device: str = 'cuda',
    use_bf16: bool = True,
    tta_level: str = 'flip',
) -> np.ndarray:
    """
    Inference with test-time augmentation.

    Args:
        tta_level:
            - "none": No TTA
            - "flip": Flip augmentations (4x)
            - "full": Flip + rotation (8x)
    """
    preds = []

    # Original
    print("  TTA: Original")
    pred = sliding_window_inference(model, volume, patch_size, overlap, device, use_bf16)
    preds.append(pred)

    if tta_level in ['flip', 'full']:
        # Flip augmentations
        for axis in [0, 1, 2]:
            print(f"  TTA: Flip axis {axis}")
            vol_flip = np.flip(volume, axis).copy()
            pred_flip = sliding_window_inference(model, vol_flip, patch_size, overlap, device, use_bf16)
            preds.append(np.flip(pred_flip, axis).copy())

    if tta_level == 'full':
        # 90° rotations in each plane
        for k in [1, 2, 3]:
            print(f"  TTA: Rotate k={k}")
            vol_rot = np.rot90(volume, k=k, axes=(1, 2)).copy()
            pred_rot = sliding_window_inference(model, vol_rot, patch_size, overlap, device, use_bf16)
            preds.append(np.rot90(pred_rot, k=-k, axes=(1, 2)).copy())

    # Average all predictions
    print(f"  TTA: Averaging {len(preds)} predictions")
    return np.mean(preds, axis=0)


# =============================================================================
# POST-PROCESSING FUNCTIONS
# =============================================================================

def surface_aware_smoothing(mask: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Smooth the surface while preserving topology.
    Works by smoothing the signed distance field.
    """
    # Compute signed distance field
    dist_inside = distance_transform_edt(mask)
    dist_outside = distance_transform_edt(~mask)
    sdf = dist_inside - dist_outside

    # Smooth the SDF
    sdf_smooth = gaussian_filter(sdf.astype(np.float32), sigma=sigma)

    # Re-threshold
    smoothed = (sdf_smooth > 0).astype(np.uint8)

    return smoothed


def slice_wise_hole_fill(mask: np.ndarray, axes: List[int] = [0, 1, 2]) -> np.ndarray:
    """
    Fill holes in each 2D slice across specified axes.
    This catches "tunnels" that 3D hole filling misses.
    """
    filled = mask.copy()

    for axis in axes:
        for i in range(mask.shape[axis]):
            if axis == 0:
                slice_2d = filled[i, :, :]
                filled[i, :, :] = binary_fill_holes(slice_2d)
            elif axis == 1:
                slice_2d = filled[:, i, :]
                filled[:, i, :] = binary_fill_holes(slice_2d)
            else:
                slice_2d = filled[:, :, i]
                filled[:, :, i] = binary_fill_holes(slice_2d)

    return filled


def fill_small_tunnels(mask: np.ndarray, max_diameter: int = 2) -> np.ndarray:
    """
    Fill small tunnels by closing operation, keeping only near-surface changes.
    """
    struct = generate_binary_structure(3, 1)  # 6-connectivity

    # Dilate then erode (closing)
    dilated = binary_dilation(mask, structure=struct, iterations=max_diameter)
    closed = binary_erosion(dilated, structure=struct, iterations=max_diameter)

    # Only add voxels close to original surface
    dist_to_surface = distance_transform_edt(~mask)
    near_surface = dist_to_surface <= max_diameter

    # Combine
    filled = mask | (closed & near_surface)

    return filled.astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """Remove connected components smaller than min_size."""
    struct_26 = generate_binary_structure(3, 3)  # 26-connectivity
    labeled, n_components = label(mask, structure=struct_26)

    if n_components == 0:
        return mask

    # Get component sizes
    component_sizes = np.bincount(labeled.ravel())

    # Create mask of small components
    small_components = component_sizes < min_size
    small_components[0] = False  # Background is not a component

    # Remove small components
    small_mask = small_components[labeled]
    mask_cleaned = mask.copy()
    mask_cleaned[small_mask] = 0

    return mask_cleaned


def count_components(mask: np.ndarray) -> int:
    """Count 26-connected components."""
    struct_26 = generate_binary_structure(3, 3)
    _, n = label(mask, structure=struct_26)
    return n


def topology_safe_operation(mask: np.ndarray, operation, *args, **kwargs) -> np.ndarray:
    """
    Apply an operation but revert if it merges components.
    """
    n_before = count_components(mask)
    result = operation(mask, *args, **kwargs)
    n_after = count_components(result)

    if n_after < n_before:
        print(f"  Warning: Operation merged components ({n_before} -> {n_after}), reverting")
        return mask

    return result


def comprehensive_postprocess(
    pred_prob: np.ndarray,
    threshold: float = 0.5,
    surface_sigma: float = 0.5,
    min_component_size: int = 100,
    hole_fill_axes: List[int] = [0, 1, 2],
    max_tunnel_diameter: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Comprehensive topology-aware post-processing pipeline.

    Pipeline:
    1. Threshold probability map
    2. Remove small components
    3. Surface-aware smoothing
    4. Slice-wise hole filling (safe)
    5. Small tunnel filling (safe)
    6. Final cleanup
    """
    if verbose:
        print("Post-processing pipeline:")

    # Step 1: Threshold
    mask = (pred_prob > threshold).astype(np.uint8)
    if verbose:
        fg_pct = 100 * mask.mean()
        print(f"  1. Threshold ({threshold}): FG={fg_pct:.2f}%")

    # Step 2: Remove small components
    n_before = count_components(mask)
    mask = remove_small_components(mask, min_component_size)
    n_after = count_components(mask)
    if verbose:
        print(f"  2. Remove small components (<{min_component_size}): {n_before} -> {n_after}")

    # Step 3: Surface-aware smoothing
    mask = surface_aware_smoothing(mask, sigma=surface_sigma)
    if verbose:
        fg_pct = 100 * mask.mean()
        print(f"  3. Surface smoothing (σ={surface_sigma}): FG={fg_pct:.2f}%")

    # Step 4: Slice-wise hole filling (topology-safe)
    mask = topology_safe_operation(mask, slice_wise_hole_fill, axes=hole_fill_axes)
    if verbose:
        fg_pct = 100 * mask.mean()
        print(f"  4. Slice-wise hole fill: FG={fg_pct:.2f}%")

    # Step 5: Small tunnel filling (topology-safe)
    mask = topology_safe_operation(mask, fill_small_tunnels, max_diameter=max_tunnel_diameter)
    if verbose:
        fg_pct = 100 * mask.mean()
        print(f"  5. Tunnel fill (d={max_tunnel_diameter}): FG={fg_pct:.2f}%")

    # Step 6: Final small component removal
    mask = remove_small_components(mask, min_component_size)
    n_final = count_components(mask)
    if verbose:
        fg_pct = 100 * mask.mean()
        print(f"  6. Final cleanup: {n_final} components, FG={fg_pct:.2f}%")

    return mask


# =============================================================================
# MAIN INFERENCE FUNCTION
# =============================================================================

def load_volume(path):
    """Load 3D TIFF volume."""
    try:
        import tifffile
        return tifffile.imread(str(path))
    except:
        im = Image.open(path)
        return np.stack([np.array(p) for p in ImageSequence.Iterator(im)], axis=0)


def save_volume(path, volume):
    """Save 3D volume as TIFF."""
    import tifffile
    tifffile.imwrite(str(path), volume.astype(np.uint8), compression='zlib')


def load_model(checkpoint_path: Path, device: str = 'cuda') -> nn.Module:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    model = TopoPreservingUNet3D(
        features=cfg.FEATURES,
        n_blocks=cfg.N_BLOCKS,
        use_attention=True,
        use_hybrid_conv=True,
        use_surface_refinement=True,
        use_deep_supervision=False,  # Disable for inference
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle torch.compile prefix
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best dice: {checkpoint.get('best_dice', 'unknown')}")

    return model


def run_inference_single(
    model: nn.Module,
    volume_path: Path,
    output_path: Path,
    cfg: InferenceConfig,
) -> np.ndarray:
    """Run inference on a single volume."""
    print(f"\nProcessing: {volume_path.name}")

    # Load volume
    volume = load_volume(volume_path).astype(np.float32)
    print(f"  Volume shape: {volume.shape}")

    # Run inference with TTA
    print(f"  Running inference (TTA={cfg.TTA_LEVEL})...")
    pred_prob = inference_with_tta(
        model, volume, cfg.PATCH_SIZE, cfg.OVERLAP,
        cfg.DEVICE, cfg.USE_BFLOAT16, cfg.TTA_LEVEL
    )

    # Post-processing
    print(f"  Post-processing...")
    pred_mask = comprehensive_postprocess(
        pred_prob,
        threshold=cfg.THRESHOLD,
        surface_sigma=cfg.SURFACE_SMOOTH_SIGMA,
        min_component_size=cfg.MIN_COMPONENT_SIZE,
        hole_fill_axes=cfg.HOLE_FILL_AXES,
        max_tunnel_diameter=cfg.MAX_TUNNEL_DIAMETER,
        verbose=True,
    )

    # Save
    save_volume(output_path, pred_mask)
    print(f"  Saved to: {output_path}")

    return pred_mask


def run_inference_all(cfg: InferenceConfig):
    """Run inference on all test volumes."""
    # Load model
    model = load_model(cfg.CHECKPOINT_PATH, cfg.DEVICE)

    # Compile for speed (optional)
    if hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model, mode='reduce-overhead')

    # Find test volumes
    test_dir = cfg.TEST_DATA_ROOT / "test"
    if not test_dir.exists():
        test_dir = cfg.TEST_DATA_ROOT

    test_volumes = sorted(test_dir.glob("*.tif")) + sorted(test_dir.glob("*.tiff"))
    print(f"\nFound {len(test_volumes)} test volumes")

    # Process each volume
    for vol_path in test_volumes:
        output_path = cfg.OUTPUT_DIR / vol_path.name
        run_inference_single(model, vol_path, output_path, cfg)

    print("\n" + "="*70)
    print("INFERENCE COMPLETE")
    print("="*70)
    print(f"Predictions saved to: {cfg.OUTPUT_DIR}")


# =============================================================================
# SUBMISSION HELPER
# =============================================================================

def create_submission(predictions_dir: Path, output_csv: Path):
    """Create submission CSV from predictions."""
    import pandas as pd

    pred_files = sorted(predictions_dir.glob("*.tif")) + sorted(predictions_dir.glob("*.tiff"))

    rows = []
    for pred_path in tqdm(pred_files, desc="Creating submission"):
        vol_id = pred_path.stem
        mask = load_volume(pred_path)

        # RLE encoding
        flat = mask.ravel()
        # ... (add RLE encoding logic if needed)

        rows.append({'id': vol_id, 'prediction': mask.tobytes()})

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Submission saved to: {output_csv}")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    # Run inference
    run_inference_all(cfg)
