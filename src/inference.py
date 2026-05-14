from .models import *
from .utils import *
from .postprocessing import *
import gc
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import warnings
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import (
    binary_fill_holes, distance_transform_edt, gaussian_filter,
    label, generate_binary_structure, binary_dilation
)
from skimage.morphology import skeletonize

warnings.filterwarnings('ignore')

# =============================================================================
# MULTI-GPU DETECTION
# =============================================================================
N_GPUS = torch.cuda.device_count()
print(f"Available GPUs: {N_GPUS}")
for i in range(N_GPUS):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

@dataclass
class Config:
    # Paths
    TEST_ROOT: Path = Path("/kaggle/input/vesuvius-challenge-surface-detection/test_images")
    CHECKPOINT_PATH: Path = Path("/kaggle/input/models/manish756/v11-vesuvius-model/pytorch/default/6/checkpoints_v11/fold_0/best_model.pth")
    OUTPUT_DIR: Path = Path("/kaggle/working")
    
    # Model (must match training)
    TRAIN_PATCH_SIZE: Tuple[int, int, int] = (192, 192, 192)
    FEATURES: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 320, 320])
    N_BLOCKS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 6])
    
    # Inference - MATCH TRAINING!
    INFER_PATCH_SIZE: Tuple[int, int, int] = (192, 192, 192)  # Same as training!
    OVERLAP: float = 0.7
    TTA_LEVEL: str = "flip"
    USE_FLOAT16: bool = True
    
    # Multi-GPU settings
    USE_MULTI_GPU: bool = True
    BATCH_SIZE: int = 1 * N_GPUS  # 1 patch per GPU (192³ is large)
    
    # Post-processing - MATCH BASELINE!
    THRESHOLD: float = 0.70  # Baseline uses 0.70, not 0.50!
    
    DEVICE: str = "cuda"

cfg = Config()
print(f"\nConfiguration:")
print(f"  Inference patch: {cfg.INFER_PATCH_SIZE} (same as training)")
print(f"  Batch size: {cfg.BATCH_SIZE} (across {N_GPUS} GPUs)")
print(f"  Threshold: {cfg.THRESHOLD}")
print(f"  TTA: {cfg.TTA_LEVEL}")

# Clear GPU memory on all devices
for i in range(N_GPUS):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
gc.collect()

# =============================================================================
# MULTI-GPU INFERENCE - MATCHING V11 TRAINING EXACTLY
# =============================================================================

def robust_zscore_normalize(img, lower_percentile=0.5, upper_percentile=99.5):
    """
    From V11 training notebook - percentile clipping + Z-score.
    """
    p_low = np.percentile(img, lower_percentile)
    p_high = np.percentile(img, upper_percentile)
    img_clipped = np.clip(img, p_low, p_high)
    mean = img_clipped.mean()
    std = img_clipped.std()
    img_norm = (img_clipped - mean) / (std + 1e-8)
    return img_norm.astype(np.float32)


def create_gaussian_weight(patch_size, sigma=0.125):
    d, h, w = patch_size
    gz = np.exp(-0.5 * ((np.arange(d) - d/2) / (d * sigma)) ** 2)
    gy = np.exp(-0.5 * ((np.arange(h) - h/2) / (h * sigma)) ** 2)
    gx = np.exp(-0.5 * ((np.arange(w) - w/2) / (w * sigma)) ** 2)
    return (gz[:, None, None] * gy[None, :, None] * gx[None, None, :]).astype(np.float32)


def get_patch_positions(volume_shape, patch_size, overlap=0.5):
    """Generate all patch positions for the volume."""
    D, H, W = volume_shape
    pd, ph, pw = patch_size
    sd, sh, sw = int(pd*(1-overlap)), int(ph*(1-overlap)), int(pw*(1-overlap))
    
    z_pos = list(range(0, max(1, D-pd+1), sd))
    if len(z_pos) == 0 or z_pos[-1] + pd < D: z_pos.append(max(0, D - pd))
    y_pos = list(range(0, max(1, H-ph+1), sh))
    if len(y_pos) == 0 or y_pos[-1] + ph < H: y_pos.append(max(0, H - ph))
    x_pos = list(range(0, max(1, W-pw+1), sw))
    if len(x_pos) == 0 or x_pos[-1] + pw < W: x_pos.append(max(0, W - pw))
    
    positions = []
    for z in z_pos:
        for y in y_pos:
            for x in x_pos:
                positions.append((z, y, x))
    return positions


@torch.no_grad()
def sliding_window_inference_multigpu(model, volume, patch_size, overlap=0.5, batch_size=2):
    """
    Multi-GPU sliding window inference.
    """
    model.eval()
    
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    
    # Pad if needed
    pad_d, pad_h, pad_w = max(0, pd-D), max(0, ph-H), max(0, pw-W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0,pad_d),(0,pad_h),(0,pad_w)), mode='reflect')
        D, H, W = volume.shape
    
    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    weight_sum = np.zeros((D, H, W), dtype=np.float32)
    gauss = create_gaussian_weight(patch_size)
    
    # Get all patch positions
    positions = get_patch_positions((D, H, W), patch_size, overlap)
    
    # Normalize volume EXACTLY as in training
    vol_norm = robust_zscore_normalize(volume, lower_percentile=0.5, upper_percentile=99.5)
    
    print(f"  Total patches: {len(positions)}, Batch size: {batch_size}")
    
    # Process in batches
    for batch_start in tqdm(range(0, len(positions), batch_size), desc="Inference"):
        batch_end = min(batch_start + batch_size, len(positions))
        batch_positions = positions[batch_start:batch_end]
        
        # Extract patches for this batch
        patches = []
        for (z, y, x) in batch_positions:
            patch = vol_norm[z:z+pd, y:y+ph, x:x+pw]
            patches.append(patch)
        
        # Stack into batch tensor [B, 1, D, H, W]
        batch_tensor = torch.from_numpy(np.stack(patches)[:, None]).cuda().half()
        
        # Forward pass
        with torch.cuda.amp.autocast(dtype=torch.float16):
            batch_pred = torch.sigmoid(model(batch_tensor))
        
        # Convert back to numpy
        batch_pred = batch_pred.squeeze(1).float().cpu().numpy()
        
        # Accumulate predictions
        for i, (z, y, x) in enumerate(batch_positions):
            pred_sum[z:z+pd, y:y+ph, x:x+pw] += batch_pred[i] * gauss
            weight_sum[z:z+pd, y:y+ph, x:x+pw] += gauss
        
        # Cleanup
        del batch_tensor, batch_pred, patches
        
        # Periodic GPU cleanup
        if (batch_start // batch_size) % 10 == 0:
            torch.cuda.empty_cache()
    
    pred = pred_sum / np.maximum(weight_sum, 1e-8)
    
    # Remove padding
    if pad_d > 0: pred = pred[:-pad_d]
    if pad_h > 0: pred = pred[:, :-pad_h]
    if pad_w > 0: pred = pred[:, :, :-pad_w]
    
    return pred


@torch.no_grad()
def inference_with_tta_multigpu(model, volume, patch_size, overlap=0.5, batch_size=2, tta='flip'):
    """TTA with multi-GPU support."""
    # Original
    print("  TTA: Original")
    pred = sliding_window_inference_multigpu(model, volume, patch_size, overlap, batch_size)
    
    if tta == 'none':
        return pred
    
    preds = [pred]
    del pred
    gc.collect()
    for i in range(N_GPUS):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    
    if tta in ['flip', 'full']:
        for axis in [0, 1, 2]:
            print(f"  TTA: Flip axis {axis}")
            vol_flip = np.flip(volume, axis).copy()
            pred_flip = sliding_window_inference_multigpu(model, vol_flip, patch_size, overlap, batch_size)
            preds.append(np.flip(pred_flip, axis).copy())
            
            del vol_flip, pred_flip
            gc.collect()
            for i in range(N_GPUS):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    
    print(f"  TTA: Averaging {len(preds)} predictions")
    result = np.mean(preds, axis=0)
    del preds
    gc.collect()
    
    return result

print("Inference functions ready (MATCHING V11 TRAINING)")
print(f"  Normalization: Percentile clipping (0.5-99.5%) + Z-score")
print(f"  Patch size: {cfg.INFER_PATCH_SIZE}")

# =============================================================================
# LOAD MODEL WITH MULTI-GPU (DataParallel)
# =============================================================================

# Clear all GPU memory
for i in range(N_GPUS):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
gc.collect()

print(f"Loading: {cfg.CHECKPOINT_PATH}")

# Create model on primary GPU first
model = TopoPreservingUNet3D(features=cfg.FEATURES, n_blocks=cfg.N_BLOCKS).cuda()

# Load checkpoint
ckpt = torch.load(cfg.CHECKPOINT_PATH, map_location='cuda:0', weights_only=False)
state = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model_state_dict'].items()}
model.load_state_dict(state, strict=False)

print(f"  Epoch: {ckpt.get('epoch', '?')}, Best Dice: {ckpt.get('best_dice', '?'):.4f}")

# Wrap with DataParallel for multi-GPU
if N_GPUS > 1 and cfg.USE_MULTI_GPU:
    print(f"\n>>> Enabling DataParallel across {N_GPUS} GPUs")
    model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
    print(f"  Device IDs: {list(range(N_GPUS))}")
else:
    print("\n>>> Single GPU mode")

# Convert to half precision
model.half()
model.eval()

# Memory report
print("\nGPU Memory Usage:")
for i in range(N_GPUS):
    mem = torch.cuda.memory_allocated(i) / 1e9
    print(f"  GPU {i}: {mem:.2f} GB")

print("\nModel ready for inference!")

# =============================================================================
# RUN INFERENCE (MULTI-GPU) - OPTIMIZED POST-PROCESSING
# =============================================================================

if __name__ == '__main__':
    test_files = sorted(cfg.TEST_ROOT.glob("*.tif")) + sorted(cfg.TEST_ROOT.glob("*.tiff"))
    print(f"Found {len(test_files)} test volumes")

    # Create output directory for masks
    mask_dir = cfg.OUTPUT_DIR / "masks"
    mask_dir.mkdir(exist_ok=True)

    for vol_path in test_files:
        vol_id = vol_path.stem
        print(f"\n{'='*70}")
        print(f"Processing: {vol_id}")
        print(f"{'='*70}")
        
        # Clear memory on all GPUs before each volume
        for i in range(N_GPUS):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
        gc.collect()
        
        # Load volume
        volume = tifffile.imread(str(vol_path)).astype(np.float32)
        original_shape = volume.shape
        print(f"Shape: {original_shape}")
        
        # Inference with multi-GPU function
        print(f"Running inference (patch={cfg.INFER_PATCH_SIZE}, TTA={cfg.TTA_LEVEL}, batch={cfg.BATCH_SIZE})...")
        pred_prob = inference_with_tta_multigpu(
            model, volume, cfg.INFER_PATCH_SIZE, cfg.OVERLAP,
            cfg.BATCH_SIZE, cfg.TTA_LEVEL
        )
        
        del volume
        gc.collect()
        
        # OPTIMIZED Post-processing:
        # - NO Frangi (was hurting score)
        # - Fixed threshold 0.5 (not adaptive 0.30)
        # - 2D slicewise morphology
        pred_mask = postprocess_v11(
            pred_prob,
            threshold=0.5,  # Fixed threshold
            min_component_size=50,
            use_morphology=True,
            use_hole_fill=True,
            verbose=True
        )
        
        # Verify shape matches original
        assert pred_mask.shape == original_shape, f"Shape mismatch: {pred_mask.shape} vs {original_shape}"
        
        # Save as TIFF (no compression)
        mask_path = mask_dir / f"{vol_id}.tif"
        save_mask_tiff(pred_mask, mask_path)
        
        # Check file size
        actual_size = mask_path.stat().st_size / 1e6
        print(f"  Saved: {mask_path.name} ({actual_size:.2f} MB)")
        
        # Cleanup
        del pred_prob, pred_mask
        gc.collect()
        for i in range(N_GPUS):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")

    # =============================================================================
    # CREATE SUBMISSION ZIP
    # =============================================================================

    submission_zip = cfg.OUTPUT_DIR / "submission.zip"
    create_submission_zip(mask_dir, submission_zip)

    # Verify submission
    print(f"\nSubmission contents:")
    with zipfile.ZipFile(submission_zip, 'r') as zf:
        for info in zf.infolist():
            print(f"  {info.filename}: {info.file_size / 1e6:.2f} MB")

    print(f"\n>>> Ready to submit: {submission_zip}")