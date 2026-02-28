"""
Vesuvius Challenge 2025 - V4 Model Inference & Submission
=========================================================
Model: 6-stage ResEncUNet3D with SafeInstanceNorm3d + scSE
Patch Size: 128x128x128
Best Val Dice: 0.6200 (epoch 250)
"""

import os
import sys
import gc
import zipfile
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from scipy import ndimage
from skimage.morphology import remove_small_objects

try:
    import cc3d
    USE_CC3D = True
except ImportError:
    USE_CC3D = False
    print("cc3d not found, using scipy for connected components")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - MATCHES V4 TRAINING EXACTLY
# =============================================================================
class Config:
    TEST_IMAGES_DIR = Path("/kaggle/input/vesuvius-challenge-surface-detection/test_images")
    TEST_CSV = Path("/kaggle/input/vesuvius-challenge-surface-detection/test.csv")

    # V4 Model Checkpoint - UPDATE THIS PATH
    CHECKPOINT_PATH = Path("/kaggle/input/v4-model/pytorch/default/2/checkpoints_v4/best_model.pth")

    OUTPUT_DIR = Path("/kaggle/working/submission_masks")
    SUBMISSION_ZIP = Path("/kaggle/working/submission.zip")

    # ==========================================================================
    # ARCHITECTURE - MUST MATCH V4 TRAINING EXACTLY
    # ==========================================================================
    PATCH_SIZE: Tuple[int, int, int] = (128, 128, 128)  # V4 used 128³
    FEATURES: List[int] = [32, 64, 128, 256, 320, 320]
    N_BLOCKS: List[int] = [1, 3, 4, 6, 6, 6]
    USE_SCSE: bool = True
    USE_DEEP_SUPERVISION: bool = True

    # ==========================================================================
    # INFERENCE SETTINGS
    # ==========================================================================
    OVERLAP: float = 0.5  # 50% overlap for faster inference
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True

    # ==========================================================================
    # POST-PROCESSING
    # ==========================================================================
    FINAL_THRESHOLD: float = 0.5  # Standard threshold
    MIN_COMPONENT_SIZE: int = 50  # Remove small noise

cfg = Config()
cfg.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print("="*70)
print("V4 MODEL INFERENCE")
print("="*70)
print(f"Device: {cfg.DEVICE}")
if cfg.DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
print(f"Patch size: {cfg.PATCH_SIZE}")
print(f"Overlap: {cfg.OVERLAP}")
print(f"Threshold: {cfg.FINAL_THRESHOLD}")
print("="*70)


# =============================================================================
# MODEL ARCHITECTURE - EXACT COPY FROM V4 TRAINING
# Key difference: SafeInstanceNorm3d instead of GroupNorm
# =============================================================================

class SafeInstanceNorm3d(nn.Module):
    """
    Safe InstanceNorm3d that handles edge cases.
    This is the key difference from the GroupNorm model.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super(SafeInstanceNorm3d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True, unbiased=False)
        var_safe = torch.clamp(var, min=self.eps)
        x_norm = (x - mean) / torch.sqrt(var_safe + self.eps)

        if self.affine:
            x_norm = self.weight.view(1, -1, 1, 1, 1) * x_norm + self.bias.view(1, -1, 1, 1, 1)

        return x_norm


class ConvBlock(nn.Module):
    """Conv block with SafeInstanceNorm3d (V4 architecture)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=True),
            SafeInstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, channels, n_convs=2):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ConvBlock(channels, channels) for _ in range(n_convs)]
        )

    def forward(self, x):
        return x + self.blocks(x)


class scSEBlock(nn.Module):
    """Concurrent Spatial and Channel Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.cse_pool = nn.AdaptiveAvgPool3d(1)
        self.cse_fc1 = nn.Linear(channels, channels // reduction)
        self.cse_fc2 = nn.Linear(channels // reduction, channels)
        self.sse_conv = nn.Conv3d(channels, 1, 1)

    def forward(self, x):
        b, c, d, h, w = x.shape
        cse = self.cse_pool(x).view(b, c)
        cse = F.relu(self.cse_fc1(cse))
        cse = torch.sigmoid(self.cse_fc2(cse)).view(b, c, 1, 1, 1)
        sse = torch.sigmoid(self.sse_conv(x))
        return x * cse + x * sse


class ResEncUNet3D(nn.Module):
    """
    V4 Architecture: 6-stage ResEncUNet3D with SafeInstanceNorm3d + scSE
    """
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        features: List[int] = None,
        n_blocks: List[int] = None,
        use_scse: bool = True,
        use_deep_supervision: bool = True,
    ):
        super().__init__()

        if features is None:
            features = [32, 64, 128, 256, 320, 320]
        if n_blocks is None:
            n_blocks = [1, 3, 4, 6, 6, 6]

        self.features = features
        self.use_scse = use_scse
        self.use_deep_supervision = use_deep_supervision
        self.n_stages = len(features)

        # Encoder
        self.encoders = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i, (feat, nb) in enumerate(zip(features, n_blocks)):
            in_channels = in_ch if i == 0 else features[i - 1]
            encoder = nn.Sequential(
                ConvBlock(in_channels, feat),
                *[ResBlock(feat, n_convs=2) for _ in range(nb)]
            )
            self.encoders.append(encoder)

            if use_scse:
                self.attentions.append(scSEBlock(feat))
            else:
                self.attentions.append(nn.Identity())

            if i < len(features) - 1:
                self.pools.append(nn.Conv3d(feat, feat, kernel_size=2, stride=2, bias=True))

        # Decoder
        self.ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        for i in range(len(features) - 2, -1, -1):
            up_feat = features[i + 1]
            out_feat = features[i]
            self.ups.append(nn.ConvTranspose3d(up_feat, out_feat, kernel_size=2, stride=2, bias=True))
            self.dec_convs.append(ConvBlock(out_feat * 2, out_feat))

        self.final = nn.Conv3d(features[0], out_ch, 1, bias=True)

        # Deep supervision heads
        if use_deep_supervision:
            self.ds_heads = nn.ModuleList()
            n_ds_outputs = min(4, len(features) - 1)
            for i in range(n_ds_outputs):
                ds_in_channels = features[-(i + 2)]
                self.ds_heads.append(nn.Conv3d(ds_in_channels, out_ch, 1, bias=True))

    def forward(self, x):
        enc_features = []

        for i, (enc, att) in enumerate(zip(self.encoders, self.attentions)):
            x = enc(x)
            x = att(x)
            enc_features.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)

        enc_features = enc_features[::-1]
        x = enc_features[0]

        ds_outputs = []

        for i, (up, dec) in enumerate(zip(self.ups, self.dec_convs)):
            x = up(x)
            skip = enc_features[i + 1]

            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = dec(x)

            if self.use_deep_supervision and self.training and i < len(self.ds_heads):
                ds_outputs.append(self.ds_heads[i](x))

        out = self.final(x)

        # During inference (.eval() mode), return only main output
        if self.use_deep_supervision and self.training:
            return {'output': out, 'deep': ds_outputs}
        return out


# =============================================================================
# NORMALIZATION - MUST MATCH V4 TRAINING EXACTLY
# =============================================================================
def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization (same as V4 training) - returns float32"""
    vol = volume.astype(np.float32)
    return (vol - vol.mean()) / (vol.std() + 1e-8)


# =============================================================================
# POST-PROCESSING
# =============================================================================
def postprocess(pred_prob, threshold=0.5, min_component_size=50):
    """
    Simple post-processing: threshold + remove small components.
    """
    print(f"  Prob range: [{pred_prob.min():.4f}, {pred_prob.max():.4f}]")
    print(f"  FG% before threshold: {100 * (pred_prob > threshold).mean():.3f}%")

    # Threshold
    binary = (pred_prob > threshold).astype(bool)

    # Remove small objects
    if min_component_size > 0:
        cleaned = remove_small_objects(binary, min_size=min_component_size, connectivity=3)
    else:
        cleaned = binary

    final = cleaned.astype(np.uint8)
    print(f"  FG% after post-processing: {100 * final.mean():.3f}%")

    return final


# =============================================================================
# SLIDING WINDOW INFERENCE
# =============================================================================
@torch.no_grad()
def sliding_window_inference(
    model,
    volume: np.ndarray,
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    overlap: float = 0.5,
    device: str = "cuda",
    use_amp: bool = True,
) -> np.ndarray:
    """Sliding window inference with Gaussian weighting."""
    model.eval()

    D, H, W = volume.shape
    pd, ph, pw = patch_size
    sd, sh, sw = int(pd * (1 - overlap)), int(ph * (1 - overlap)), int(pw * (1 - overlap))

    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    # Gaussian weighting
    sigma = 0.125
    gauss_z = np.exp(-0.5 * ((np.arange(pd) - pd/2) / (pd * sigma)) ** 2)
    gauss_y = np.exp(-0.5 * ((np.arange(ph) - ph/2) / (ph * sigma)) ** 2)
    gauss_x = np.exp(-0.5 * ((np.arange(pw) - pw/2) / (pw * sigma)) ** 2)
    gauss_weight = gauss_z[:, None, None] * gauss_y[None, :, None] * gauss_x[None, None, :]
    gauss_weight = gauss_weight.astype(np.float32)

    # Generate positions
    z_pos = list(range(0, max(1, D - pd + 1), sd))
    if D > pd and (D - pd) not in z_pos:
        z_pos.append(D - pd)
    y_pos = list(range(0, max(1, H - ph + 1), sh))
    if H > ph and (H - ph) not in y_pos:
        y_pos.append(H - ph)
    x_pos = list(range(0, max(1, W - pw + 1), sw))
    if W > pw and (W - pw) not in x_pos:
        x_pos.append(W - pw)

    # Normalize entire volume
    vol_norm = normalize_volume(volume)

    total_patches = len(z_pos) * len(y_pos) * len(x_pos)
    print(f"  Total patches: {total_patches}")

    for z in tqdm(z_pos, desc="Inference", leave=False):
        for y in y_pos:
            for x in x_pos:
                patch = vol_norm[z:z+pd, y:y+ph, x:x+pw].copy()
                actual_shape = patch.shape

                # Pad if needed
                if patch.shape != (pd, ph, pw):
                    pad = [
                        (0, max(0, pd - patch.shape[0])),
                        (0, max(0, ph - patch.shape[1])),
                        (0, max(0, pw - patch.shape[2]))
                    ]
                    patch = np.pad(patch, pad, mode='reflect')

                # CRITICAL: Ensure float32 dtype
                patch = patch.astype(np.float32)
                inp = torch.from_numpy(patch[None, None]).float().to(device)

                with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                    out = model(inp)
                    if isinstance(out, dict):
                        out = out['output']
                    out = torch.sigmoid(out).squeeze().cpu().numpy()

                # Crop to actual size
                out = out[:actual_shape[0], :actual_shape[1], :actual_shape[2]]
                weight = gauss_weight[:actual_shape[0], :actual_shape[1], :actual_shape[2]]

                pred_sum[z:z+out.shape[0], y:y+out.shape[1], x:x+out.shape[2]] += out * weight
                count[z:z+out.shape[0], y:y+out.shape[1], x:x+out.shape[2]] += weight

    return pred_sum / np.maximum(count, 1e-8)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Load test metadata
    test_df = pd.read_csv(cfg.TEST_CSV)
    print(f"\nFound {len(test_df)} test volumes")

    # Initialize V4 model
    print("\nInitializing V4 model (SafeInstanceNorm3d)...")
    model = ResEncUNet3D(
        features=cfg.FEATURES,
        n_blocks=cfg.N_BLOCKS,
        use_scse=cfg.USE_SCSE,
        use_deep_supervision=cfg.USE_DEEP_SUPERVISION,
    )

    # Load weights
    print(f"Loading weights from: {cfg.CHECKPOINT_PATH}")
    if not cfg.CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.CHECKPOINT_PATH}")

    checkpoint = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.DEVICE, weights_only=False)

    # Handle checkpoint format
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'best_score' in checkpoint:
            print(f"  Checkpoint best score: {checkpoint['best_score']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
    else:
        state_dict = checkpoint

    # Clean keys (remove module. and _orig_mod. prefixes from torch.compile)
    cleaned_state = {}
    for k, v in state_dict.items():
        key = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned_state[key] = v

    # Load weights
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model = model.to(cfg.DEVICE).eval()
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded: {param_count:.1f}M parameters")

    # Create submission
    print("\n" + "="*70)
    print("RUNNING V4 INFERENCE")
    print("="*70)
    print(f"Patch size: {cfg.PATCH_SIZE}")
    print(f"Overlap: {cfg.OVERLAP}")
    print(f"Threshold: {cfg.FINAL_THRESHOLD}")
    print(f"Min component size: {cfg.MIN_COMPONENT_SIZE}")
    print("="*70)

    with zipfile.ZipFile(cfg.SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for idx, row in test_df.iterrows():
            image_id = row["id"]
            tif_path = cfg.TEST_IMAGES_DIR / f"{image_id}.tif"

            print(f"\n[{idx+1}/{len(test_df)}] Processing {image_id}...")
            volume = tifffile.imread(str(tif_path))
            print(f"  Volume shape: {volume.shape}")

            # Inference
            prob_map = sliding_window_inference(
                model=model,
                volume=volume,
                patch_size=cfg.PATCH_SIZE,
                overlap=cfg.OVERLAP,
                device=cfg.DEVICE,
                use_amp=cfg.USE_AMP,
            )

            # Post-process
            prediction = postprocess(
                prob_map,
                threshold=cfg.FINAL_THRESHOLD,
                min_component_size=cfg.MIN_COMPONENT_SIZE,
            )

            # Save
            out_path = cfg.OUTPUT_DIR / f"{image_id}.tif"
            tifffile.imwrite(out_path, prediction.astype(np.uint8))
            zf.write(out_path, arcname=f"{image_id}.tif")
            out_path.unlink()

            # Cleanup
            del volume, prob_map, prediction
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(f"Submission created: {cfg.SUBMISSION_ZIP}")
    print("="*70)


if __name__ == "__main__":
    main()
