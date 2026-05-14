import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import json
from scipy.ndimage import distance_transform_edt

# =============================================================================
# CELL 6: DATASET WITH ROBUST NORMALIZATION
# =============================================================================

def load_volume(path):
    try:
        import tifffile
        return tifffile.imread(str(path))
    except:
        im = Image.open(path)
        return np.stack([np.array(p) for p in ImageSequence.Iterator(im)], axis=0)


def robust_zscore_normalize(img, lower_percentile=0.5, upper_percentile=99.5):
    """
    Robust Z-score normalization with percentile clipping.
    
    This is the standard in medical imaging (nnU-Net, MONAI):
    1. Clip outliers using percentiles (handles artifacts, noise)
    2. Z-score normalize using clipped statistics
    
    Why this works better:
    - CT/X-ray data often has outliers (bright artifacts, air pockets)
    - Percentile clipping removes these before computing mean/std
    - Results in more stable training and better generalization
    """
    # Compute percentiles for clipping
    p_low = np.percentile(img, lower_percentile)
    p_high = np.percentile(img, upper_percentile)
    
    # Clip to remove outliers
    img_clipped = np.clip(img, p_low, p_high)
    
    # Compute statistics on clipped data (more robust)
    mean = img_clipped.mean()
    std = img_clipped.std()
    
    # Normalize (use clipped image)
    img_norm = (img_clipped - mean) / (std + 1e-8)
    
    return img_norm.astype(np.float32)


VOLUME_CACHE = {}
FG_COORDS_CACHE = {}

def preload_volumes(volume_ids, images_dir, labels_dir):
    """Preload volumes with robust normalization."""
    global VOLUME_CACHE, FG_COORDS_CACHE
    to_load = [vid for vid in volume_ids if vid not in VOLUME_CACHE]
    if not to_load:
        print(f"All {len(volume_ids)} volumes cached")
        return
    print(f"Preloading {len(to_load)} volumes with robust Z-score normalization...")
    for vid in tqdm(to_load, desc="Loading"):
        img = load_volume(Path(images_dir) / f"{vid}.tif").astype(np.float32)
        lbl = load_volume(Path(labels_dir) / f"{vid}.tif").astype(np.uint8)
        
        # Robust normalization (percentile clipping + Z-score)
        img = robust_zscore_normalize(img, lower_percentile=0.5, upper_percentile=99.5)
        
        VOLUME_CACHE[vid] = (img, lbl)
        fg = np.argwhere(lbl == 1)
        FG_COORDS_CACHE[vid] = fg[np.random.choice(len(fg), min(10000, len(fg)), replace=False)] if len(fg) > 0 else None
    print(f"Cached: {len(VOLUME_CACHE)} volumes ({sum(v[0].nbytes+v[1].nbytes for v in VOLUME_CACHE.values())/1e9:.1f} GB)")


class VesuviusDatasetV11(Dataset):
    def __init__(self, volume_ids, images_dir, labels_dir, patch_size=(192,192,192),
                 is_train=True, patches_per_volume=12, fg_oversample=0.6):
        self.patch_size = patch_size
        self.is_train = is_train
        self.patches_per_volume = patches_per_volume
        self.fg_oversample = fg_oversample
        self.volume_ids = volume_ids
        preload_volumes(volume_ids, images_dir, labels_dir)
        print(f"Dataset: {len(self)} samples ({'train' if is_train else 'val'})")
    
    def __len__(self):
        return len(self.volume_ids) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vid = self.volume_ids[idx // self.patches_per_volume]
        img, lbl = VOLUME_CACHE[vid]
        d, h, w = img.shape
        pd, ph, pw = self.patch_size
        
        if d < pd or h < ph or w < pw:
            img = np.pad(img, ((0, max(0,pd-d)), (0, max(0,ph-h)), (0, max(0,pw-w))), mode='reflect')
            lbl = np.pad(lbl, ((0, max(0,pd-d)), (0, max(0,ph-h)), (0, max(0,pw-w))), mode='constant', constant_values=2)
            d, h, w = img.shape
        
        fg = FG_COORDS_CACHE.get(vid)
        if self.is_train and random.random() < self.fg_oversample and fg is not None and len(fg) > 0:
            c = fg[random.randint(0, len(fg)-1)]
            z, y, x = [max(0, min(c[i] - self.patch_size[i]//2, s - self.patch_size[i])) for i, s in enumerate([d,h,w])]
        else:
            z, y, x = [random.randint(0, max(0, s - p)) for s, p in zip([d,h,w], self.patch_size)]
        
        img_p = img[z:z+pd, y:y+ph, x:x+pw].copy()
        lbl_p = lbl[z:z+pd, y:y+ph, x:x+pw].copy()
        
        return {
            'image': torch.from_numpy(img_p).unsqueeze(0).float(),
            'label': torch.from_numpy((lbl_p == 1).astype(np.float32)).unsqueeze(0),
            'ignore_mask': torch.from_numpy((lbl_p == 2).astype(np.float32)).unsqueeze(0),
        }

print("VesuviusDatasetV11 ready with ROBUST NORMALIZATION")
print("Normalization: Percentile clipping (0.5-99.5%) + Z-score")
print("  - Removes outliers/artifacts before normalization")
print("  - More stable training, better generalization")