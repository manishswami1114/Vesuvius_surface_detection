# =============================================================================
# CELL 1: IMPORTS & CONFIG
# =============================================================================

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gc
import json
import random
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
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

@dataclass
class Config:
    # Data paths
    DATA_ROOT: Path = Path("")
    CHECKPOINT_DIR: Path = Path("")  # Save checkpoints here
    LOAD_CHECKPOINT: Path = Path("")  # Set to checkpoint path to resume training
    
    # ==========================================================================
    # MODEL CONFIG - 6 stage, 192 patch
    # ==========================================================================
    PATCH_SIZE: Tuple[int, int, int] = (192, 192, 192)
    FEATURES: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 320, 320])
    N_BLOCKS: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 6, 6])
    USE_ATTENTION: bool = True
    USE_HYBRID_CONV: bool = True
    USE_SURFACE_REFINEMENT: bool = True
    USE_DEEP_SUPERVISION: bool = True
    
    # ==========================================================================
    # TRAINING CONFIG
    # ==========================================================================
    EPOCHS_PER_FOLD: int = 800
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 16
    PREFETCH_FACTOR: int = 4
    
    # ==========================================================================
    # OPTIMIZER (from report)
    # ==========================================================================
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 1e-2
    WARMUP_EPOCHS: int = 5
    ETA_MIN: float = 1e-6
    GRADIENT_CLIP: float = 1.0
    
    # ==========================================================================
    # LOSS WEIGHTS (ALL from epoch 0)
    # ==========================================================================
    DICE_WEIGHT: float = 0.25
    BCE_WEIGHT: float = 0.10
    CLDICE_WEIGHT: float = 0.30
    SURFACE_WEIGHT: float = 0.15
    TOPO_WEIGHT: float = 0.20
    
    # ==========================================================================
    # AUGMENTATION (mild, surface-preserving)
    # ==========================================================================
    AUG_FLIP: bool = True
    AUG_ROTATE: bool = True
    AUG_ELASTIC: bool = True
    AUG_ELASTIC_SIGMA: float = 2.0
    AUG_AFFINE: bool = True
    AUG_AFFINE_SCALE: Tuple[float, float] = (0.9, 1.1)
    AUG_NOISE: bool = True
    AUG_CONTRAST: bool = True
    AUG_BRIGHTNESS: bool = True
    AUG_OCCLUSION: bool = True
    
    # ==========================================================================
    # VALIDATION & CV
    # ==========================================================================
    VAL_EVERY: int = 5
    SAVE_EVERY: int = 10
    N_FOLDS: int = 3
    CV_SEED: int = 42
    
    # ==========================================================================
    # DATA
    # ==========================================================================
    PATCHES_PER_VOLUME: int = 12
    FG_OVERSAMPLE_RATIO: float = 0.6
    
    # Device & precision
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_BFLOAT16: bool = True
    SEED: int = 42
    
    def __post_init__(self):
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

cfg = Config()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

set_seed(cfg.SEED)

print("="*70)
print("V11 - TOPOLOGY-PRESERVING TRAINING")
print("="*70)
print(f"Patch: {cfg.PATCH_SIZE} | BS: {cfg.BATCH_SIZE} | Epochs: {cfg.EPOCHS_PER_FOLD}")
print(f"Stages: {len(cfg.FEATURES)} | Features: {cfg.FEATURES}")
print("="*70)
print("Loss weights (ALL from epoch 0):")
print(f"  Dice={cfg.DICE_WEIGHT}, BCE={cfg.BCE_WEIGHT}, clDice={cfg.CLDICE_WEIGHT}")
print(f"  Surface={cfg.SURFACE_WEIGHT}, Topo={cfg.TOPO_WEIGHT}")
print("="*70)
print(f"Checkpoint save dir: {cfg.CHECKPOINT_DIR}")
print(f"Load checkpoint: {cfg.LOAD_CHECKPOINT}")
print("="*70)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.is_bf16_supported():
        print("bfloat16: SUPPORTED")
    else:
        print("bfloat16: NOT SUPPORTED - using float16")
        cfg.USE_BFLOAT16 = False
print("="*70)