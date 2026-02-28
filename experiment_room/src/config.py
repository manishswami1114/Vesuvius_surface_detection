"""
Vesuvius Challenge Experiment Lab - Configuration
================================================
Centralized configuration for all experiments.
Optimized for MacBook GPU (MPS) with 64x64x64 patches.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    # Paths - adjust for Kaggle vs local
    data_root: str = "/Volumes/New_HDD2/KaggleCompetitions/Vesuvius_challenge/vesuvius-challenge-surface-detection"
    train_images_dir: str = "train_images"
    train_labels_dir: str = "train_labels"
    deprecated_images_dir: str = "deprecated_train_images"
    deprecated_labels_dir: str = "deprecated_train_labels"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    
    # Volume dimensions
    volume_size: Tuple[int, int, int] = (320, 320, 320)
    
    # Patch extraction
    patch_size: Tuple[int, int, int] = (64, 64, 64)
    patches_per_volume: int = 8
    
    # Class definitions
    num_classes: int = 2
    background_label: int = 0
    surface_label: int = 1
    unlabeled_label: int = 2
    
    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    max_samples: int = 100
    
    # Augmentation
    use_augmentation: bool = True
    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    elastic_prob: float = 0.2


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "unet3d"
    in_channels: int = 1
    out_channels: int = 1
    init_features: int = 32
    depth: int = 4
    use_attention: bool = False
    attention_type: str = "additive"
    use_residual: bool = False
    use_deep_supervision: bool = False
    dropout_rate: float = 0.1
    norm_type: str = "batch"


@dataclass
class LossConfig:
    """Loss function configuration."""
    loss_type: str = "dice"
    dice_weight: float = 0.5
    bce_weight: float = 0.3
    focal_weight: float = 0.2
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    dice_smooth: float = 1e-5
    ds_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 10
    batch_size: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    use_amp: bool = True
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    early_stopping_patience: int = 5
    device: str = "auto"
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "vesuvius_baseline"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    log_dir: str = "./logs"
    vis_dir: str = "./visualizations"
    log_interval: int = 10
    vis_interval: int = 1
    
    def get_device(self) -> torch.device:
        """Automatically detect best available device."""
        if self.training.device == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(self.training.device)
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment identifier."""
        return f"{self.experiment_name}_{self.model.architecture}_{self.loss.loss_type}"