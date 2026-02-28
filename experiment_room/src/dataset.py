"""
Vesuvius Challenge Experiment Lab - Dataset
==========================================
Efficient 3D TIFF volume loading with patch extraction and augmentation.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile


class VesuviusDataset(Dataset):
    """
    Dataset for Vesuvius scroll surface detection.
    
    Features:
    - Efficient LZW-compressed TIFF loading
    - Random patch extraction from 320³ volumes
    - Proper handling of unlabeled regions (label=2)
    - On-the-fly augmentation
    """
    
    def __init__(
        self,
        image_ids: List[str],
        data_root: str,
        images_dir: str,
        labels_dir: str,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        patches_per_volume: int = 8,
        unlabeled_value: int = 2,
        augment: bool = False,
        flip_prob: float = 0.5,
        cache_volumes: bool = False
    ):
        self.image_ids = image_ids
        self.data_root = Path(data_root)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.unlabeled_value = unlabeled_value
        self.augment = augment
        self.flip_prob = flip_prob
        self.cache_volumes = cache_volumes
        
        # Volume cache
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Calculate total samples
        self.total_patches = len(image_ids) * patches_per_volume
    
    def __len__(self) -> int:
        return self.total_patches
    
    def _load_tiff(self, path: Path) -> np.ndarray:
        """Load LZW-compressed multipage TIFF as 3D numpy array."""
        try:
            volume = tifffile.imread(str(path))
            return volume.astype(np.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            raise
    
    def _load_volume_pair(self, image_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and label volumes for given ID."""
        
        if self.cache_volumes and image_id in self._cache:
            return self._cache[image_id]
        
        # Try main directories first, then deprecated
        img_path = self.data_root / self.images_dir / f"{image_id}.tif"
        lbl_path = self.data_root / self.labels_dir / f"{image_id}.tif"
        
        if not img_path.exists():
            # Try deprecated directories
            img_path = self.data_root / "deprecated_train_images" / f"{image_id}.tif"
            lbl_path = self.data_root / "deprecated_train_labels" / f"{image_id}.tif"
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_id}")
        
        image = self._load_tiff(img_path)
        label = self._load_tiff(lbl_path)
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        if self.cache_volumes:
            self._cache[image_id] = (image, label)
        
        return image, label
    
    def _extract_random_patch(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract random patch from volume."""
        
        D, H, W = image.shape
        pd, ph, pw = self.patch_size
        
        # Random start coordinates
        d = random.randint(0, max(0, D - pd))
        h = random.randint(0, max(0, H - ph))
        w = random.randint(0, max(0, W - pw))
        
        # Extract patches
        img_patch = image[d:d+pd, h:h+ph, w:w+pw]
        lbl_patch = label[d:d+pd, h:h+ph, w:w+pw]
        
        # Create mask (1 = valid, 0 = unlabeled/ignore)
        mask = (lbl_patch != self.unlabeled_value).astype(np.float32)
        
        # Convert labels: keep 0/1, set unlabeled (2) to 0 for loss
        binary_label = (lbl_patch == 1).astype(np.float32)
        
        return img_patch, binary_label, mask
    
    def _augment(
        self,
        image: np.ndarray,
        label: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        
        # Random flips along each axis
        for axis in [0, 1, 2]:
            if random.random() < self.flip_prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()
        
        # Random 90-degree rotations in xy plane
        if random.random() < 0.5:
            k = random.randint(1, 3)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        
        # Random intensity augmentation
        if random.random() < 0.3:
            # Brightness
            image = image + random.uniform(-0.1, 0.1)
            image = np.clip(image, 0, 1)
        
        if random.random() < 0.3:
            # Contrast
            factor = random.uniform(0.9, 1.1)
            image = (image - 0.5) * factor + 0.5
            image = np.clip(image, 0, 1)
        
        # Gaussian noise
        if random.random() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        return image, label, mask
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single training sample."""
        
        # Map index to volume and patch
        volume_idx = idx // self.patches_per_volume
        image_id = self.image_ids[volume_idx]
        
        # Load volume pair
        image, label = self._load_volume_pair(image_id)
        
        # Extract random patch
        img_patch, lbl_patch, mask = self._extract_random_patch(image, label)
        
        # Augment if enabled
        if self.augment:
            img_patch, lbl_patch, mask = self._augment(img_patch, lbl_patch, mask)
        
        # Convert to tensors (add channel dimension)
        img_tensor = torch.from_numpy(img_patch).unsqueeze(0)  # [1, D, H, W]
        lbl_tensor = torch.from_numpy(lbl_patch).unsqueeze(0)  # [1, D, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)      # [1, D, H, W]
        
        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'mask': mask_tensor,
            'image_id': image_id
        }


class VesuviusInferenceDataset(Dataset):
    """Dataset for inference with sliding window."""
    
    def __init__(
        self,
        image_path: str,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5
    ):
        self.image_path = Path(image_path)
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Load volume
        self.volume = tifffile.imread(str(image_path)).astype(np.float32)
        self.volume = (self.volume - self.volume.min()) / (self.volume.max() - self.volume.min() + 1e-8)
        
        # Calculate patch positions
        self.positions = self._calculate_positions()
    
    def _calculate_positions(self) -> List[Tuple[int, int, int]]:
        """Calculate all patch starting positions with overlap."""
        positions = []
        D, H, W = self.volume.shape
        pd, ph, pw = self.patch_size
        
        stride_d = int(pd * (1 - self.overlap))
        stride_h = int(ph * (1 - self.overlap))
        stride_w = int(pw * (1 - self.overlap))
        
        for d in range(0, D - pd + 1, stride_d):
            for h in range(0, H - ph + 1, stride_h):
                for w in range(0, W - pw + 1, stride_w):
                    positions.append((d, h, w))
        
        # Ensure coverage of edges
        if D - pd not in [p[0] for p in positions]:
            for h in range(0, H - ph + 1, stride_h):
                for w in range(0, W - pw + 1, stride_w):
                    positions.append((D - pd, h, w))
        
        return positions
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        d, h, w = self.positions[idx]
        pd, ph, pw = self.patch_size
        
        patch = self.volume[d:d+pd, h:h+ph, w:w+pw]
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        
        return {
            'image': patch_tensor,
            'position': (d, h, w)
        }


def create_data_splits(
    csv_path: str,
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create stratified train/val splits based on scroll_id.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    df = pd.read_csv(csv_path)
    
    if max_samples is not None:
        # Stratified sampling
        df = df.groupby('scroll_id').apply(
            lambda x: x.sample(min(len(x), max_samples // df['scroll_id'].nunique()), random_state=seed)
        ).reset_index(drop=True)
        
        if len(df) > max_samples:
            df = df.sample(max_samples, random_state=seed)
    
    # Stratified split by scroll_id
    train_ids = []
    val_ids = []
    
    for scroll_id in df['scroll_id'].unique():
        scroll_samples = df[df['scroll_id'] == scroll_id]['id'].tolist()
        random.shuffle(scroll_samples)
        
        split_idx = int(len(scroll_samples) * train_ratio)
        train_ids.extend(scroll_samples[:split_idx])
        val_ids.extend(scroll_samples[split_idx:])
    
    # Convert to strings
    train_ids = [str(x) for x in train_ids]
    val_ids = [str(x) for x in val_ids]
    
    random.shuffle(train_ids)
    random.shuffle(val_ids)
    
    return train_ids, val_ids


def create_dataloaders(
    config,
    train_ids: List[str],
    val_ids: List[str]
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    train_dataset = VesuviusDataset(
        image_ids=train_ids,
        data_root=config.data.data_root,
        images_dir=config.data.train_images_dir,
        labels_dir=config.data.train_labels_dir,
        patch_size=config.data.patch_size,
        patches_per_volume=config.data.patches_per_volume,
        unlabeled_value=config.data.unlabeled_label,
        augment=config.data.use_augmentation,
        flip_prob=config.data.flip_prob,
        cache_volumes=True
    )
    
    val_dataset = VesuviusDataset(
        image_ids=val_ids,
        data_root=config.data.data_root,
        images_dir=config.data.train_images_dir,
        labels_dir=config.data.train_labels_dir,
        patch_size=config.data.patch_size,
        patches_per_volume=config.data.patches_per_volume,
        unlabeled_value=config.data.unlabeled_label,
        augment=False,
        cache_volumes=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    config.data.data_root = "/Volumes/New_HDD2/KaggleCompetitions/Vesuvius_challenge/vesuvius-challenge-surface-detection"
    
    # Just print expected behavior
    print("Dataset Configuration:")
    print(f"  Patch size: {config.data.patch_size}")
    print(f"  Patches per volume: {config.data.patches_per_volume}")
    print(f"  Max samples: {config.data.max_samples}")
    print(f"  Unlabeled value: {config.data.unlabeled_label}")