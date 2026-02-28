# =============================================================================
# Convert TIFF files to NPY for faster loading
# =============================================================================
# Run this ONCE before training:
#   python convert_tiff_to_npy.py
#
# This will create .npy files alongside .tif files
# NPY loading is ~5-10x faster than TIFF

import os
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

def convert_tiff_to_npy(data_root: str):
    """Convert all TIFF files to NPY format."""
    data_root = Path(data_root)

    # Convert train images
    images_dir = data_root / "train_images"
    labels_dir = data_root / "train_labels"

    for directory in [images_dir, labels_dir]:
        if not directory.exists():
            print(f"Directory not found: {directory}")
            continue

        tiff_files = list(directory.glob("*.tif"))
        print(f"\nConverting {len(tiff_files)} files in {directory.name}...")

        for tif_path in tqdm(tiff_files, desc=directory.name):
            npy_path = tif_path.with_suffix('.npy')

            # Skip if already converted
            if npy_path.exists():
                continue

            # Load TIFF
            data = tifffile.imread(str(tif_path))

            # Save as NPY (uncompressed for fast loading)
            np.save(str(npy_path), data)

        # Verify
        npy_files = list(directory.glob("*.npy"))
        print(f"  Created {len(npy_files)} NPY files")


if __name__ == "__main__":
    # Kaggle path
    kaggle_path = "/kaggle/input/3d-volume-training-data"

    # Local path (if running locally)
    local_path = "/Users/manishswami/developer/Vesuvius-inference/data"

    # Try Kaggle first, then local
    if os.path.exists(kaggle_path):
        convert_tiff_to_npy(kaggle_path)
    elif os.path.exists(local_path):
        convert_tiff_to_npy(local_path)
    else:
        print("Data directory not found!")
        print("Please set the correct path in the script.")
