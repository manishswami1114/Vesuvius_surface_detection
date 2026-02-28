"""
Vesuvius Challenge - Optimized Submission Notebook
===================================================

This notebook performs inference with:
1. Sliding window inference (Gaussian weighting)
2. Test-Time Augmentation (TTA)
3. Topology-preserving post-processing
4. Ensemble support (multiple models)

Expected runtime: ~120 volumes in hidden test set
"""

import glob
import os
import sys
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import ndimage
import tifffile

os.environ["KERAS_BACKEND"] = "torch"
warnings.filterwarnings('ignore')

import keras
from keras import ops
import torch
import tensorflow as tf

# Mixed precision for faster inference
keras.mixed_precision.set_global_policy("mixed_float16")

import medicai
from medicai.transforms import Compose, NormalizeIntensity
from medicai.models import SegFormer, SwinUNETR
from medicai.utils import SlidingWindowInference


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model - should match training
    input_shape = (192, 192, 192)
    num_classes = 3

    # Inference settings
    sw_overlap = 0.5  # Overlap for sliding window
    sw_batch_size = 2  # Can increase for H100

    # TTA settings
    use_tta = True
    tta_flips = [0, 1, 2]  # Flip along these axes

    # Post-processing
    min_component_size = 1000
    fill_holes_size = 500

    # Ensemble
    use_ensemble = False
    model_paths = [
        '/kaggle/input/your-model/best_model.weights.h5',
        # Add more model paths for ensemble
    ]

    # Paths
    test_dir = '/kaggle/input/vesuvius-3d-segmentation/test_images'
    output_dir = '/kaggle/working/submission'

config = Config()


# ============================================================================
# POST-PROCESSING
# ============================================================================

def remove_small_components(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove connected components smaller than min_size.

    Why this helps VOI and TopoScore:
    - Small disconnected components are often noise
    - They cause over-segmentation (hurts VOI_split)
    - They add extra k=0 features (hurts TopoScore)

    Parameters:
    - min_size: Components smaller than this are removed
    """
    labeled, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return binary_mask

    component_sizes = ndimage.sum(
        binary_mask, labeled, range(1, num_features + 1)
    )

    # Create mask for components to keep
    keep_mask = np.zeros_like(binary_mask)
    for i, size in enumerate(component_sizes, start=1):
        if size >= min_size:
            keep_mask[labeled == i] = 1

    return keep_mask


def fill_small_holes(binary_mask: np.ndarray, max_hole_size: int) -> np.ndarray:
    """
    Fill holes smaller than max_hole_size.

    Why this helps TopoScore:
    - Small holes add extra k=2 features (cavities)
    - Filling them improves Betti matching

    Caveat:
    - May slightly hurt SurfaceDice if holes are real
    - Tune max_hole_size on validation data
    """
    # Invert to make holes foreground
    inverted = 1 - binary_mask

    # Find connected components (holes)
    labeled_holes, num_holes = ndimage.label(inverted)

    if num_holes == 0:
        return binary_mask

    hole_sizes = ndimage.sum(
        inverted, labeled_holes, range(1, num_holes + 1)
    )

    # Fill small holes
    filled = binary_mask.copy()
    for i, size in enumerate(hole_sizes, start=1):
        if size <= max_hole_size:
            filled[labeled_holes == i] = 1

    return filled


def morphological_cleanup(binary_mask: np.ndarray) -> np.ndarray:
    """
    Light morphological cleanup to smooth boundaries.

    Why this helps SurfaceDice:
    - Removes tiny protrusions (noise)
    - Smooths surface for better distance matching

    Uses small structuring element (3³) to minimize impact.
    """
    struct = ndimage.generate_binary_structure(3, 1)

    # Open: erosion followed by dilation (removes small protrusions)
    opened = ndimage.binary_opening(binary_mask, struct, iterations=1)

    # Close: dilation followed by erosion (fills small gaps)
    closed = ndimage.binary_closing(opened, struct, iterations=1)

    return closed.astype(np.uint8)


def post_process(pred_mask: np.ndarray) -> np.ndarray:
    """
    Full post-processing pipeline.

    Order matters:
    1. Remove small components (topology cleanup)
    2. Fill small holes (topology cleanup)
    3. Morphological cleanup (boundary smoothing)
    """
    # Ensure binary
    binary = (pred_mask > 0).astype(np.uint8)

    # Step 1: Remove small components
    binary = remove_small_components(binary, config.min_component_size)

    # Step 2: Fill small holes
    binary = fill_small_holes(binary, config.fill_holes_size)

    # Step 3: Morphological cleanup (optional, can skip if hurts)
    # binary = morphological_cleanup(binary)

    return binary.astype(np.uint8)


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Z-score normalization for a volume."""
    # Only normalize non-zero voxels
    mask = volume > 0
    if mask.any():
        mean = volume[mask].mean()
        std = volume[mask].std()
        if std > 0:
            volume = (volume - mean) / std
    return volume


def load_test_volume(path: str) -> np.ndarray:
    """Load a test volume from TIFF."""
    volume = tifffile.imread(path)
    return volume


def predict_single_volume(
    model,
    volume: np.ndarray,
    swi: SlidingWindowInference,
    use_tta: bool = True
) -> np.ndarray:
    """
    Predict on a single volume with optional TTA.

    TTA Strategy:
    - Average predictions from flipped versions
    - Flips along D, H, W axes
    - More TTA = better accuracy but slower

    Returns:
    - Class prediction (argmax of probabilities)
    """
    # Add batch and channel dimensions: (D, H, W) -> (1, D, H, W, 1)
    x = volume[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Normalize
    x = normalize_volume(x)

    predictions = []

    # Original prediction
    pred = swi(x)  # (1, D, H, W, num_classes)
    predictions.append(pred)

    if use_tta:
        for axis in config.tta_flips:
            # Flip input (adjust axis for batch and channel dims)
            x_flip = np.flip(x, axis=axis + 1)

            # Predict
            pred_flip = swi(x_flip)

            # Flip prediction back
            pred_flip = np.flip(pred_flip, axis=axis + 1)
            predictions.append(pred_flip)

    # Average predictions
    avg_pred = np.mean(predictions, axis=0)

    # Get class prediction
    pred_class = np.argmax(avg_pred[0], axis=-1).astype(np.uint8)

    return pred_class


def predict_ensemble(
    models: list,
    swi_list: list,
    volume: np.ndarray,
    use_tta: bool = True
) -> np.ndarray:
    """
    Ensemble prediction from multiple models.

    Why ensemble helps:
    - Different models have different error patterns
    - Averaging reduces individual model errors
    - Typically +1-2% improvement

    Strategy: Average softmax probabilities, then argmax
    """
    x = volume[np.newaxis, ..., np.newaxis].astype(np.float32)
    x = normalize_volume(x)

    all_probs = []

    for model, swi in zip(models, swi_list):
        predictions = []

        # Original
        pred = swi(x)
        predictions.append(pred)

        if use_tta:
            for axis in config.tta_flips:
                x_flip = np.flip(x, axis=axis + 1)
                pred_flip = swi(x_flip)
                pred_flip = np.flip(pred_flip, axis=axis + 1)
                predictions.append(pred_flip)

        # Average TTA predictions for this model
        avg_pred = np.mean(predictions, axis=0)
        all_probs.append(avg_pred)

    # Ensemble: average across models
    ensemble_pred = np.mean(all_probs, axis=0)
    pred_class = np.argmax(ensemble_pred[0], axis=-1).astype(np.uint8)

    return pred_class


# ============================================================================
# MAIN SUBMISSION
# ============================================================================

def create_model(model_type='segformer_b3'):
    """Create model (must match training configuration)."""
    input_shape_with_channel = config.input_shape + (1,)

    if model_type == 'segformer_b3':
        model = SegFormer(
            input_shape=input_shape_with_channel,
            encoder_name='mit_b3',
            classifier_activation='softmax',
            num_classes=config.num_classes,
            dropout=0.0,  # No dropout at inference
        )
    elif model_type == 'swinunetr':
        model = SwinUNETR(
            input_shape=input_shape_with_channel,
            num_classes=config.num_classes,
            feature_size=48,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            dropout=0.0,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    """Main submission pipeline."""
    print("Vesuvius Challenge - Inference Pipeline")
    print("=" * 50)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load test metadata
    test_csv = pd.read_csv('/kaggle/input/vesuvius-3d-segmentation/test.csv')
    print(f"Number of test volumes: {len(test_csv)}")

    # Create model and load weights
    print("\nLoading model...")
    model = create_model('segformer_b3')

    # Try to load weights (adjust path for your submission)
    weight_path = config.model_paths[0] if config.model_paths else 'model.weights.h5'
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Weight file not found at {weight_path}")

    # Create sliding window inference
    swi = SlidingWindowInference(
        model,
        num_classes=config.num_classes,
        roi_size=config.input_shape,
        sw_batch_size=config.sw_batch_size,
        overlap=config.sw_overlap,
        mode='gaussian',
    )

    # Process each test volume
    print("\nProcessing test volumes...")
    for idx, row in tqdm(test_csv.iterrows(), total=len(test_csv)):
        volume_id = row['id']
        volume_path = f"{config.test_dir}/{volume_id}.tif"

        try:
            # Load volume
            volume = load_test_volume(volume_path)
            original_shape = volume.shape

            # Predict
            pred = predict_single_volume(model, volume, swi, use_tta=config.use_tta)

            # Post-process
            pred = post_process(pred)

            # Ensure shape matches (should already, but safety check)
            if pred.shape != original_shape:
                print(f"Warning: Shape mismatch for {volume_id}. Resizing...")
                pred = ndimage.zoom(
                    pred,
                    [o/p for o, p in zip(original_shape, pred.shape)],
                    order=0  # Nearest neighbor for labels
                )

            # Save prediction
            output_path = f"{config.output_dir}/{volume_id}.tif"
            tifffile.imwrite(output_path, pred.astype(np.uint8))

        except Exception as e:
            print(f"Error processing {volume_id}: {e}")
            # Save empty prediction as fallback
            empty_pred = np.zeros(volume.shape, dtype=np.uint8)
            output_path = f"{config.output_dir}/{volume_id}.tif"
            tifffile.imwrite(output_path, empty_pred)

    print(f"\nPredictions saved to {config.output_dir}/")

    # Create submission CSV
    submission_files = glob.glob(f"{config.output_dir}/*.tif")
    submission_df = pd.DataFrame({
        'id': [os.path.basename(f).replace('.tif', '') for f in submission_files],
        'tif_paths': submission_files
    })
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission CSV created!")


if __name__ == "__main__":
    main()
