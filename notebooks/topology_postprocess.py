"""
Vesuvius Challenge 2025 - Topology-Optimized Post-Processing

Target: Break 0.550 LB barrier through post-processing only.

LB Score = 0.30 × TopoScore + 0.35 × SurfaceDice + 0.35 × VOI_score

Key insight: Val Dice 0.60 → LB 0.550, Val Dice 0.63 → LB 0.543 (WORSE!)
Dice optimization hurts topology. Post-processing must preserve topology.

This script provides multiple strategies:
1. Skeleton-preserving threshold (keeps thin connections)
2. VOI-aware component filtering (fixes split/merge)
3. Hysteresis thresholding (preserves connectivity)
4. Topology-aware hole filling (doesn't destroy structure)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
from skimage.morphology import skeletonize_3d, remove_small_objects, remove_small_holes

try:
    import cc3d
    USE_CC3D = True
except ImportError:
    USE_CC3D = False


def connected_components_3d(volume, connectivity=26):
    """3D connected components with cc3d or scipy fallback."""
    if USE_CC3D:
        return cc3d.connected_components(volume.astype(np.uint8), connectivity=connectivity)
    else:
        if connectivity == 26:
            struct = ndimage.generate_binary_structure(3, 3)
        elif connectivity == 6:
            struct = ndimage.generate_binary_structure(3, 1)
        else:
            struct = ndimage.generate_binary_structure(3, 2)
        labeled, _ = ndimage.label(volume, structure=struct)
        return labeled


# =============================================================================
# STRATEGY 1: Hysteresis Thresholding (Best for preserving connections)
# =============================================================================

def hysteresis_threshold_3d(prob_map, low=0.2, high=0.5):
    """
    Hysteresis thresholding: keeps weak voxels if connected to strong ones.

    This preserves thin connections that would be lost with simple thresholding.
    Critical for VOI score (prevents over-segmentation/breaking).

    Args:
        prob_map: Probability map from model [0, 1]
        low: Lower threshold (keep if connected to high)
        high: Upper threshold (definitely foreground)

    Returns:
        Binary mask
    """
    # Strong foreground (definitely keep)
    high_mask = prob_map >= high

    # Weak foreground (keep if connected to strong)
    low_mask = prob_map >= low

    # Label connected components in low_mask
    labeled = connected_components_3d(low_mask)

    # Find which components contain high-confidence voxels
    labels_with_high = np.unique(labeled[high_mask])
    labels_with_high = labels_with_high[labels_with_high > 0]

    # Keep only components that contain high-confidence voxels
    result = np.isin(labeled, labels_with_high)

    return result.astype(np.uint8)


# =============================================================================
# STRATEGY 2: Skeleton-Preserving Post-Processing
# =============================================================================

def skeleton_preserving_postprocess(
    prob_map,
    threshold=0.3,
    skeleton_threshold=0.15,
    max_skeleton_distance=3,
    min_component_size=50,
):
    """
    Post-processing that preserves skeleton connectivity.

    Key idea: Extract skeleton at lower threshold, then keep predictions
    that are close to skeleton. This prevents breaking thin connections.

    Args:
        prob_map: Probability map [0, 1]
        threshold: Main binarization threshold
        skeleton_threshold: Lower threshold for skeleton extraction
        max_skeleton_distance: Max distance from skeleton to keep
        min_component_size: Remove components smaller than this

    Returns:
        Binary mask
    """
    # Step 1: Extract skeleton at lower threshold (more connectivity)
    skeleton_base = prob_map >= skeleton_threshold
    skeleton = skeletonize_3d(skeleton_base)

    if skeleton.sum() == 0:
        # Fallback to simple threshold if skeleton is empty
        return (prob_map >= threshold).astype(np.uint8)

    # Step 2: Compute distance from skeleton
    skeleton_dist = distance_transform_edt(~skeleton)

    # Step 3: Main threshold
    binary = prob_map >= threshold

    # Step 4: Keep only voxels near skeleton
    near_skeleton = skeleton_dist <= max_skeleton_distance
    result = binary & near_skeleton

    # Step 5: Also add skeleton itself (ensures connectivity)
    result = result | skeleton

    # Step 6: Remove small components
    if min_component_size > 0:
        result = remove_small_objects(result, min_size=min_component_size, connectivity=3)

    return result.astype(np.uint8)


# =============================================================================
# STRATEGY 3: VOI-Aware Component Filtering
# =============================================================================

def voi_aware_postprocess(
    prob_map,
    threshold=0.3,
    min_component_size=100,
    merge_distance=2,
    size_ratio_threshold=0.01,
):
    """
    Post-processing optimized for VOI score.

    VOI measures:
    - VOI_split (over-segmentation): breaking one component into many
    - VOI_merge (under-segmentation): merging separate components

    This function:
    1. Removes tiny components (reduces VOI_split from noise)
    2. Merges close components that are likely the same structure

    Args:
        prob_map: Probability map [0, 1]
        threshold: Binarization threshold
        min_component_size: Remove components smaller than this
        merge_distance: Merge components within this distance
        size_ratio_threshold: Remove components smaller than this ratio of largest

    Returns:
        Binary mask
    """
    # Step 1: Threshold
    binary = prob_map >= threshold

    # Step 2: Label components
    labeled = connected_components_3d(binary)
    n_components = labeled.max()

    if n_components == 0:
        return binary.astype(np.uint8)

    # Step 3: Get component sizes
    component_sizes = {}
    for i in range(1, n_components + 1):
        component_sizes[i] = (labeled == i).sum()

    max_size = max(component_sizes.values())

    # Step 4: Remove tiny components (absolute and relative)
    result = np.zeros_like(binary)
    kept_labels = []

    for label, size in component_sizes.items():
        # Keep if above absolute threshold AND relative threshold
        if size >= min_component_size and size >= max_size * size_ratio_threshold:
            result[labeled == label] = 1
            kept_labels.append(label)

    # Step 5: Merge close components (reduces VOI_split from fragmentation)
    if merge_distance > 0 and len(kept_labels) > 1:
        # Dilate to merge nearby components
        dilated = binary_dilation(result, iterations=merge_distance)
        # Re-label
        relabeled = connected_components_3d(dilated)
        # Erode back
        result = binary_erosion(dilated, iterations=merge_distance)
        # Intersect with original to not over-expand
        result = result & (prob_map >= threshold * 0.8)

    return result.astype(np.uint8)


# =============================================================================
# STRATEGY 4: Multi-Scale Consensus
# =============================================================================

def multiscale_consensus_postprocess(
    prob_map,
    thresholds=[0.2, 0.3, 0.4, 0.5],
    consensus_ratio=0.5,
    min_component_size=50,
):
    """
    Use multiple thresholds and keep voxels that appear in most.

    This creates robust predictions that are less sensitive to threshold choice.
    Good for Surface Dice (smoother boundaries).

    Args:
        prob_map: Probability map [0, 1]
        thresholds: List of thresholds to use
        consensus_ratio: Keep if appears in this fraction of thresholds
        min_component_size: Remove small components

    Returns:
        Binary mask
    """
    # Count how many thresholds include each voxel
    vote_count = np.zeros_like(prob_map, dtype=np.int32)

    for thresh in thresholds:
        vote_count += (prob_map >= thresh).astype(np.int32)

    # Keep voxels that appear in enough thresholds
    min_votes = int(len(thresholds) * consensus_ratio)
    result = vote_count >= min_votes

    # Remove small components
    if min_component_size > 0:
        result = remove_small_objects(result, min_size=min_component_size, connectivity=3)

    return result.astype(np.uint8)


# =============================================================================
# STRATEGY 5: Topology-Preserving Hole Filling
# =============================================================================

def topology_safe_hole_fill(
    binary_mask,
    max_hole_size=1000,
    preserve_tunnels=True,
):
    """
    Fill holes without destroying topology.

    Key insight: binary_closing can destroy thin structures.
    This method only fills small enclosed cavities.

    Args:
        binary_mask: Binary mask (0/1)
        max_hole_size: Only fill holes smaller than this
        preserve_tunnels: Don't fill holes that might be real tunnels

    Returns:
        Filled mask
    """
    result = binary_mask.astype(bool)

    # Find holes (background components that don't touch border)
    background = ~result
    labeled_bg = connected_components_3d(background)

    # Find which labels touch the border
    border_labels = set()
    border_labels.update(np.unique(labeled_bg[0, :, :]))
    border_labels.update(np.unique(labeled_bg[-1, :, :]))
    border_labels.update(np.unique(labeled_bg[:, 0, :]))
    border_labels.update(np.unique(labeled_bg[:, -1, :]))
    border_labels.update(np.unique(labeled_bg[:, :, 0]))
    border_labels.update(np.unique(labeled_bg[:, :, -1]))

    # Fill only small internal holes
    for label in range(1, labeled_bg.max() + 1):
        if label in border_labels:
            continue  # Skip - touches border (not enclosed)

        hole_mask = labeled_bg == label
        hole_size = hole_mask.sum()

        if hole_size <= max_hole_size:
            if preserve_tunnels:
                # Check if hole might be a tunnel (elongated shape)
                # Tunnels have high surface-to-volume ratio
                surface = hole_mask ^ binary_erosion(hole_mask)
                surface_ratio = surface.sum() / (hole_size + 1e-8)

                # Spherical hole: ratio ~4.84/r, Tunnel: much higher ratio
                # Don't fill if likely a tunnel
                if surface_ratio < 1.0:  # Likely enclosed cavity, not tunnel
                    result[hole_mask] = True
            else:
                result[hole_mask] = True

    return result.astype(np.uint8)


# =============================================================================
# COMBINED STRATEGY (Recommended)
# =============================================================================

def topology_optimized_postprocess(
    prob_map,
    # Thresholding
    threshold=0.3,
    use_hysteresis=True,
    hysteresis_low=0.15,
    hysteresis_high=0.4,
    # Component filtering
    min_component_size=50,
    size_ratio_threshold=0.005,
    # Skeleton preservation
    use_skeleton=True,
    skeleton_threshold=0.1,
    max_skeleton_distance=5,
    # Hole filling
    fill_holes=False,
    max_hole_size=500,
    # Output
    verbose=True,
):
    """
    Combined topology-optimized post-processing.

    Recommended settings for breaking 0.550 LB:
    - threshold=0.3 (your finding)
    - use_hysteresis=True (preserves connections)
    - use_skeleton=True (preserves thin structures)
    - fill_holes=False (can hurt more than help)

    Args:
        prob_map: Probability map from model [0, 1]
        ... (see individual parameters)

    Returns:
        Binary mask optimized for LB score
    """
    if verbose:
        print(f"  Input prob range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")

    # Step 1: Thresholding
    if use_hysteresis:
        binary = hysteresis_threshold_3d(prob_map, low=hysteresis_low, high=hysteresis_high)
        if verbose:
            print(f"  After hysteresis ({hysteresis_low}/{hysteresis_high}): {100*binary.mean():.2f}% FG")
    else:
        binary = (prob_map >= threshold).astype(np.uint8)
        if verbose:
            print(f"  After threshold ({threshold}): {100*binary.mean():.2f}% FG")

    # Step 2: Skeleton-guided refinement
    if use_skeleton:
        # Extract skeleton at even lower threshold
        skeleton_base = prob_map >= skeleton_threshold
        skeleton = skeletonize_3d(skeleton_base)

        if skeleton.sum() > 0:
            skeleton_dist = distance_transform_edt(~skeleton)
            near_skeleton = skeleton_dist <= max_skeleton_distance

            # Keep: (thresholded AND near skeleton) OR skeleton itself
            binary = ((binary > 0) & near_skeleton) | skeleton

            if verbose:
                print(f"  After skeleton filtering (d≤{max_skeleton_distance}): {100*binary.mean():.2f}% FG")

    # Step 3: Connected component filtering
    labeled = connected_components_3d(binary)
    n_before = labeled.max()

    if n_before > 0:
        sizes = {}
        for i in range(1, n_before + 1):
            sizes[i] = (labeled == i).sum()

        max_size = max(sizes.values())

        result = np.zeros_like(binary, dtype=np.uint8)
        for label, size in sizes.items():
            if size >= min_component_size and size >= max_size * size_ratio_threshold:
                result[labeled == label] = 1

        binary = result
        n_after = connected_components_3d(binary).max()

        if verbose:
            print(f"  Component filtering: {n_before} → {n_after} components")
            print(f"  After filtering: {100*binary.mean():.2f}% FG")

    # Step 4: Topology-safe hole filling (optional)
    if fill_holes:
        before = binary.sum()
        binary = topology_safe_hole_fill(binary, max_hole_size=max_hole_size)
        after = binary.sum()
        if verbose:
            print(f"  Hole filling: +{after-before} voxels")

    if verbose:
        print(f"  Final: {100*binary.mean():.2f}% FG, {connected_components_3d(binary).max()} components")

    return binary.astype(np.uint8)


# =============================================================================
# GRID SEARCH FOR OPTIMAL PARAMETERS
# =============================================================================

def grid_search_postprocess(
    prob_map,
    ground_truth,
    ignore_mask=None,
    thresholds=[0.2, 0.25, 0.3, 0.35, 0.4],
    hysteresis_lows=[0.1, 0.15, 0.2],
    hysteresis_highs=[0.35, 0.4, 0.45, 0.5],
    min_sizes=[25, 50, 100],
    use_skeleton_options=[True, False],
    verbose=True,
):
    """
    Grid search to find optimal post-processing parameters.

    Run this on validation data to find best settings for LB.

    Args:
        prob_map: Model probability output
        ground_truth: Binary ground truth (0, 1, with 2=ignore)
        ignore_mask: Optional explicit ignore mask
        ... parameter ranges

    Returns:
        Best parameters and results
    """
    from itertools import product

    # Create binary GT and ignore mask
    gt_binary = (ground_truth == 1).astype(np.uint8)
    if ignore_mask is None:
        ignore_mask = (ground_truth == 2)

    def compute_dice(pred, gt, ignore):
        pred_valid = pred.copy()
        gt_valid = gt.copy()
        pred_valid[ignore] = 0
        gt_valid[ignore] = 0

        inter = (pred_valid & gt_valid).sum()
        union = pred_valid.sum() + gt_valid.sum()
        return (2 * inter + 1e-5) / (union + 1e-5)

    best_dice = -1
    best_params = {}
    results = []

    # Test hysteresis
    for low, high in product(hysteresis_lows, hysteresis_highs):
        if low >= high:
            continue
        for min_size, use_skel in product(min_sizes, use_skeleton_options):
            pred = topology_optimized_postprocess(
                prob_map,
                use_hysteresis=True,
                hysteresis_low=low,
                hysteresis_high=high,
                min_component_size=min_size,
                use_skeleton=use_skel,
                verbose=False,
            )

            dice = compute_dice(pred, gt_binary, ignore_mask)
            n_comp = connected_components_3d(pred).max()

            result = {
                'method': 'hysteresis',
                'low': low,
                'high': high,
                'min_size': min_size,
                'use_skeleton': use_skel,
                'dice': dice,
                'n_components': n_comp,
                'fg_ratio': pred.mean(),
            }
            results.append(result)

            if dice > best_dice:
                best_dice = dice
                best_params = result.copy()

    # Test simple threshold
    for thresh, min_size, use_skel in product(thresholds, min_sizes, use_skeleton_options):
        pred = topology_optimized_postprocess(
            prob_map,
            threshold=thresh,
            use_hysteresis=False,
            min_component_size=min_size,
            use_skeleton=use_skel,
            verbose=False,
        )

        dice = compute_dice(pred, gt_binary, ignore_mask)
        n_comp = connected_components_3d(pred).max()

        result = {
            'method': 'threshold',
            'threshold': thresh,
            'min_size': min_size,
            'use_skeleton': use_skel,
            'dice': dice,
            'n_components': n_comp,
            'fg_ratio': pred.mean(),
        }
        results.append(result)

        if dice > best_dice:
            best_dice = dice
            best_params = result.copy()

    if verbose:
        print(f"\nBest result: Dice = {best_dice:.4f}")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

    return best_params, results


# =============================================================================
# MAIN - Example usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Topology-optimized post-processing")
    parser.add_argument("--prob", type=str, required=True, help="Probability map TIF")
    parser.add_argument("--output", type=str, required=True, help="Output binary TIF")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--hysteresis", action="store_true", help="Use hysteresis thresholding")
    parser.add_argument("--hysteresis-low", type=float, default=0.15)
    parser.add_argument("--hysteresis-high", type=float, default=0.4)
    parser.add_argument("--skeleton", action="store_true", help="Use skeleton preservation")
    parser.add_argument("--min-size", type=int, default=50)

    args = parser.parse_args()

    import tifffile

    print(f"Loading {args.prob}...")
    prob = tifffile.imread(args.prob).astype(np.float32)

    # Normalize if needed (check if already 0-1)
    if prob.max() > 1.0:
        prob = prob / prob.max()

    print("Processing...")
    result = topology_optimized_postprocess(
        prob,
        threshold=args.threshold,
        use_hysteresis=args.hysteresis,
        hysteresis_low=args.hysteresis_low,
        hysteresis_high=args.hysteresis_high,
        use_skeleton=args.skeleton,
        min_component_size=args.min_size,
        verbose=True,
    )

    print(f"Saving {args.output}...")
    tifffile.imwrite(args.output, result)
    print("Done!")
