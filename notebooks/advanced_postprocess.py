# =============================================================================
# ADVANCED POST-PROCESSING FOR VESUVIUS CHALLENGE
# =============================================================================
# Based on insights from competition discussion:
# - Host recommendations (skeletonization, Frangi filter, merge detection)
# - hengck23's hole filling approach
# - Level 1-3 processing strategies
# =============================================================================

import numpy as np
from scipy import ndimage
from scipy.ndimage import (
    binary_fill_holes, distance_transform_edt, gaussian_filter,
    label, generate_binary_structure, binary_dilation, binary_erosion,
    median_filter
)
from skimage.morphology import skeletonize, skeletonize_3d, medial_axis
from skimage.filters import frangi, sato, meijering
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LEVEL 1: IMAGE PROCESSING (CED / Directional Blur)
# =============================================================================

def coherence_enhancing_diffusion(img, sigma=1.0, iterations=5):
    """
    Coherence Enhancing Diffusion - smooths along sheet direction.
    Fixes small gaps while preserving sheet structure.
    """
    from scipy.ndimage import gaussian_filter

    result = img.astype(np.float32)

    for _ in range(iterations):
        # Compute structure tensor (simplified)
        gx = np.gradient(result, axis=2)
        gy = np.gradient(result, axis=1)
        gz = np.gradient(result, axis=0)

        # Smooth gradients
        gx = gaussian_filter(gx, sigma)
        gy = gaussian_filter(gy, sigma)
        gz = gaussian_filter(gz, sigma)

        # Diffusion along dominant direction
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2 + 1e-8)
        result = gaussian_filter(result, sigma * 0.5)

    return result


def directional_blur_2d_slices(mask, sigma=1.0, axis=0):
    """
    Apply directional blur along sheet orientation per 2D slice.
    Helps connect broken lines.
    """
    result = mask.astype(np.float32)

    for i in range(mask.shape[axis]):
        if axis == 0:
            slice_2d = result[i]
        elif axis == 1:
            slice_2d = result[:, i, :]
        else:
            slice_2d = result[:, :, i]

        # Apply oriented Gaussian
        blurred = gaussian_filter(slice_2d, sigma=[sigma, sigma*0.3])

        if axis == 0:
            result[i] = blurred
        elif axis == 1:
            result[:, i, :] = blurred
        else:
            result[:, :, i] = blurred

    return result


# =============================================================================
# LEVEL 2: FRANGI/SATO FILTER (Host Recommended)
# =============================================================================

def apply_frangi_filter_2d(volume, sigmas=range(1, 4), black_ridges=False):
    """
    Apply Frangi vesselness filter slice-by-slice.
    Enhances sheet-like structures.
    """
    result = np.zeros_like(volume, dtype=np.float32)

    for i in range(volume.shape[0]):
        try:
            result[i] = frangi(volume[i].astype(np.float32),
                              sigmas=sigmas, black_ridges=black_ridges)
        except:
            result[i] = volume[i]

    return result


def apply_sato_filter_2d(volume, sigmas=range(1, 4), black_ridges=False):
    """
    Apply Sato tubeness filter slice-by-slice.
    Alternative to Frangi for sheet detection.
    """
    result = np.zeros_like(volume, dtype=np.float32)

    for i in range(volume.shape[0]):
        try:
            result[i] = sato(volume[i].astype(np.float32),
                            sigmas=sigmas, black_ridges=black_ridges)
        except:
            result[i] = volume[i]

    return result


def edt_frangi_pipeline(mask, blur_sigma=1.0, frangi_sigmas=range(1, 3)):
    """
    Host-recommended pipeline: EDT -> Gaussian -> Frangi -> Threshold
    Creates cleaner sheet-like predictions.
    """
    # Step 1: Distance transform (gets midline)
    dist = distance_transform_edt(mask)

    # Step 2: Gaussian blur
    dist_blur = gaussian_filter(dist, sigma=blur_sigma)

    # Step 3: Normalize
    if dist_blur.max() > 0:
        dist_norm = dist_blur / dist_blur.max()
    else:
        return mask

    # Step 4: Frangi filter (slice-wise for speed)
    frangi_result = apply_frangi_filter_2d(dist_norm, sigmas=frangi_sigmas)

    # Step 5: Threshold
    threshold = 0.1 * frangi_result.max() if frangi_result.max() > 0 else 0.5
    result = (frangi_result > threshold).astype(np.uint8)

    return result


# =============================================================================
# LEVEL 2: SKELETONIZATION & MERGE DETECTION (Host Recommended)
# =============================================================================

def skeletonize_2d_slicewise(mask, axis=0):
    """
    2D slicewise skeletonization (Host recommended over 3D).
    Works better for sheet-like structures.
    """
    skeleton = np.zeros_like(mask, dtype=np.uint8)

    for i in range(mask.shape[axis]):
        if axis == 0:
            slice_2d = mask[i] > 0
            skeleton[i] = skeletonize(slice_2d).astype(np.uint8)
        elif axis == 1:
            slice_2d = mask[:, i, :] > 0
            skeleton[:, i, :] = skeletonize(slice_2d).astype(np.uint8)
        else:
            slice_2d = mask[:, :, i] > 0
            skeleton[:, :, i] = skeletonize(slice_2d).astype(np.uint8)

    return skeleton


def detect_merge_points(skeleton):
    """
    Detect merge/junction points in skeleton.
    A merge point has >2 neighbors (Host recommendation).

    Returns: binary mask of merge points
    """
    # 26-connectivity kernel
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    # Count neighbors
    neighbor_count = ndimage.convolve(skeleton.astype(np.int32), kernel, mode='constant')

    # Merge points: skeleton voxels with >2 neighbors
    merge_points = (skeleton > 0) & (neighbor_count > 2)

    return merge_points.astype(np.uint8)


def detect_endpoints(skeleton):
    """
    Detect endpoints in skeleton (voxels with exactly 1 neighbor).
    Used for hole filling - these need to be connected.
    """
    kernel = np.ones((3, 3, 3), dtype=np.int32)
    kernel[1, 1, 1] = 0

    neighbor_count = ndimage.convolve(skeleton.astype(np.int32), kernel, mode='constant')

    # Endpoints: skeleton voxels with exactly 1 neighbor
    endpoints = (skeleton > 0) & (neighbor_count == 1)

    return endpoints.astype(np.uint8)


def split_at_merge_points(mask, skeleton=None, dilation_radius=2):
    """
    Split merged sheets at detected merge points.
    """
    if skeleton is None:
        skeleton = skeletonize_2d_slicewise(mask)

    merge_points = detect_merge_points(skeleton)

    if merge_points.sum() == 0:
        return mask

    # Dilate merge points
    struct = generate_binary_structure(3, 1)
    merge_dilated = binary_dilation(merge_points, structure=struct, iterations=dilation_radius)

    # Remove merge regions from mask
    split_mask = mask.copy()
    split_mask[merge_dilated] = 0

    return split_mask


# =============================================================================
# LEVEL 2: Z-SWEEP / TRACKING (Slice Consistency)
# =============================================================================

def z_sweep_consistency(mask, min_overlap=0.3):
    """
    Enforce consistency between adjacent slices.
    Removes components that don't persist across slices.
    """
    result = mask.copy()
    D, H, W = mask.shape

    # Forward sweep
    for z in range(1, D):
        prev_slice = result[z-1]
        curr_slice = result[z]

        # Label components in current slice
        labeled, n_components = ndimage.label(curr_slice)

        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            comp_size = comp_mask.sum()

            # Check overlap with previous slice
            overlap = (prev_slice & comp_mask).sum()
            overlap_ratio = overlap / (comp_size + 1e-8)

            # Remove if no sufficient overlap (unless first appearance)
            if overlap_ratio < min_overlap and z > 5:
                # Check if component existed in earlier slices
                existed_before = False
                for prev_z in range(max(0, z-5), z):
                    if (result[prev_z] & comp_mask).sum() > 0:
                        existed_before = True
                        break

                if not existed_before:
                    result[z][comp_mask] = 0

    return result


def temporal_smoothing(prob_volume, kernel_size=3):
    """
    Smooth predictions along z-axis for temporal consistency.
    """
    from scipy.ndimage import uniform_filter1d

    smoothed = uniform_filter1d(prob_volume.astype(np.float32),
                                size=kernel_size, axis=0, mode='nearest')
    return smoothed


# =============================================================================
# LEVEL 3: HOLE FILLING (hengck23's approach)
# =============================================================================

def find_endpoint_pairs(endpoints, max_distance=20):
    """
    Find pairs of endpoints that should be connected.
    Uses nearest-neighbor matching.
    """
    coords = np.argwhere(endpoints > 0)

    if len(coords) < 2:
        return []

    pairs = []
    used = set()

    for i, coord1 in enumerate(coords):
        if i in used:
            continue

        min_dist = float('inf')
        best_j = -1

        for j, coord2 in enumerate(coords):
            if j <= i or j in used:
                continue

            dist = np.linalg.norm(coord1 - coord2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_j = j

        if best_j >= 0:
            pairs.append((tuple(coord1), tuple(coords[best_j])))
            used.add(i)
            used.add(best_j)

    return pairs


def trace_path_between_points(prob_map, start, end, step_size=1):
    """
    Trace a path between two points following highest probability.
    Simple greedy path finding.
    """
    path = [start]
    current = np.array(start, dtype=np.float32)
    target = np.array(end, dtype=np.float32)

    max_steps = int(np.linalg.norm(target - current) * 3)

    for _ in range(max_steps):
        if np.linalg.norm(current - target) < step_size * 2:
            path.append(tuple(target.astype(int)))
            break

        # Direction to target
        direction = target - current
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Sample nearby points
        best_next = current + direction * step_size
        best_prob = -1

        for offset in np.random.randn(5, 3) * step_size * 0.5:
            candidate = current + direction * step_size + offset
            candidate = np.clip(candidate, 0, np.array(prob_map.shape) - 1)

            try:
                prob = prob_map[int(candidate[0]), int(candidate[1]), int(candidate[2])]
                if prob > best_prob:
                    best_prob = prob
                    best_next = candidate
            except IndexError:
                continue

        current = best_next
        path.append(tuple(current.astype(int)))

    return path


def fill_holes_by_tracing(mask, prob_map, max_gap_distance=15):
    """
    Fill holes by tracing paths between endpoint pairs.
    Based on hengck23's approach.
    """
    # Get skeleton
    skeleton = skeletonize_2d_slicewise(mask)

    # Find endpoints
    endpoints = detect_endpoints(skeleton)

    # Find pairs to connect
    pairs = find_endpoint_pairs(endpoints, max_distance=max_gap_distance)

    # Trace paths and fill
    filled = mask.copy()

    for start, end in pairs:
        path = trace_path_between_points(prob_map, start, end)

        for point in path:
            try:
                # Fill with small dilation
                z, y, x = point
                filled[max(0,z-1):z+2, max(0,y-1):y+2, max(0,x-1):x+2] = 1
            except:
                continue

    return filled


# =============================================================================
# SLICE-WISE HOLE FILLING (Multiple Directions)
# =============================================================================

def slice_wise_hole_fill_multi_axis(mask):
    """
    Fill holes in 2D slices across all 3 axes.
    Catches "tunnels" that 3D hole filling misses.
    """
    filled = mask.copy()

    # Process each axis
    for axis in [0, 1, 2]:
        for i in range(mask.shape[axis]):
            if axis == 0:
                slice_2d = filled[i]
                filled[i] = binary_fill_holes(slice_2d)
            elif axis == 1:
                slice_2d = filled[:, i, :]
                filled[:, i, :] = binary_fill_holes(slice_2d)
            else:
                slice_2d = filled[:, :, i]
                filled[:, :, i] = binary_fill_holes(slice_2d)

    return filled


# =============================================================================
# COMPONENT ANALYSIS
# =============================================================================

def count_components_26(mask):
    """Count 26-connected components."""
    struct = generate_binary_structure(3, 3)
    _, n = label(mask, structure=struct)
    return n


def remove_small_components(mask, min_size=100):
    """Remove small connected components."""
    struct = generate_binary_structure(3, 3)
    labeled, n = label(mask, structure=struct)

    if n == 0:
        return mask

    sizes = np.bincount(labeled.ravel())
    small = sizes < min_size
    small[0] = False

    result = mask.copy()
    result[small[labeled]] = 0

    return result


def topology_safe_operation(mask, operation, **kwargs):
    """Apply operation but revert if components merge."""
    n_before = count_components_26(mask)
    result = operation(mask, **kwargs)
    n_after = count_components_26(result)

    if n_after < n_before:
        print(f"  [REVERT] {operation.__name__}: {n_before} -> {n_after} components")
        return mask

    return result


# =============================================================================
# SURFACE SMOOTHING
# =============================================================================

def surface_aware_smoothing(mask, sigma=0.5):
    """Smooth surface via signed distance field."""
    dist_in = distance_transform_edt(mask)
    dist_out = distance_transform_edt(~mask)
    sdf = dist_in - dist_out
    sdf_smooth = gaussian_filter(sdf.astype(np.float32), sigma=sigma)
    return (sdf_smooth > 0).astype(np.uint8)


# =============================================================================
# COMPREHENSIVE PIPELINE
# =============================================================================

def advanced_postprocess(
    pred_prob: np.ndarray,
    threshold: float = 0.5,
    use_frangi: bool = True,
    use_hole_tracing: bool = True,
    use_merge_splitting: bool = False,
    min_component_size: int = 100,
    surface_sigma: float = 0.5,
    verbose: bool = True
) -> np.ndarray:
    """
    Advanced post-processing pipeline with all techniques.

    Pipeline:
    1. Threshold
    2. Remove small components
    3. (Optional) Frangi enhancement
    4. Surface smoothing
    5. Slice-wise hole filling
    6. (Optional) Hole tracing between endpoints
    7. (Optional) Split at merge points
    8. Final cleanup
    """
    if verbose:
        print("Advanced Post-processing:")

    # Step 1: Threshold
    mask = (pred_prob > threshold).astype(np.uint8)
    if verbose:
        print(f"  1. Threshold ({threshold}): FG={100*mask.mean():.2f}%")

    # Step 2: Remove small components
    n1 = count_components_26(mask)
    mask = remove_small_components(mask, min_component_size)
    n2 = count_components_26(mask)
    if verbose:
        print(f"  2. Remove small (<{min_component_size}): {n1} -> {n2} components")

    # Step 3: Frangi enhancement (optional)
    if use_frangi:
        try:
            enhanced = edt_frangi_pipeline(mask)
            # Combine with original
            mask = (mask | enhanced).astype(np.uint8)
            if verbose:
                print(f"  3. Frangi enhancement: FG={100*mask.mean():.2f}%")
        except Exception as e:
            if verbose:
                print(f"  3. Frangi skipped: {e}")

    # Step 4: Surface smoothing
    mask = surface_aware_smoothing(mask, sigma=surface_sigma)
    if verbose:
        print(f"  4. Surface smooth (σ={surface_sigma}): FG={100*mask.mean():.2f}%")

    # Step 5: Slice-wise hole filling
    mask = topology_safe_operation(mask, slice_wise_hole_fill_multi_axis)
    if verbose:
        print(f"  5. Slice-wise hole fill: FG={100*mask.mean():.2f}%")

    # Step 6: Hole tracing (optional)
    if use_hole_tracing:
        try:
            mask = fill_holes_by_tracing(mask, pred_prob, max_gap_distance=15)
            if verbose:
                print(f"  6. Hole tracing: FG={100*mask.mean():.2f}%")
        except Exception as e:
            if verbose:
                print(f"  6. Hole tracing skipped: {e}")

    # Step 7: Split at merge points (optional, use carefully)
    if use_merge_splitting:
        try:
            n_before = count_components_26(mask)
            mask = split_at_merge_points(mask)
            n_after = count_components_26(mask)
            if verbose:
                print(f"  7. Merge splitting: {n_before} -> {n_after} components")
        except Exception as e:
            if verbose:
                print(f"  7. Merge splitting skipped: {e}")

    # Step 8: Final cleanup
    mask = remove_small_components(mask, min_component_size)
    n_final = count_components_26(mask)
    if verbose:
        print(f"  8. Final: {n_final} components, FG={100*mask.mean():.2f}%")

    return mask


# =============================================================================
# QUICK PRESETS
# =============================================================================

def postprocess_conservative(pred_prob, threshold=0.5):
    """Conservative: minimal changes, preserve original topology."""
    return advanced_postprocess(
        pred_prob, threshold=threshold,
        use_frangi=False, use_hole_tracing=False,
        use_merge_splitting=False, verbose=True
    )


def postprocess_aggressive(pred_prob, threshold=0.5):
    """Aggressive: fill holes, split merges."""
    return advanced_postprocess(
        pred_prob, threshold=threshold,
        use_frangi=True, use_hole_tracing=True,
        use_merge_splitting=True, verbose=True
    )


def postprocess_balanced(pred_prob, threshold=0.5):
    """Balanced: fill holes but don't split (recommended)."""
    return advanced_postprocess(
        pred_prob, threshold=threshold,
        use_frangi=True, use_hole_tracing=True,
        use_merge_splitting=False, verbose=True
    )


if __name__ == "__main__":
    print("Advanced Post-processing Module Loaded")
    print("\nAvailable functions:")
    print("  - advanced_postprocess(): Full pipeline with options")
    print("  - postprocess_conservative(): Minimal changes")
    print("  - postprocess_balanced(): Recommended default")
    print("  - postprocess_aggressive(): Maximum hole filling")
    print("\nLevel 1 (Image Processing):")
    print("  - coherence_enhancing_diffusion()")
    print("  - directional_blur_2d_slices()")
    print("\nLevel 2 (Feature Enhancement):")
    print("  - edt_frangi_pipeline()")
    print("  - skeletonize_2d_slicewise()")
    print("  - detect_merge_points()")
    print("  - split_at_merge_points()")
    print("\nLevel 3 (Hole Filling):")
    print("  - fill_holes_by_tracing()")
    print("  - find_endpoint_pairs()")
