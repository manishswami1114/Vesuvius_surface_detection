import numpy as np
import torch
from scipy import ndimage

# =============================================================================
# POST-PROCESSING (OPTIMIZED FOR BEST SCORE)
# =============================================================================
# Changes from previous version:
# - Removed Frangi filter (was hurting score - mean dropped from 0.118 to 0.089)
# - Fixed threshold 0.5 instead of adaptive (0.30 was too aggressive)
# - Kept 2D slicewise operations (gentle, topology-safe)

from scipy.ndimage import (
    binary_closing, binary_opening, binary_fill_holes,
    label, generate_binary_structure, binary_dilation, binary_erosion
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_components(mask):
    """Count connected components using 26-connectivity."""
    struct = generate_binary_structure(3, 3)
    _, n = label(mask, structure=struct)
    return n


def remove_small_components(mask, min_size=50):
    """Remove components smaller than min_size voxels."""
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


def topology_safe_operation(mask, operation_func, name="op"):
    """Apply operation only if it doesn't REDUCE component count."""
    n_before = count_components(mask)
    result = operation_func(mask)
    n_after = count_components(result)
    
    if n_after < n_before:
        print(f"    [REVERT] {name}: would merge {n_before}->{n_after} components")
        return mask
    return result


# =============================================================================
# 2D SLICE-WISE OPERATIONS (gentle, preserves thin structures)
# =============================================================================

def slicewise_hole_fill(mask):
    """Fill holes slice-by-slice in all 3 axes."""
    filled = mask.copy()
    for i in range(mask.shape[0]):
        filled[i] = binary_fill_holes(filled[i])
    for i in range(mask.shape[1]):
        filled[:, i, :] = binary_fill_holes(filled[:, i, :])
    for i in range(mask.shape[2]):
        filled[:, :, i] = binary_fill_holes(filled[:, :, i])
    return filled


def slicewise_morphology(mask, operation='close', iterations=1):
    """Apply morphological operations slice-by-slice."""
    result = mask.copy()
    struct_2d = generate_binary_structure(2, 1)  # 4-connectivity (gentler)
    
    for axis in range(3):
        for i in range(mask.shape[axis]):
            if axis == 0:
                slc = result[i]
            elif axis == 1:
                slc = result[:, i, :]
            else:
                slc = result[:, :, i]
            
            if operation == 'close':
                slc_new = binary_closing(slc, structure=struct_2d, iterations=iterations)
            elif operation == 'open':
                slc_new = binary_opening(slc, structure=struct_2d, iterations=iterations)
            else:
                slc_new = slc
            
            if axis == 0:
                result[i] = slc_new
            elif axis == 1:
                result[:, i, :] = slc_new
            else:
                result[:, :, i] = slc_new
    
    return result


# =============================================================================
# MAIN POST-PROCESSING PIPELINE (OPTIMIZED)
# =============================================================================

def postprocess_v11(pred_prob, 
                    threshold=0.5,  # Fixed threshold (not adaptive)
                    min_component_size=50,
                    use_morphology=True,
                    use_hole_fill=True,
                    verbose=True):
    """
    Optimized post-processing pipeline.
    
    Changes for better score:
    - NO Frangi filter (was reducing mean probability)
    - Fixed threshold 0.5 (adaptive 0.30 was too aggressive)
    - 2D slice-wise morphology (gentle)
    - Topology-safe operations
    """
    if verbose:
        print("Post-processing (OPTIMIZED):")
        print(f"  Input: min={pred_prob.min():.3f}, max={pred_prob.max():.3f}, mean={pred_prob.mean():.3f}")
    
    # Step 1: Fixed threshold (0.5 is standard)
    mask = (pred_prob > threshold).astype(np.uint8)
    fg_pct = 100 * mask.mean()
    if verbose:
        print(f"  1. Threshold ({threshold}): {mask.sum():,} voxels, FG={fg_pct:.2f}%")
    
    if mask.sum() == 0:
        if verbose:
            print("  WARNING: Empty mask!")
        return mask
    
    # Step 2: Remove small components
    n_before = count_components(mask)
    mask = remove_small_components(mask, min_component_size)
    n_after = count_components(mask)
    if verbose:
        print(f"  2. Remove small (<{min_component_size}): {n_before}->{n_after} components")
    
    # Step 3: 2D slice-wise closing (topology-safe)
    if use_morphology:
        mask = topology_safe_operation(
            mask, 
            lambda m: slicewise_morphology(m, 'close', iterations=1),
            "slicewise_close"
        )
        if verbose:
            print(f"  3. Slicewise closing: FG={100*mask.mean():.2f}%")
    
    # Step 4: 2D slice-wise hole filling (topology-safe)
    if use_hole_fill:
        mask = topology_safe_operation(mask, slicewise_hole_fill, "slicewise_hole_fill")
        if verbose:
            print(f"  4. Slicewise hole fill: FG={100*mask.mean():.2f}%")
    
    # Step 5: 2D slice-wise opening (topology-safe) - cleanup
    if use_morphology:
        mask = topology_safe_operation(
            mask,
            lambda m: slicewise_morphology(m, 'open', iterations=1),
            "slicewise_open"
        )
        if verbose:
            print(f"  5. Slicewise opening: FG={100*mask.mean():.2f}%")
    
    # Step 6: Final cleanup
    mask = remove_small_components(mask, min_component_size)
    n_final = count_components(mask)
    
    if verbose:
        print(f"  Final: {n_final} components, {mask.sum():,} voxels, FG={100*mask.mean():.2f}%")
    
    return mask


print("Post-processing ready (OPTIMIZED)")
print("Changes:")
print("  - NO Frangi filter (was hurting score)")
print("  - Fixed threshold 0.5 (not adaptive)")
print("  - 2D slicewise morphology (gentle)")
print("  - Topology-safe operations")