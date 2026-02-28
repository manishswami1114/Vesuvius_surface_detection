"""
Quick test script to verify hysteresis thresholding works correctly.

Run this locally before submitting to Kaggle to verify:
1. The hysteresis algorithm produces expected FG%
2. The morphological operations work
3. The post-processing pipeline completes

Usage:
    python test_hysteresis.py
"""

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, ball, closing

def hysteresis_threshold_3d(prob_map, threshold_high, threshold_low):
    """Apply seeded hysteresis thresholding to a 3D probability map."""
    seeds = prob_map >= threshold_high
    weak = prob_map >= threshold_low
    struct = ndi.generate_binary_structure(3, 3)  # 26-connectivity
    result = ndi.binary_propagation(seeds, structure=struct, mask=weak)
    return result.astype(bool)


def test_hysteresis():
    """Test hysteresis thresholding with synthetic data."""
    print("=" * 60)
    print("Testing Hysteresis Thresholding")
    print("=" * 60)

    # Create a synthetic probability map
    np.random.seed(42)
    shape = (64, 64, 64)

    # Create a base probability map with some structure
    prob_map = np.random.rand(*shape) * 0.3  # Background noise 0-0.3

    # Add a high-confidence region (simulating a vessel)
    prob_map[20:45, 20:45, 20:45] = 0.7  # Medium confidence region
    prob_map[25:40, 25:40, 25:40] = 0.9  # High confidence core

    # Add an isolated low-confidence region (should be removed by hysteresis)
    prob_map[50:55, 50:55, 50:55] = 0.5  # Not connected to seeds

    print(f"\nSynthetic probability map:")
    print(f"  Shape: {prob_map.shape}")
    print(f"  Range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")

    # Test different threshold combinations
    threshold_configs = [
        (0.85, 0.40, "Conservative"),
        (0.80, 0.35, "Balanced"),
        (0.70, 0.30, "Aggressive"),
    ]

    for t_high, t_low, name in threshold_configs:
        print(f"\n{name} (HIGH={t_high}, LOW={t_low}):")

        # Before hysteresis
        seeds_pct = 100 * (prob_map >= t_high).mean()
        weak_pct = 100 * (prob_map >= t_low).mean()
        print(f"  Seeds (>= {t_high}): {seeds_pct:.2f}%")
        print(f"  Weak regions (>= {t_low}): {weak_pct:.2f}%")

        # Apply hysteresis
        result = hysteresis_threshold_3d(prob_map, t_high, t_low)
        hysteresis_pct = 100 * result.mean()
        print(f"  After hysteresis: {hysteresis_pct:.2f}%")

        # Apply morphological closing
        if hysteresis_pct > 0:
            selem = ball(2)
            closed = closing(result, selem)
            closed_pct = 100 * closed.mean()
            print(f"  After closing (r=2): {closed_pct:.2f}%")

            # Remove small objects
            cleaned = remove_small_objects(closed.astype(bool), min_size=50, connectivity=3)
            final_pct = 100 * cleaned.mean()
            print(f"  After dust removal (min=50): {final_pct:.2f}%")

    # Verify the isolated region was removed
    print("\n" + "-" * 60)
    print("Verification:")

    result = hysteresis_threshold_3d(prob_map, 0.80, 0.35)

    # Check the main structure is preserved
    main_structure_preserved = result[30, 30, 30]  # Center of main structure
    print(f"  Main structure preserved: {main_structure_preserved}")

    # Check the isolated region was removed
    isolated_removed = not result[52, 52, 52]  # Center of isolated region
    print(f"  Isolated region removed: {isolated_removed}")

    if main_structure_preserved and isolated_removed:
        print("\n✓ Hysteresis thresholding working correctly!")
    else:
        print("\n✗ Unexpected behavior in hysteresis thresholding")

    print("=" * 60)


def test_real_probability_simulation():
    """Simulate a more realistic probability distribution."""
    print("\n" + "=" * 60)
    print("Simulating Realistic Probability Distribution")
    print("=" * 60)

    np.random.seed(123)
    shape = (128, 128, 128)

    # Create realistic-ish probability map
    prob_map = np.random.rand(*shape) * 0.2  # Background noise

    # Add vessel-like structures with varying confidence
    # Strong vessel core
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = np.array(shape) // 2

    # Create full coordinate arrays
    Z, Y, X = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    # Tube along Z axis
    tube_mask = ((Y - center[1])**2 + (X - center[2])**2) < 15**2
    prob_map[tube_mask] = 0.6 + 0.3 * np.random.rand(tube_mask.sum())

    # Sphere at center (high confidence)
    dist = np.sqrt((Z - center[0])**2 + (Y - center[1])**2 + (X - center[2])**2)
    sphere_mask = dist < 20
    prob_map[sphere_mask] = 0.8 + 0.15 * np.random.rand(sphere_mask.sum())

    print(f"Realistic simulation:")
    print(f"  Shape: {prob_map.shape}")
    print(f"  Range: [{prob_map.min():.3f}, {prob_map.max():.3f}]")
    print(f"  Mean: {prob_map.mean():.3f}")

    # Apply hysteresis
    t_high, t_low = 0.80, 0.35
    seeds_pct = 100 * (prob_map >= t_high).mean()
    weak_pct = 100 * (prob_map >= t_low).mean()

    result = hysteresis_threshold_3d(prob_map, t_high, t_low)
    final_pct = 100 * result.mean()

    print(f"\nWith HIGH={t_high}, LOW={t_low}:")
    print(f"  Seeds: {seeds_pct:.2f}%")
    print(f"  Weak: {weak_pct:.2f}%")
    print(f"  Final: {final_pct:.2f}%")

    # Compare with simple threshold
    simple_50 = 100 * (prob_map >= 0.50).mean()
    simple_55 = 100 * (prob_map >= 0.55).mean()
    print(f"\nComparison with simple thresholding:")
    print(f"  Simple threshold 0.50: {simple_50:.2f}%")
    print(f"  Simple threshold 0.55: {simple_55:.2f}%")
    print(f"  Hysteresis {t_high}/{t_low}: {final_pct:.2f}%")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_hysteresis()
    test_real_probability_simulation()
    print("\nAll tests completed!")
