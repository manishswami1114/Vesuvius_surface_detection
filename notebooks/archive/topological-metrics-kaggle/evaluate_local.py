#!/usr/bin/env python3
"""
Vesuvius Challenge 2025 - Local Evaluation Script

Usage:
    python evaluate_local.py --pred prediction.tif --gt ground_truth.tif
    python evaluate_local.py --pred-dir ./predictions --gt-dir ./ground_truth
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import tifffile
except ImportError:
    print("ERROR: tifffile not installed. Run: pip install tifffile")
    sys.exit(1)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from topometrics import compute_leaderboard_score
except ImportError as e:
    print(f"ERROR: Could not import topometrics: {e}")
    print("Make sure to build Betti-Matching-3D first: make build-betti && make dev")
    sys.exit(1)


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D TIF volume."""
    return tifffile.imread(str(path))


def evaluate_single(
    pred_path: Path,
    gt_path: Path,
    threshold: Optional[float] = None,
    ignore_label: int = 2,
    surface_tolerance: float = 2.0,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a single prediction against ground truth.

    Args:
        pred_path: Path to prediction TIF
        gt_path: Path to ground truth TIF
        threshold: Binarization threshold (None = use != 0)
        ignore_label: Label value to ignore in GT
        surface_tolerance: Tolerance for Surface Dice
        verbose: Print detailed results

    Returns:
        Dictionary with all metrics
    """
    pred = load_volume(pred_path)
    gt = load_volume(gt_path)

    # Apply threshold if specified
    if threshold is not None:
        pred = (pred > threshold).astype(np.uint8)

    report = compute_leaderboard_score(
        predictions=pred,
        labels=gt,
        dims=(0, 1, 2),
        spacing=(1.0, 1.0, 1.0),
        surface_tolerance=surface_tolerance,
        voi_connectivity=26,
        combine_weights=(0.3, 0.35, 0.35),
        fg_threshold=threshold,
        ignore_label=ignore_label,
    )

    results = {
        'leaderboard': report.score,
        'toposcore': report.topo.toposcore,
        'topoF1_0': report.topo.topoF1_by_dim.get(0, float('nan')),
        'topoF1_1': report.topo.topoF1_by_dim.get(1, float('nan')),
        'topoF1_2': report.topo.topoF1_by_dim.get(2, float('nan')),
        'surface_dice': report.surface_dice,
        'voi_score': report.voi.voi_score,
        'voi_total': report.voi.voi_total,
        'voi_split': report.voi.voi_split,
        'voi_merge': report.voi.voi_merge,
    }

    if verbose:
        print("=" * 60)
        print("VESUVIUS CHALLENGE 2025 - EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nPrediction: {pred_path.name}")
        print(f"Ground Truth: {gt_path.name}")
        print(f"Threshold: {threshold if threshold else 'None (!=0)'}")
        print("-" * 60)
        print(f"\n{'LEADERBOARD SCORE:':<25} {report.score:.4f}")
        print("-" * 60)
        print(f"\n{'Component':<25} {'Score':<10} {'Weight':<10} {'Contribution':<10}")
        print("-" * 60)

        topo_contrib = 0.30 * report.topo.toposcore
        surf_contrib = 0.35 * report.surface_dice
        voi_contrib = 0.35 * report.voi.voi_score

        print(f"{'TopoScore':<25} {report.topo.toposcore:.4f}     30%       {topo_contrib:.4f}")
        print(f"{'SurfaceDice':<25} {report.surface_dice:.4f}     35%       {surf_contrib:.4f}")
        print(f"{'VOI Score':<25} {report.voi.voi_score:.4f}     35%       {voi_contrib:.4f}")
        print("-" * 60)

        print(f"\nTOPOLOGY DETAILS:")
        print(f"  TopoF1_0 (components): {report.topo.topoF1_by_dim.get(0, float('nan')):.4f}")
        print(f"  TopoF1_1 (tunnels):    {report.topo.topoF1_by_dim.get(1, float('nan')):.4f}")
        print(f"  TopoF1_2 (cavities):   {report.topo.topoF1_by_dim.get(2, float('nan')):.4f}")

        print(f"\nVOI DETAILS:")
        print(f"  VOI Total: {report.voi.voi_total:.4f} (lower is better)")
        print(f"  VOI Split: {report.voi.voi_split:.4f} (over-segmentation)")
        print(f"  VOI Merge: {report.voi.voi_merge:.4f} (under-segmentation)")

        print("=" * 60)

    return results


def evaluate_directory(
    pred_dir: Path,
    gt_dir: Path,
    threshold: Optional[float] = None,
    ignore_label: int = 2,
    surface_tolerance: float = 2.0,
) -> None:
    """Evaluate all TIF files in directories."""
    import csv

    pred_files = sorted(pred_dir.glob("*.tif")) + sorted(pred_dir.glob("*.tiff"))

    if not pred_files:
        print(f"No TIF files found in {pred_dir}")
        return

    results = []

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            # Try with .tiff extension
            gt_path = gt_dir / pred_path.stem + ".tiff"
            if not gt_path.exists():
                print(f"Warning: No GT found for {pred_path.name}, skipping")
                continue

        print(f"\nEvaluating: {pred_path.name}")
        result = evaluate_single(
            pred_path, gt_path, threshold, ignore_label, surface_tolerance, verbose=False
        )
        result['case'] = pred_path.stem
        results.append(result)
        print(f"  LB: {result['leaderboard']:.4f} | Topo: {result['toposcore']:.4f} | "
              f"SurfDice: {result['surface_dice']:.4f} | VOI: {result['voi_score']:.4f}")

    if results:
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        lb_scores = [r['leaderboard'] for r in results]
        print(f"Mean LB Score: {np.mean(lb_scores):.4f} ± {np.std(lb_scores):.4f}")
        print(f"Min: {np.min(lb_scores):.4f}, Max: {np.max(lb_scores):.4f}")

        # Save CSV
        csv_path = pred_dir / "evaluation_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Challenge 2025 - Local Evaluation"
    )

    # Single file mode
    parser.add_argument("--pred", type=Path, help="Path to prediction TIF")
    parser.add_argument("--gt", type=Path, help="Path to ground truth TIF")

    # Directory mode
    parser.add_argument("--pred-dir", type=Path, help="Directory with prediction TIFs")
    parser.add_argument("--gt-dir", type=Path, help="Directory with ground truth TIFs")

    # Options
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binarization threshold (default: None, uses !=0)")
    parser.add_argument("--ignore-label", type=int, default=2,
                        help="GT label value to ignore (default: 2)")
    parser.add_argument("--surface-tolerance", type=float, default=2.0,
                        help="Surface Dice tolerance in voxels (default: 2.0)")

    args = parser.parse_args()

    if args.pred and args.gt:
        # Single file mode
        evaluate_single(
            args.pred, args.gt, args.threshold, args.ignore_label, args.surface_tolerance
        )
    elif args.pred_dir and args.gt_dir:
        # Directory mode
        evaluate_directory(
            args.pred_dir, args.gt_dir, args.threshold, args.ignore_label, args.surface_tolerance
        )
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python evaluate_local.py --pred pred.tif --gt gt.tif")
        print("  python evaluate_local.py --pred pred.tif --gt gt.tif --threshold 0.3")
        print("  python evaluate_local.py --pred-dir ./preds --gt-dir ./gts")


if __name__ == "__main__":
    main()
