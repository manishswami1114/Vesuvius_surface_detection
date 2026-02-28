# Vesuvius Challenge 2025 - Evaluation Metrics Guide

## Quick Reference

```
Leaderboard Score = 0.30 × TopoScore + 0.35 × SurfaceDice + 0.35 × VOI_score
```

| Metric | Weight | Measures | Range | Higher = Better |
|--------|--------|----------|-------|-----------------|
| **TopoScore** | 30% | Topology preservation (components, tunnels, cavities) | [0, 1] | Yes |
| **SurfaceDice** | 35% | Surface accuracy within tolerance | [0, 1] | Yes |
| **VOI_score** | 35% | Segmentation quality (split/merge errors) | (0, 1] | Yes |

---

## 1. Topological Score (TopoScore)

### What it Measures
Evaluates how well the prediction preserves topological features:
- **b0 (dim 0)**: Connected components - separate sheets remain separate
- **b1 (dim 1)**: Tunnels/loops - holes through the structure
- **b2 (dim 2)**: Cavities - enclosed void spaces

### How it Works
Uses **Betti Matching** algorithm to match features between prediction and ground truth:

```
For each dimension k ∈ {0, 1, 2}:
  m_k = matched features (correct topology)
  p_k = total predicted features
  g_k = total ground truth features

  TopoF1_k = 2 × m_k / (p_k + g_k)
```

### Weight Distribution
Default weights: `w0=0.34, w1=0.33, w2=0.33`

Only **active dimensions** (where features exist) contribute. Weights are renormalized.

### Key Insight for Training
- **Don't merge separate sheets** → hurts b0
- **Don't create false holes** → hurts b1
- **Don't fill cavities incorrectly** → hurts b2

---

## 2. Surface Dice at Tolerance

### What it Measures
Surface accuracy with spatial tolerance τ (default: 2.0 voxels)

### Formula
```
SurfaceDice = (|∂P ∩ N_τ(∂G)| + |∂G ∩ N_τ(∂P)|) / (|∂P| + |∂G|)
```

Where:
- ∂P = prediction surface
- ∂G = ground truth surface
- N_τ = τ-voxel neighborhood

### Key Insight for Training
- Predictions within 2 voxels of true surface are OK
- Focus on **surface accuracy**, not just volume
- Thin structures need precise boundaries

---

## 3. Variation of Information (VOI)

### What it Measures
Segmentation quality via connected components analysis:
- **VOI_split** = H(GT | Pred) - over-segmentation (too many pieces)
- **VOI_merge** = H(Pred | GT) - under-segmentation (merged structures)

### Score Mapping
```
VOI_total = VOI_split + VOI_merge  (lower is better)
VOI_score = 1 / (1 + α × VOI_total)  (higher is better)
```

Default α = 1.0, connectivity = 26 (3D)

### Key Insight for Training
- **Breaking connections** → increases VOI_split
- **Merging separate sheets** → increases VOI_merge
- Both hurt the final score

---

## Understanding Your Score

### Example Analysis

```
Your Score: 0.550 with Val Dice 0.60
  - TopoScore: ~0.45 (30% weight) → contributes 0.135
  - SurfaceDice: ~0.65 (35% weight) → contributes 0.228
  - VOI_score: ~0.55 (35% weight) → contributes 0.193
```

### Why Higher Dice Hurts LB

| Val Dice | LB Score | What Happens |
|----------|----------|--------------|
| 0.60 | 0.550 | Better topology |
| 0.63 | 0.543 | Dice↑ but Topo↓, VOI↓ |

**Explanation**: Optimizing Dice fills gaps and smooths predictions, which:
- Merges separate components → hurts TopoScore (b0)
- Breaks thin connections → hurts VOI
- May improve pixel overlap but destroys structure

---

## How to Improve Each Component

### Improving TopoScore (+30% weight)

| Problem | Solution |
|---------|----------|
| Merged components | Use skeleton loss from epoch 0 |
| False tunnels | Add Betti proxy loss |
| Broken connections | Skeleton-guided architecture |

**Best Losses**: clDice, SkeletonRecall, Betti Matching

### Improving SurfaceDice (+35% weight)

| Problem | Solution |
|---------|----------|
| Fuzzy boundaries | Surface-aware loss |
| Wrong surface location | Signed Distance Function (SDF) target |
| Oversized predictions | Conservative thresholding |

**Best Losses**: SDF Loss, Boundary Loss

### Improving VOI (+35% weight)

| Problem | Solution |
|---------|----------|
| Over-segmentation | Higher threshold, erosion post-processing |
| Under-segmentation | Skeleton-guided decoder, clDice |
| Wrong component count | Connected component filtering |

**Best Losses**: VOI-aware losses, Skeleton supervision

---

## Post-Processing for Better LB

### 1. Threshold Selection
```python
# Don't just use 0.5!
# Lower threshold preserves thin connections
threshold = 0.3  # Your finding that 0.3 works well
```

### 2. Skeleton-Guided Post-Processing
```python
# Keep predictions connected to skeleton
skeleton = skeletonize_3d(prediction > threshold)
skeleton_connected = binary & (distance_to_skeleton <= 5)
```

### 3. Connected Component Filtering
```python
# Remove spurious small components
labeled = cc3d.connected_components(binary)
for i in range(1, labeled.max() + 1):
    if (labeled == i).sum() < min_size:
        binary[labeled == i] = 0
```

---

## Python API Usage

```python
from topometrics import compute_leaderboard_score
import tifffile

# Load volumes
pred = tifffile.imread("prediction.tif")
gt = tifffile.imread("ground_truth.tif")

# Compute score
report = compute_leaderboard_score(
    predictions=pred,
    labels=gt,
    dims=(0, 1, 2),
    spacing=(1.0, 1.0, 1.0),
    surface_tolerance=2.0,
    voi_connectivity=26,
    combine_weights=(0.3, 0.35, 0.35),
    ignore_label=2,  # Ignore label value
)

# Results
print(f"Leaderboard Score: {report.score:.4f}")
print(f"TopoScore: {report.topo.toposcore:.4f}")
print(f"  - TopoF1_0 (components): {report.topo.topoF1_by_dim[0]:.4f}")
print(f"  - TopoF1_1 (tunnels): {report.topo.topoF1_by_dim[1]:.4f}")
print(f"  - TopoF1_2 (cavities): {report.topo.topoF1_by_dim[2]:.4f}")
print(f"Surface Dice: {report.surface_dice:.4f}")
print(f"VOI Score: {report.voi.voi_score:.4f}")
print(f"  - VOI Split: {report.voi.voi_split:.4f}")
print(f"  - VOI Merge: {report.voi.voi_merge:.4f}")
```

---

## CLI Usage

```bash
python metric_compute.py \
  --gt-dir /path/to/ground_truth \
  --pred-dir /path/to/predictions \
  --outdir ./metrics_output \
  --spacing 1 1 1 \
  --surface-tolerance 2.0 \
  --ignore-label 2 \
  --workers 4
```

Outputs:
- `metrics_per_case.csv` - Per-case metrics
- `metrics_summary.json` - Aggregate statistics
- Histogram plots

---

## Key Takeaways for Vesuvius Challenge

1. **65% of score is topology-related** (TopoScore + VOI)
2. **Dice optimization hurts topology** - your finding confirms this
3. **Use skeleton losses from epoch 0** - don't wait until epoch 300
4. **Lower thresholds preserve connectivity** - 0.3 works better than 0.5
5. **Dual-stream architectures** help separate seg from skeleton learning
6. **SDF auxiliary target** encodes topology implicitly

---

## Architecture Recommendations

Based on metric analysis:

| Component | Recommendation | Why |
|-----------|----------------|-----|
| Loss | clDice + SkeletonRecall from epoch 0 | Direct topology optimization |
| Architecture | Dual-stream (Seg + Skeleton) | Separate optimization paths |
| Auxiliary | SDF prediction | Implicit topology encoding |
| Attention | D-LKA for thin structures | Large receptive field |
| Post-proc | Skeleton-guided + CC filtering | Clean up predictions |

---

## References

1. [Betti Matching Error (arXiv 2407.04683)](https://arxiv.org/abs/2407.04683)
2. [Surface Dice (arXiv 1809.04430)](https://arxiv.org/abs/1809.04430)
3. [Variation of Information](https://en.wikipedia.org/wiki/Variation_of_information)
4. [clDice (CVPR 2021)](https://github.com/jocpae/clDice)
5. [Skeleton Recall (ECCV 2024)](https://github.com/MIC-DKFZ/Skeleton-Recall)
6. [Topology-aware metrics paper (arXiv 2412.14619)](https://arxiv.org/abs/2412.14619)
