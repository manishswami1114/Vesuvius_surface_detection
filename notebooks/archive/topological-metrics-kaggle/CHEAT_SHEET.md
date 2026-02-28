# Vesuvius Challenge 2025 - Metrics Cheat Sheet

## Leaderboard Formula
```
LB = 0.30 × TopoScore + 0.35 × SurfaceDice + 0.35 × VOI_score
```

## Quick Metric Reference

| Metric | Weight | What It Measures | Key |
|--------|--------|------------------|-----|
| **TopoScore** | 30% | Betti matching (b0, b1, b2) | Don't merge/split components |
| **SurfaceDice** | 35% | Surface accuracy ±2 voxels | Precise boundaries |
| **VOI_score** | 35% | Split/merge errors | Correct component count |

## Betti Numbers Explained

| Dimension | Name | Meaning | Example |
|-----------|------|---------|---------|
| b0 | Components | Separate pieces | 5 sheets = b0=5 |
| b1 | Tunnels | Holes through structure | Donut hole = b1=1 |
| b2 | Cavities | Enclosed voids | Hollow sphere = b2=1 |

## Your Finding: Dice ≠ LB

```
Val Dice 0.60 → LB 0.550 ✓ (better topology)
Val Dice 0.63 → LB 0.543 ✗ (worse topology)
```

**Why?** Higher Dice often means:
- Filled gaps → merged components → worse b0
- Smoothed predictions → broken thin connections → worse VOI

## Improvement Strategies

### TopoScore (+30%)
```
✓ Use skeleton losses from epoch 0 (not epoch 300)
✓ clDice preserves centerline connectivity
✓ Betti matching loss directly optimizes topology
```

### SurfaceDice (+35%)
```
✓ SDF (Signed Distance Function) auxiliary target
✓ Surface-aware boundary loss
✓ Don't over-smooth predictions
```

### VOI (+35%)
```
✓ Skeleton-guided architecture (dual-stream)
✓ Lower threshold (0.3 vs 0.5) preserves connections
✓ Connected component post-processing
```

## Optimal Threshold

Your finding: **threshold = 0.3** works better than 0.5

Why? Lower threshold:
- Preserves thin connections
- Fewer broken components
- Better VOI and TopoScore

## Post-Processing Tips

```python
# 1. Lower threshold
binary = prob > 0.3  # not 0.5

# 2. Keep skeleton-connected regions
skeleton = skeletonize_3d(binary)
dist_to_skel = distance_transform_edt(~skeleton)
binary = binary & (dist_to_skel <= 5)

# 3. Remove small spurious components
labeled = cc3d.connected_components(binary)
for i in range(1, labeled.max() + 1):
    if (labeled == i).sum() < 100:
        binary[labeled == i] = 0
```

## Training Loss Recommendations

### V7 Loss Schedule (Topology-First)
```
Epoch 0-300:   Dice + BCE + Skeleton + SDF
Epoch 300-600: + clDice
Epoch 600+:    + Betti Proxy
```

### Weight Distribution
```python
DICE_WEIGHT = 0.2      # Reduced (was 0.3)
BCE_WEIGHT = 0.1       # Reduced (was 0.2)
SKELETON_WEIGHT = 0.25 # Same
CLDICE_WEIGHT = 0.25   # Same
SDF_WEIGHT = 0.1       # NEW
BETTI_PROXY = 0.1      # NEW
```

## Quick Evaluation

```bash
# Single file
python evaluate_local.py --pred pred.tif --gt gt.tif --threshold 0.3

# Directory
python evaluate_local.py --pred-dir ./preds --gt-dir ./gts
```

## Key Takeaways

1. **65% of LB is topology** (TopoScore + VOI)
2. **Dice optimization hurts LB** - your experiments prove this
3. **Skeleton from epoch 0** - don't wait
4. **Threshold 0.3** - preserves connections
5. **Dual-stream architecture** - separate seg and skeleton learning
