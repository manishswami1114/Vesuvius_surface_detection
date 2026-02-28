# Vesuvius Challenge - Comprehensive Improvement Guide

## Executive Summary

This guide details all improvements made to the baseline notebook to maximize the competition score. The competition metric is:

```
Score = 0.30 × TopoScore + 0.35 × SurfaceDice@τ + 0.35 × VOI_score
```

**Baseline Performance:** ~0.57 validation dice
**Expected Improved Performance:** 0.70-0.80+ validation dice

## Key Improvements Overview

| Improvement | Baseline | Improved | Impact on Metric | Expected Gain |
|-------------|----------|----------|------------------|---------------|
| Model Size | 4.5M (mit_b0) | 13M+ (mit_b3) | All metrics | +5-8% |
| Input Patch | 128³ | 192³ or 256³ | TopoScore, VOI | +3-5% |
| Batch Size | 1 | 2-4 | Training stability | +1-2% |
| clDice iters | 1 | 30 | TopoScore, VOI | +5-10% |
| Post-processing | None | Component filtering | VOI, TopoScore | +3-5% |
| TTA | None | 4-8x | All metrics | +1-2% |
| **Total Expected** | | | | **+15-25%** |

---

## Detailed Improvement Analysis

### 1. Model Architecture Upgrade

**Baseline:** SegFormer with mit_b0 encoder (4.5M parameters)
**Improved:** SegFormer with mit_b3 encoder (13M parameters) or SwinUNETR

**Why this works:**
- Larger models have more capacity to learn complex topology patterns
- Better feature extraction at multiple scales
- More effective attention mechanisms for capturing long-range dependencies

**Trade-offs:**
- Pros: Better accuracy, more robust predictions
- Cons: Slower training (2-3x), more memory required

**Recommended models for H100:**
```
SegFormer-B3:  13M params, good balance (RECOMMENDED)
SegFormer-B5:  82M params, highest accuracy but slow
SwinUNETR:     25M params, excellent for 3D medical imaging
```

### 2. Larger Input Patches

**Baseline:** 128×128×128 voxels
**Improved:** 192×192×192 voxels (or 256³ if memory allows)

**Why this works:**
- More context helps the model understand scroll structure
- Can see more of adjacent wraps, reducing merger errors
- Better for TopoScore (fewer artificial splits/merges)

**Mathematical reasoning:**
- 128³ patch might contain only partial wraps
- 192³ contains ~3.4x more voxels, likely showing complete wrap structures
- Model can learn to avoid bridging adjacent wraps when it sees both

**Memory impact:**
```
128³ × batch_size=1:  ~2.1M voxels/batch
192³ × batch_size=2:  ~14.2M voxels/batch (6.7x more)
H100 80GB can handle this easily
```

### 3. Critical: clDice Loss with Proper Iterations

**Baseline:** `clDice iters=1` (essentially useless)
**Improved:** `clDice iters=30`

**Why this is the MOST IMPORTANT fix:**

clDice (Centerline Dice) Loss measures skeleton overlap, not just voxel overlap. The `iters` parameter controls the number of morphological erosion iterations used to compute the skeleton (centerline).

```
iters=1:  Only 1 erosion step → NOT a real skeleton, just slightly eroded surface
iters=30: 30 erosion steps → Proper 1-voxel thick skeleton representing topology
```

**Impact on each metric:**
- **VOI_score:** Skeleton-based loss penalizes artificial bridges between wraps. A thin bridge has low standard Dice impact but HUGE clDice impact.
- **TopoScore:** Preserves connectivity of individual components, reducing k=0 errors
- **SurfaceDice:** Indirect improvement through better topology

**Visual explanation:**
```
Without clDice (or iters=1):
[Wrap 1]---bridge---[Wrap 2]  ← Model might create thin bridge
                                This barely affects Dice but kills VOI!

With clDice (iters=30):
[Wrap 1]            [Wrap 2]  ← Model learns to avoid bridges
                                Skeleton would have impossible connection
```

**Computational cost:**
- iters=1: ~1ms per sample
- iters=30: ~30-50ms per sample
- Worth the cost! This single change can provide +5-10% improvement.

### 4. Combined Loss Function

**Baseline:** DiceCE + clDice (weak)
**Improved:** DiceCE (0.4) + clDice (0.4) + Tversky (0.2)

**Why each component:**

| Loss | Target Metric | Role |
|------|---------------|------|
| DiceCE | SurfaceDice | General segmentation accuracy |
| clDice | VOI + TopoScore | Topology preservation |
| Tversky | SurfaceDice | Boundary precision (FP/FN balance) |

**Tversky Loss explained:**
```python
Tversky(α=0.3, β=0.7)
# α = False Positive penalty
# β = False Negative penalty

# Higher β (0.7) means:
# - Model is penalized more for missing foreground
# - Encourages complete segmentation of scroll surface
# - Reduces splits in wraps
```

### 5. Post-Processing Pipeline

**Baseline:** None
**Improved:** Component filtering + hole filling

**Why this works:**

The competition metric explicitly penalizes:
- Extra components (hurts VOI through over-segmentation)
- Extra holes/handles (hurts TopoScore through Betti mismatch)

**Step 1: Remove small components**
```python
min_component_size = 1000  # ~10³ voxels

# Small disconnected pieces are usually:
# - Noise from model uncertainty
# - Fragments from imperfect boundaries
# - They hurt VOI (H(GT|Pred) increases)
```

**Step 2: Fill small holes**
```python
fill_holes_size = 500  # ~8³ voxels

# Small holes are usually:
# - Prediction uncertainty in middle of surface
# - They hurt TopoScore (extra k=2 features)
```

**Potential downside:**
- May slightly hurt SurfaceDice if removing real small components
- Tune thresholds on validation data!

### 6. Test-Time Augmentation (TTA)

**Baseline:** None
**Improved:** 4-8x augmentation (flips + optional rotations)

**Strategy:**
```python
# Average predictions from:
1. Original
2. Flip along axis 0 (D)
3. Flip along axis 1 (H)
4. Flip along axis 2 (W)
# Optionally add 90° rotations for more TTA
```

**Why this works:**
- Model predictions vary slightly with orientation
- Averaging reduces variance, improves consistency
- Especially helps at boundaries (improves SurfaceDice)

**Expected gain:** +0.5-1.5% across all metrics

### 7. Training Optimizations for H100

**Batch size:**
- Baseline: 1
- Improved: 2-4

**Why larger batch helps:**
- More stable gradient estimates
- Better use of GPU parallelism
- Can use larger learning rate

**Mixed precision:**
```python
keras.mixed_precision.set_global_policy("mixed_float16")
```
- Cuts memory usage ~40%
- Speeds up training ~50%
- No accuracy loss with modern GPUs

**Gradient clipping:**
```python
gradients = [torch.clamp(g, -1.0, 1.0) for g in gradients]
```
- Prevents gradient explosion
- More stable training with larger batches

---

## Implementation Checklist

### Must-Have (High Impact)
- [x] Upgrade model to mit_b3 or larger
- [x] Increase clDice iterations to 30+
- [x] Add post-processing (component filtering)
- [x] Use mixed precision training
- [x] Increase patch size to 192³

### Should-Have (Medium Impact)
- [x] Add Tversky loss to combination
- [x] Implement TTA for inference
- [x] Use more validation volumes (2+)
- [x] Add gradient clipping

### Nice-to-Have (Lower Impact)
- [ ] Ensemble multiple models
- [ ] Cross-validation (5-fold)
- [ ] Progressive resizing (train 128→192→256)
- [ ] More aggressive augmentation

---

## Expected Training Curves

### Baseline (problematic)
```
Epoch 1:  Train Dice: 0.67, Val Dice: 0.54
Epoch 10: Train Dice: 0.70, Val Dice: 0.53  ← Overfitting!
Epoch 50: Train Dice: 0.72, Val Dice: 0.57
```

### Improved (expected)
```
Epoch 1:  Train Dice: 0.65, Val Dice: 0.55
Epoch 10: Train Dice: 0.72, Val Dice: 0.65
Epoch 50: Train Dice: 0.78, Val Dice: 0.72
Epoch 100: Train Dice: 0.80, Val Dice: 0.75  ← Better generalization
```

---

## Troubleshooting

### Out of Memory
```python
# Reduce batch_size to 1
config.batch_size = 1

# Or reduce patch size to 160³
config.input_shape = (160, 160, 160)

# Or use gradient accumulation
accumulation_steps = 2
```

### Training Unstable (loss exploding)
```python
# Reduce learning rate
config.peak_lr = 1e-4  # Instead of 5e-4

# Increase warmup
config.warmup_ratio = 0.1  # Instead of 0.05
```

### Validation Score Not Improving
1. Check clDice iterations (must be 30+)
2. Verify post-processing is applied
3. Try different model architecture
4. Add more augmentation

---

## Model Ensemble Strategy

For maximum score, ensemble multiple models:

```python
models = [
    ('segformer_b3', 0.3),   # Weight
    ('segformer_b5', 0.3),
    ('swinunetr', 0.4),
]

# Average softmax probabilities, then argmax
ensemble_pred = sum(w * model(x) for model, w in models)
final_pred = argmax(ensemble_pred)
```

**Expected gain from ensemble:** +2-4%

---

## Competition Strategy Timeline

### Week 1-2: Establish Baseline
- Implement all improvements from this guide
- Validate CV-LB correlation
- Submit simple model to understand test set

### Week 3-4: Model Iteration
- Try different architectures
- Optimize hyperparameters
- Start ensemble preparation

### Final Week: Ensemble & Selection
- Finalize ensemble weights
- Create diverse submissions
- Select final 2 submissions carefully

---

## Files Provided

1. `vesuvius_improved_training.py` - Full training pipeline
2. `vesuvius_inference_submission.py` - Inference and submission
3. `IMPROVEMENTS_GUIDE.md` - This documentation

## Quick Start

```python
# 1. Install dependencies
pip install keras-nightly medicai --break-system-packages

# 2. Run training
python vesuvius_improved_training.py

# 3. Run inference
python vesuvius_inference_submission.py
```

---

## References

1. [clDice Loss Paper](https://arxiv.org/abs/2003.07311) - Topology-preserving loss
2. [SurfaceDice](https://arxiv.org/abs/1809.04430) - Surface distance metric
3. [Betti Matching](https://arxiv.org/abs/2407.04683) - Topological feature matching
4. [Vesuvius Challenge](https://scrollprize.org/) - Competition context
