import numpy as np
import cc3d
from scipy import ndimage
from skimage import morphology

class AdvancedPostProcessing:
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def remove_small_components(self, mask: np.ndarray) -> np.ndarray:
        labeled = cc3d.connected_components(mask.astype(np.uint8), connectivity=26)
        if labeled.max() == 0:
            return mask
        stats = cc3d.statistics(labeled)
        sizes = stats['voxel_counts'][1:]
        keep_labels = [i + 1 for i, size in enumerate(sizes) if size >= self.cfg.MIN_COMPONENT_SIZE]
        if len(keep_labels) == 0:
            largest_label = int(np.argmax(sizes) + 1) if len(sizes) > 0 else []
            keep_labels = [largest_label] if largest_label else []
        if len(keep_labels) > self.cfg.MAX_COMPONENTS:
            label_sizes = [(lbl, sizes[lbl-1]) for lbl in keep_labels]
            label_sizes.sort(key=lambda x: x[1], reverse=True)
            keep_labels = [lbl for lbl, _ in label_sizes[:self.cfg.MAX_COMPONENTS]]
        if len(keep_labels) == 0:
            return np.zeros_like(mask)
        clean_mask = np.isin(labeled, keep_labels).astype(np.uint8)
        return clean_mask

    def fill_holes(self, mask: np.ndarray) -> np.ndarray:
        filled = mask.copy()
        for z in range(mask.shape[0]):
            filled[z] = ndimage.binary_fill_holes(mask[z])
        diff = filled & (~mask.astype(bool))
        labeled_holes = cc3d.connected_components(diff.astype(np.uint8))
        if labeled_holes.max() > 0:
            stats = cc3d.statistics(labeled_holes)
            for i, size in enumerate(stats['voxel_counts'][1:]):
                if size < self.cfg.HOLE_FILL_THRESHOLD:
                    mask[labeled_holes == (i + 1)] = 1
        return mask

    def morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        kernel = morphology.ball(self.cfg.MORPH_KERNEL)
        closed = morphology.binary_closing(mask, footprint=kernel)
        opened = morphology.binary_opening(closed, footprint=kernel)
        return opened.astype(np.uint8)

    def iterative_stabilization(self, mask: np.ndarray, verbose: bool = True) -> np.ndarray:
        prev_n_components = None
        stable_count = 0
        for round_idx in range(self.cfg.STABILIZATION_ROUNDS):
            mask = self.remove_small_components(mask)
            mask = self.fill_holes(mask)
            mask = self.morphological_operations(mask)
            labeled = cc3d.connected_components(mask.astype(np.uint8))
            n_components = labeled.max()
            if verbose and round_idx % 10 == 0:
                fg_pct = mask.sum() / mask.size * 100
                print(f"      Round {round_idx:2d}: {n_components} components, {fg_pct:.2f}% fg")
            if prev_n_components == n_components:
                stable_count += 1
                if stable_count >= 5:
                    if verbose:
                        print(f"      ✓ Converged at round {round_idx}")
                    break
            else:
                stable_count = 0
            prev_n_components = n_components
        return mask.astype(np.uint8)

    def __call__(self, pred_volume: np.ndarray, verbose: bool = True) -> np.ndarray:
        mask = (pred_volume == 1).astype(np.uint8)
        mask = self.iterative_stabilization(mask, verbose=verbose)
        return mask
