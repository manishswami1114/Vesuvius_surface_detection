import numpy as np

def normalize_volume(volume: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    p1, p99 = np.percentile(volume, [pmin, pmax])
    vol = np.clip(volume, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol.astype('float32')
