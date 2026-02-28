from PIL import Image
import numpy as np
from pathlib import Path
import tifffile

def load_3d_tiff(filepath: str) -> np.ndarray:
    """Load 3D tiff into numpy array (D, H, W)."""
    try:
        arr = tifffile.imread(filepath)
        return arr
    except Exception:
        frames = []
        with Image.open(filepath) as img:
            try:
                while True:
                    frames.append(np.array(img.copy()))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass
        return np.stack(frames, axis=0)

def save_3d_tiff(volume: np.ndarray, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), volume.astype(np.uint8))
