# src/vesuvius/config.py
import os
from pathlib import Path
import yaml

class CFG:
    # defaults (overridden by YAML / env)
    DATA_ROOT = Path(os.environ.get('DATA_ROOT', '/Data/vesuvius-challenge-surface-detection'))
    TRAIN_CSV = DATA_ROOT / 'train.csv'
    TEST_CSV = DATA_ROOT / 'test.csv'
    TEST_IMAGES = DATA_ROOT / 'test_images'
    TRAIN_IMAGES = DATA_ROOT / 'train_images'
    TRAIN_LABELS = DATA_ROOT / 'train_labels'

    MODEL_PATH = Path(os.environ.get('MODEL_PATH', './models/segresnet_maxperf_final.pth'))
    OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/Data/Output/submission/submission_tifs'))
    SUBMISSION_ZIP = Path(os.environ.get('SUBMISSION_ZIP', '/Data/Output/submissionsubmission.zip'))

    IN_CHANNELS = 1
    OUT_CHANNELS = 3
    INIT_FILTERS = 48
    IMG_SIZE = (128,128,128)

    SW_BATCH_SIZE = 2
    SW_OVERLAP = 0.5

    MIN_COMPONENT_SIZE = 3000
    MAX_COMPONENTS = 8
    HOLE_FILL_THRESHOLD = 3000
    MORPH_KERNEL = 2
    STABILIZATION_ROUNDS = 50

    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 100
    PATCH_SIZE = (128,128,128)
    DEVICE = 'cuda' if (os.environ.get('CUDA_VISIBLE_DEVICES','') != '') else 'cpu'

    CHECKPOINT_DIR = Path(os.environ.get('CHECKPOINT_DIR', '/Data/Output/checkpoints'))
    CSV_LOG = Path(os.environ.get('CSV_LOG', '/Data/Output/submission/training_log.csv'))

def load_yaml_config(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    for k, v in data.items():
        if hasattr(CFG, k):
            setattr(CFG, k, v)
    return CFG
