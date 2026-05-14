import torch
import os
import numpy as np

# =============================================================================
# CELL 2: STRATIFIED VOLUME FOLDS
# =============================================================================

def make_stratified_volume_folds(
    csv_path: Path,
    images_dir: Path,
    labels_dir: Path,
    n_splits: int = 3,
    seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """Create stratified volume-level folds using scroll_id."""
    df = pd.read_csv(csv_path)
    
    valid_mask = df['id'].apply(
        lambda x: (images_dir / f"{x}.tif").exists() and (labels_dir / f"{x}.tif").exists()
    )
    df = df[valid_mask].reset_index(drop=True)
    
    print(f"Found {len(df)} valid volumes")
    
    if 'scroll_id' in df.columns:
        strat_col = df['scroll_id'].values
    else:
        strat_col = df['id'].apply(lambda x: x.split('_')[0] if '_' in x else 'unknown').values
    
    print(f"Scroll distribution: {pd.Series(strat_col).value_counts().to_dict()}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df['id'], strat_col)):
        train_ids = df.iloc[train_idx]['id'].tolist()
        val_ids = df.iloc[val_idx]['id'].tolist()
        assert len(set(train_ids) & set(val_ids)) == 0, "Train/val overlap!"
        splits.append((train_ids, val_ids))
        print(f"Fold {fold}: Train={len(train_ids)}, Val={len(val_ids)}")
    
    return splits

# Create folds
train_csv = cfg.DATA_ROOT / "train.csv"
train_images = cfg.DATA_ROOT / "train_images"
train_labels = cfg.DATA_ROOT / "train_labels"

if train_csv.exists():
    FOLD_SPLITS = make_stratified_volume_folds(
        train_csv, train_images, train_labels,
        n_splits=cfg.N_FOLDS, seed=cfg.CV_SEED
    )
else:
    print("train.csv not found - test mode")
    FOLD_SPLITS = []

# =============================================================================
# CELL 7: CHECKPOINT SAVE & LOAD (FIXED)
# =============================================================================

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice, history, cfg):
    """
    Save training checkpoint with all state needed to resume.
    
    Saves:
    - Model weights
    - Optimizer state (momentum, etc.)
    - Scheduler state (current LR position)
    - Current epoch
    - Best dice score
    - Training history
    - Config for reproducibility
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
        'history': history,
        'config': {
            'features': cfg.FEATURES,
            'n_blocks': cfg.N_BLOCKS,
            'patch_size': cfg.PATCH_SIZE,
            'batch_size': cfg.BATCH_SIZE,
            'learning_rate': cfg.LEARNING_RATE,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cuda'):
    """
    Load training checkpoint to resume training.
    
    Returns:
        start_epoch: Epoch to resume from (checkpoint epoch + 1)
        best_dice: Best dice score so far
        history: Training history list
    """
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Load model weights (handle torch.compile prefix)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    # Load optimizer state
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Optimizer state loaded")
    
    # Load scheduler state
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Scheduler state loaded")
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_dice = checkpoint.get('best_dice', 0)
    history = checkpoint.get('history', [])
    
    print(f"  Resuming from epoch {start_epoch}")
    print(f"  Best dice so far: {best_dice:.4f}")
    print(f"  History entries: {len(history)}")
    
    # Show config diff if available
    if 'config' in checkpoint:
        saved_cfg = checkpoint['config']
        print(f"  Saved config: patch={saved_cfg.get('patch_size')}, bs={saved_cfg.get('batch_size')}")
    
    return start_epoch, best_dice, history


def get_latest_checkpoint(checkpoint_dir, fold=None, load_dir=None):
    # Build list of directories to search (save dir first, then load dir)
    search_dirs = []
    
    save_dir = Path(checkpoint_dir)
    if fold is not None:
        save_dir = save_dir / f"fold_{fold}"
    search_dirs.append(save_dir)
    
    if load_dir:
        load_path = Path(load_dir)
        if fold is not None:
            load_path = load_path / f"fold_{fold}"
        search_dirs.append(load_path)
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Priority 1: last_checkpoint.pth (most recent state)
        last_ckpt = search_dir / "last_checkpoint.pth"
        if last_ckpt.exists():
            print(f"Found last_checkpoint.pth in {search_dir}")
            return last_ckpt
        
        # Priority 2: checkpoint_epoch_*.pth (periodic saves) - find highest epoch
        epoch_ckpts = list(search_dir.glob("checkpoint_epoch_*.pth"))
        if epoch_ckpts:
            def get_epoch(p):
                try:
                    return int(p.stem.split('_')[-1])
                except:
                    return 0
            epoch_ckpts.sort(key=get_epoch, reverse=True)
            print(f"Found checkpoint_epoch_{get_epoch(epoch_ckpts[0])}.pth in {search_dir}")
            return epoch_ckpts[0]
        
        # Priority 3: best_model.pth (fallback)
        best_ckpt = search_dir / "best_model.pth"
        if best_ckpt.exists():
            print(f"Found best_model.pth in {search_dir} (no last/periodic checkpoint)")
            return best_ckpt
    
    print(f"No checkpoint found for fold {fold}")
    return None


print(f"Save dir: {cfg.CHECKPOINT_DIR}")
print(f"Load from: {cfg.LOAD_CHECKPOINT}")

# =============================================================================
# SUBMISSION FORMAT: TIFF MASKS
# =============================================================================

import zipfile

def save_mask_tiff(mask, output_path):
    """Save mask as TIFF - NO compression to match expected file size."""
    mask_uint8 = mask.astype(np.uint8)
    # No compression - raw TIFF
    tifffile.imwrite(str(output_path), mask_uint8, compression=None)
    print(f"  Saved: {output_path} ({mask_uint8.shape}, dtype={mask_uint8.dtype})")

def create_submission_zip(mask_dir, output_zip):
    """Create submission.zip containing all mask TIFFs."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for tif_path in sorted(Path(mask_dir).glob("*.tif")):
            zf.write(tif_path, tif_path.name)
            print(f"  Added to zip: {tif_path.name}")
    print(f"Submission zip: {output_zip}")

print("TIFF submission functions loaded (no compression)")