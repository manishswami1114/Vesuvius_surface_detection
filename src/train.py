from .config import *
from .losses import *
from .models import *
from .augmentations import *
from .dataset import *
from .utils import *

# =============================================================================
# CELL 8: VALIDATION
# =============================================================================

def compute_dice(pred, gt):
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return (2 * inter + 1e-5) / (union + 1e-5)


@torch.no_grad()
def validate_fast(model, loader, device, use_bf16=True):
    """Fast patch-based validation (Dice only)."""
    model.eval()
    total_dice = 0
    n = 0
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    for batch in loader:
        images = batch['image'].to(device, dtype=dtype)
        labels = batch['label'].numpy()
        ignore = batch['ignore_mask'].numpy()
        
        with autocast(device_type='cuda', dtype=dtype):
            out = model(images)
            if isinstance(out, dict): out = out['output']
            probs = torch.sigmoid(out).float().cpu().numpy()
        
        for b in range(images.shape[0]):
            pred = (probs[b,0] > 0.5).astype(np.uint8)
            tgt = labels[b,0].astype(np.uint8)
            ign = ignore[b,0] > 0.5
            pred[ign] = 0
            tgt[ign] = 0
            total_dice += compute_dice(pred, tgt)
            n += 1
    
    return total_dice / max(n, 1)

print("Validation ready")

# =============================================================================
# CELL 9: TRAINING LOOP (FIXED - saves last checkpoint every epoch)
# =============================================================================

import sys
import time

def get_warmup_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr


def train_fold_v11(fold: int, train_ids: List[str], val_ids: List[str], resume_from: Path = None) -> Dict:
    """
    Train fold with checkpoint save/load support.
    
    Checkpoint strategy:
    - last_checkpoint.pth: Saved EVERY epoch (for exact resume)
    - best_model.pth: Saved when validation improves
    - checkpoint_epoch_N.pth: Saved every SAVE_EVERY epochs (backup)
    
    Args:
        fold: Fold number
        train_ids: Training volume IDs
        val_ids: Validation volume IDs
        resume_from: Path to checkpoint to resume from (optional)
    """
    print("="*70)
    print(f"FOLD {fold} TRAINING")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)}")
    print("="*70)

    train_images = cfg.DATA_ROOT / "train_images"
    train_labels = cfg.DATA_ROOT / "train_labels"

    # Datasets
    train_ds = VesuviusDatasetV11(train_ids, train_images, train_labels,
                                   patch_size=cfg.PATCH_SIZE, is_train=True,
                                   patches_per_volume=cfg.PATCHES_PER_VOLUME)
    val_ds = VesuviusDatasetV11(val_ids, train_images, train_labels,
                                 patch_size=cfg.PATCH_SIZE, is_train=False, patches_per_volume=4)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=cfg.PREFETCH_FACTOR)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = TopoPreservingUNet3D(
        features=cfg.FEATURES, n_blocks=cfg.N_BLOCKS,
        use_attention=cfg.USE_ATTENTION, use_hybrid_conv=cfg.USE_HYBRID_CONV,
        use_surface_refinement=cfg.USE_SURFACE_REFINEMENT,
        use_deep_supervision=cfg.USE_DEEP_SUPERVISION,
    ).to(cfg.DEVICE)

    print(f"Model: {count_params(model)/1e6:.1f}M params")

    # Loss & Optimizer
    criterion = CombinedLossV11(dice_w=cfg.DICE_WEIGHT, bce_w=cfg.BCE_WEIGHT,
                                 cldice_w=cfg.CLDICE_WEIGHT, surface_w=cfg.SURFACE_WEIGHT,
                                 topo_w=cfg.TOPO_WEIGHT)
    gpu_augment = GPUAugmentation3D(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS_PER_FOLD - cfg.WARMUP_EPOCHS, eta_min=cfg.ETA_MIN)

    # Checkpoint directory
    fold_dir = cfg.CHECKPOINT_DIR / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0
    history = []
    
    if resume_from and Path(resume_from).exists():
        start_epoch, best_dice, history = load_checkpoint(
            resume_from, model, optimizer, scheduler, cfg.DEVICE)
    elif cfg.LOAD_CHECKPOINT:
        # Try auto-find latest checkpoint for this fold
        # Search in both save dir (working) and load dir (input dataset)
        latest = get_latest_checkpoint(cfg.CHECKPOINT_DIR, fold, load_dir=cfg.LOAD_CHECKPOINT)
        if latest:
            start_epoch, best_dice, history = load_checkpoint(
                latest, model, optimizer, scheduler, cfg.DEVICE)

    # Compile model (after loading checkpoint)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    # Precision
    use_bf16 = cfg.USE_BFLOAT16 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"Precision: {'bfloat16' if use_bf16 else 'float32'}")
    print(f"Starting from epoch {start_epoch + 1}")
    print(f"Checkpoint save strategy:")
    print(f"  - last_checkpoint.pth: Every epoch")
    print(f"  - best_model.pth: When validation improves")
    print(f"  - checkpoint_epoch_N.pth: Every {cfg.SAVE_EVERY} epochs")

    for epoch in range(start_epoch, cfg.EPOCHS_PER_FOLD):
        t0 = time.time()
        model.train()

        # Warmup LR
        if epoch < cfg.WARMUP_EPOCHS:
            for pg in optimizer.param_groups:
                pg['lr'] = get_warmup_lr(epoch, cfg.WARMUP_EPOCHS, cfg.LEARNING_RATE)

        total_loss = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"F{fold} E{epoch+1}", file=sys.stdout, leave=False)
        for batch in pbar:
            images = batch['image'].to(cfg.DEVICE, dtype=dtype)
            labels = batch['label'].to(cfg.DEVICE, dtype=dtype)
            ignore = batch['ignore_mask'].to(cfg.DEVICE, dtype=dtype)

            # GPU augmentation (no CPU bottleneck)
            images, labels, ignore = gpu_augment(images, labels, ignore)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=dtype):
                out = model(images)
                losses = criterion(out, labels, ignore, epoch)

            losses['total'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            optimizer.step()

            total_loss += losses['total'].detach()
            n += 1
            # Only log every 5 batches to avoid CUDA sync
            if n % 5 == 0:
                pbar.set_postfix(loss=f"{losses['total'].item():.3f}", gnorm=f"{grad_norm.item():.2f}")

        if epoch >= cfg.WARMUP_EPOCHS:
            scheduler.step()

        train_loss = total_loss.item() / n
        dt = time.time() - t0

        # Validation
        val_dice = 0
        if (epoch + 1) % cfg.VAL_EVERY == 0:
            val_dice = validate_fast(model, val_loader, cfg.DEVICE, use_bf16)

            if val_dice > best_dice:
                best_dice = val_dice
                save_checkpoint(fold_dir / 'best_model.pth', model, optimizer, scheduler,
                               epoch, best_dice, history, cfg)
                print(f"  >>> New best Dice: {val_dice:.4f}")

        # Log
        lr = optimizer.param_groups[0]['lr']
        log = f"F{fold} E{epoch+1}/{cfg.EPOCHS_PER_FOLD} | {dt:.1f}s | Loss: {train_loss:.4f} | LR: {lr:.1e}"
        if val_dice > 0:
            log += f" | Dice: {val_dice:.4f}"
        print(log)

        history.append({
            'epoch': epoch, 'train_loss': train_loss,
            'lr': lr, 'val_dice': val_dice,
        })

        # =====================================================================
        # ALWAYS save last_checkpoint.pth (for exact resume from any epoch)
        # =====================================================================
        save_checkpoint(fold_dir / 'last_checkpoint.pth', model, optimizer, scheduler,
                       epoch, best_dice, history, cfg)

        # Periodic checkpoint (backup)
        if (epoch + 1) % cfg.SAVE_EVERY == 0:
            save_checkpoint(fold_dir / f'checkpoint_epoch_{epoch+1}.pth',
                           model, optimizer, scheduler, epoch, best_dice, history, cfg)

    print(f"\nFold {fold} complete. Best Dice: {best_dice:.4f}")

    # Save final checkpoint
    save_checkpoint(fold_dir / 'final_model.pth', model, optimizer, scheduler,
                   cfg.EPOCHS_PER_FOLD - 1, best_dice, history, cfg)

    # Save history
    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return {'fold': fold, 'best_dice': best_dice, 'history': history}

print("Training loop ready (FIXED)")
print("Checkpoint saves:")
print("  - last_checkpoint.pth: EVERY epoch (exact resume)")
print("  - best_model.pth: When validation improves")
print("  - checkpoint_epoch_N.pth: Every SAVE_EVERY epochs")

# =============================================================================
# CELL 10: RUN TRAINING
# =============================================================================

def run_3fold_cv(resume=False):
    """
    Run 3-fold CV.
    
    Args:
        resume: If True, auto-resume from latest checkpoint for each fold
                Priority: last_checkpoint.pth > checkpoint_epoch_*.pth > best_model.pth
    """
    if not FOLD_SPLITS:
        print("No fold splits available")
        return
    
    results = []
    
    for fold, (train_ids, val_ids) in enumerate(FOLD_SPLITS):
        gc.collect()
        torch.cuda.empty_cache()
        
        resume_path = None
        if resume:
            # Search in both save dir and load dir
            resume_path = get_latest_checkpoint(
                cfg.CHECKPOINT_DIR, 
                fold, 
                load_dir=cfg.LOAD_CHECKPOINT
            )
        
        result = train_fold_v11(fold, train_ids, val_ids, resume_from=resume_path)
        results.append(result)
    
    print("\n" + "="*70)
    print("3-FOLD CV COMPLETE")
    print("="*70)
    for r in results:
        print(f"  Fold {r['fold']}: Dice={r['best_dice']:.4f}")
    print(f"  Mean Dice: {np.mean([r['best_dice'] for r in results]):.4f}")
    print("="*70)
    
    return results


# =============================================================================
# USAGE:
# =============================================================================
# Fresh start (no checkpoints):
#   cv_results = run_3fold_cv(resume=False)
#
# Resume from checkpoints (priority: last > periodic > best):
#   if __name__ == "__main__":
    cv_results = run_3fold_cv(resume=True)
#
# Example: If stopped at epoch 47 with best at 45:
#   - last_checkpoint.pth contains epoch 46 state
#   - Will resume from epoch 47 (not 45 or 40)
# =============================================================================

if __name__ == "__main__":
    cv_results = run_3fold_cv(resume=True)