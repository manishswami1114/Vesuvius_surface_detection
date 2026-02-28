"""
Vesuvius Challenge Experiment Lab - Trainer
==========================================
Complete training pipeline with validation, checkpointing, and visualization.
"""

import os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import ExperimentConfig
from models import build_model, count_parameters
from losses import build_loss, CombinedLoss
from visualization import ExperimentVisualizer


class Metrics:
    """Compute segmentation metrics."""
    
    @staticmethod
    def dice_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        smooth: float = 1e-5
    ) -> float:
        """Compute Dice score."""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target = target.float()
        
        if mask is not None:
            pred_binary = pred_binary * mask
            target = target * mask
        
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        smooth: float = 1e-5
    ) -> float:
        """Compute IoU/Jaccard score."""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target = target.float()
        
        if mask is not None:
            pred_binary = pred_binary * mask
            target = target * mask
        
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def precision_recall(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        smooth: float = 1e-5
    ) -> Tuple[float, float]:
        """Compute precision and recall."""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target = target.float()
        
        if mask is not None:
            pred_binary = pred_binary * mask
            target = target * mask
        
        pred_flat = pred_binary.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        
        return precision.item(), recall.item()


class Trainer:
    """
    Training manager for Vesuvius experiments.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Visualization
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        self.config = config
        self.device = config.get_device()
        
        # Model
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.training.use_amp and self.device.type == 'cuda' else None
        self.use_amp = config.training.use_amp and self.device.type in ['cuda', 'mps']
        
        # Checkpointing
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualization
        self.visualizer = ExperimentVisualizer(
            save_dir=config.vis_dir,
            experiment_name=config.experiment_name,
            architecture=config.model.architecture,
            loss_type=config.loss.loss_type
        )
        
        # State
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Print model info
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {config.get_experiment_id()}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Architecture: {config.model.architecture}")
        print(f"Loss Function: {config.loss.loss_type}")
        print(f"Parameters: {count_parameters(model):,}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        params = self.model.parameters()
        
        if self.config.training.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.training.scheduler
        
        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        elif scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.config.training.num_epochs // 3),
                gamma=0.1
            )
        elif scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=2,
                min_lr=self.config.training.min_lr
            )
        elif scheduler_type == "polynomial":
            # nnU-Net style polynomial decay
            def poly_lr(epoch):
                return (1 - epoch / self.config.training.num_epochs) ** 0.9
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, poly_lr)
        else:
            return None
    
    def train_epoch(self) -> Tuple[float, float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        loss_breakdown = {}
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (with AMP if available)
            if self.use_amp and self.device.type == 'cuda':
                with autocast():
                    outputs = self.model(images)
                    loss_result = self.loss_fn(outputs, labels, masks)
            else:
                outputs = self.model(images)
                loss_result = self.loss_fn(outputs, labels, masks)
            
            # Handle combined loss return
            if isinstance(loss_result, tuple):
                loss, batch_breakdown = loss_result
                for k, v in batch_breakdown.items():
                    loss_breakdown[k] = loss_breakdown.get(k, 0) + v
            else:
                loss = loss_result
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
                self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                if isinstance(outputs, list):
                    outputs = outputs[0]  # Use main output for metrics
                dice = Metrics.dice_score(outputs, labels, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        avg_breakdown = {k: v / num_batches for k, v in loss_breakdown.items()}
        
        return avg_loss, avg_dice, avg_breakdown
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0
        
        # Store samples for visualization
        sample_images = []
        sample_labels = []
        sample_preds = []
        
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            outputs = self.model(images)
            
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            loss_result = self.loss_fn(outputs, labels, masks)
            loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
            
            dice = Metrics.dice_score(outputs, labels, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
            
            # Store first few batches for visualization
            if len(sample_images) < 4:
                sample_images.append(images[:1].cpu())
                sample_labels.append(labels[:1].cpu())
                sample_preds.append(torch.sigmoid(outputs[:1]).cpu())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_dice = total_dice / num_batches
        
        # Concatenate samples
        if sample_images:
            sample_images = torch.cat(sample_images, dim=0)
            sample_labels = torch.cat(sample_labels, dim=0)
            sample_preds = torch.cat(sample_preds, dim=0)
        
        return avg_loss, avg_dice, sample_images, sample_labels, sample_preds
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / f"{self.config.get_experiment_id()}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.config.get_experiment_id()}_best.pt"
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (Dice: {self.best_val_dice:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_dice = checkpoint['best_val_dice']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        start_time = time.time()
        
        print(f"\nStarting training for {self.config.training.num_epochs} epochs...")
        print(f"Visualization dir: {self.config.vis_dir}")
        print(f"Checkpoint dir: {self.checkpoint_dir}\n")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_dice, loss_breakdown = self.train_epoch()
            
            # Validate
            val_loss, val_dice, sample_imgs, sample_lbls, sample_preds = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()
            
            # Log metrics
            self.visualizer.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_dice=train_dice,
                val_dice=val_dice,
                lr=current_lr,
                loss_breakdown=loss_breakdown
            )
            
            # Check for best model
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if self.config.training.save_best_only:
                if is_best:
                    self.save_checkpoint(is_best=True)
            else:
                self.save_checkpoint(is_best=is_best)
            
            # Visualization
            if epoch % self.config.vis_interval == 0:
                self.visualizer.plot_predictions(
                    sample_imgs, sample_lbls, sample_preds, epoch
                )
                self.visualizer.plot_3d_slices(
                    sample_imgs, sample_lbls, sample_preds, epoch
                )
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs - 1}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            print(f"  Best Dice: {self.best_val_dice:.4f} @ epoch {self.best_epoch}")
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        
        # Final visualizations
        self.visualizer.plot_training_curves()
        self.visualizer.plot_final_summary(
            best_val_dice=self.best_val_dice,
            best_epoch=self.best_epoch,
            total_time=total_time
        )
        self.visualizer.save_metrics_csv()
        
        # Final histogram
        if sample_preds is not None:
            self.visualizer.plot_histogram_comparison(
                sample_preds, sample_lbls, self.current_epoch
            )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best Val Dice: {self.best_val_dice:.4f} @ epoch {self.best_epoch}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Visualizations saved to: {self.config.vis_dir}")
        print(f"{'='*60}\n")
        
        return {
            'best_val_dice': self.best_val_dice,
            'best_epoch': self.best_epoch,
            'total_time': total_time,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss
        }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run a complete experiment from config."""
    from dataset import create_data_splits, create_dataloaders
    
    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create data splits
    csv_path = os.path.join(config.data.data_root, config.data.train_csv)
    train_ids, val_ids = create_data_splits(
        csv_path,
        train_ratio=config.data.train_ratio,
        max_samples=config.data.max_samples,
        seed=config.seed
    )
    
    print(f"Train samples: {len(train_ids)}")
    print(f"Val samples: {len(val_ids)}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, train_ids, val_ids)
    
    # Build model and loss
    model = build_model(config)
    loss_fn = build_loss(config)
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train
    results = trainer.train()
    
    return results