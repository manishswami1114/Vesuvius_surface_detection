"""
Vesuvius Challenge Experiment Lab - Visualization
================================================
Comprehensive visualization for training monitoring and result analysis.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# Custom colormap for segmentation overlay
OVERLAY_CMAP = LinearSegmentedColormap.from_list(
    'overlay', 
    [(0, 0, 0, 0), (0, 1, 0, 0.7)],  # Transparent to green
    N=256
)


class ExperimentVisualizer:
    """
    Comprehensive visualization for Vesuvius experiments.
    Saves all plots with architecture and loss function info.
    """
    
    def __init__(
        self,
        save_dir: str,
        experiment_name: str,
        architecture: str,
        loss_type: str
    ):
        self.save_dir = Path(save_dir)
        self.experiment_name = experiment_name
        self.architecture = architecture
        self.loss_type = loss_type
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_dices: List[float] = []
        self.val_dices: List[float] = []
        self.learning_rates: List[float] = []
        self.epochs: List[int] = []
        
        # Component losses tracking
        self.loss_components: Dict[str, List[float]] = {}
        
        # Experiment ID for filenames
        self.exp_id = f"{experiment_name}_{architecture}_{loss_type}"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_title_suffix(self) -> str:
        """Generate title suffix with experiment info."""
        return f"\nArch: {self.architecture} | Loss: {self.loss_type}"
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure with experiment metadata in filename."""
        filename = f"{self.exp_id}_{name}_{self.timestamp}.png"
        filepath = self.save_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {filepath}")
        return filepath
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_dice: float,
        val_dice: float,
        lr: float,
        loss_breakdown: Optional[Dict[str, float]] = None
    ):
        """Log metrics for one epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_dices.append(train_dice)
        self.val_dices.append(val_dice)
        self.learning_rates.append(lr)
        
        if loss_breakdown:
            for key, value in loss_breakdown.items():
                if key not in self.loss_components:
                    self.loss_components[key] = []
                self.loss_components[key].append(value)
    
    def plot_training_curves(self) -> str:
        """Plot loss and Dice curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Training Progress{self._get_title_suffix()}", fontsize=14)
        
        # Loss curves
        ax1 = axes[0, 0]
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dice curves
        ax2 = axes[0, 1]
        ax2.plot(self.epochs, self.train_dices, 'b-', label='Train Dice', linewidth=2)
        ax2.plot(self.epochs, self.val_dices, 'r-', label='Val Dice', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Score Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Learning rate
        ax3 = axes[1, 0]
        ax3.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Loss components (if available)
        ax4 = axes[1, 1]
        if self.loss_components:
            for name, values in self.loss_components.items():
                if name != 'total' and len(values) == len(self.epochs):
                    ax4.plot(self.epochs, values, label=name, linewidth=1.5)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Component')
            ax4.set_title('Loss Components')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No component breakdown available',
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Loss Components')
        
        plt.tight_layout()
        return self._save_figure(fig, 'training_curves')
    
    def plot_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        epoch: int,
        num_samples: int = 4,
        slice_idx: Optional[int] = None
    ) -> str:
        """
        Plot predictions vs ground truth.
        
        Args:
            images: [B, 1, D, H, W] input images
            labels: [B, 1, D, H, W] ground truth
            predictions: [B, 1, D, H, W] model predictions (probabilities)
            epoch: Current epoch number
            num_samples: Number of samples to visualize
            slice_idx: Which depth slice to show (None = middle)
        """
        num_samples = min(num_samples, images.shape[0])
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        fig.suptitle(f"Predictions (Epoch {epoch}){self._get_title_suffix()}", fontsize=14)
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            img = images[i, 0].cpu().numpy()
            lbl = labels[i, 0].cpu().numpy()
            pred = predictions[i, 0].cpu().numpy()
            
            # Select slice
            D = img.shape[0]
            s = slice_idx if slice_idx is not None else D // 2
            
            # Image
            axes[i, 0].imshow(img[s], cmap='gray')
            axes[i, 0].set_title(f'Input (slice {s})')
            axes[i, 0].axis('off')
            
            # Ground truth
            axes[i, 1].imshow(lbl[s], cmap='viridis', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            axes[i, 2].imshow(pred[s], cmap='viridis', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Prediction (mean: {pred.mean():.3f})')
            axes[i, 2].axis('off')
            
            # Overlay
            axes[i, 3].imshow(img[s], cmap='gray')
            axes[i, 3].imshow(pred[s], cmap=OVERLAY_CMAP, alpha=0.7)
            axes[i, 3].contour(lbl[s], levels=[0.5], colors='red', linewidths=1)
            axes[i, 3].set_title('Overlay (pred=green, GT=red)')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        return self._save_figure(fig, f'predictions_epoch{epoch:03d}')
    
    def plot_3d_slices(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        prediction: torch.Tensor,
        epoch: int,
        num_slices: int = 8
    ) -> str:
        """Plot multiple depth slices in a grid."""
        img = image[0, 0].cpu().numpy()
        lbl = label[0, 0].cpu().numpy()
        pred = prediction[0, 0].cpu().numpy()
        
        D = img.shape[0]
        slice_indices = np.linspace(0, D - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, num_slices, figsize=(2.5 * num_slices, 7))
        fig.suptitle(f"3D Volume Slices (Epoch {epoch}){self._get_title_suffix()}", fontsize=12)
        
        for j, s in enumerate(slice_indices):
            # Image
            axes[0, j].imshow(img[s], cmap='gray')
            axes[0, j].set_title(f'z={s}', fontsize=8)
            axes[0, j].axis('off')
            
            # Ground truth
            axes[1, j].imshow(lbl[s], cmap='viridis', vmin=0, vmax=1)
            axes[1, j].axis('off')
            
            # Prediction
            axes[2, j].imshow(pred[s], cmap='viridis', vmin=0, vmax=1)
            axes[2, j].axis('off')
        
        # Row labels
        axes[0, 0].set_ylabel('Input', fontsize=10)
        axes[1, 0].set_ylabel('Ground Truth', fontsize=10)
        axes[2, 0].set_ylabel('Prediction', fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, f'3d_slices_epoch{epoch:03d}')
    
    def plot_histogram_comparison(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        epoch: int
    ) -> str:
        """Plot histogram of predictions for positive vs negative voxels."""
        pred = predictions.cpu().numpy().flatten()
        lbl = labels.cpu().numpy().flatten()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Prediction Distribution (Epoch {epoch}){self._get_title_suffix()}", fontsize=12)
        
        # Histogram by class
        ax1 = axes[0]
        ax1.hist(pred[lbl == 0], bins=50, alpha=0.7, label='Background', density=True)
        ax1.hist(pred[lbl == 1], bins=50, alpha=0.7, label='Surface', density=True)
        ax1.set_xlabel('Prediction Probability')
        ax1.set_ylabel('Density')
        ax1.set_title('Prediction Distribution by Class')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Overall histogram
        ax2 = axes[1]
        ax2.hist(pred, bins=50, alpha=0.7, density=True)
        ax2.axvline(x=0.5, color='r', linestyle='--', label='Threshold=0.5')
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Overall Prediction Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, f'histogram_epoch{epoch:03d}')
    
    def plot_final_summary(
        self,
        best_val_dice: float,
        best_epoch: int,
        total_time: float
    ) -> str:
        """Plot final training summary."""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        fig.suptitle(f"Experiment Summary{self._get_title_suffix()}", fontsize=14)
        
        # Training curves (large)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best @ {best_epoch}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Dice curves
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(self.epochs, self.train_dices, 'b-', label='Train', linewidth=2)
        ax2.plot(self.epochs, self.val_dices, 'r-', label='Val', linewidth=2)
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        ax2.axhline(y=best_val_dice, color='g', linestyle=':', alpha=0.7, label=f'Best: {best_val_dice:.4f}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title('Dice Score Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
        
        # Summary stats (text box)
        ax3 = fig.add_subplot(gs[0:2, 2])
        ax3.axis('off')
        
        summary_text = f"""
EXPERIMENT SUMMARY
{'='*30}

Architecture: {self.architecture}
Loss Function: {self.loss_type}
Experiment: {self.experiment_name}

RESULTS
{'='*30}

Best Val Dice: {best_val_dice:.4f}
Best Epoch: {best_epoch}
Final Train Loss: {self.train_losses[-1]:.4f}
Final Val Loss: {self.val_losses[-1]:.4f}

TRAINING INFO
{'='*30}

Total Epochs: {len(self.epochs)}
Total Time: {total_time/60:.1f} min
Time/Epoch: {total_time/len(self.epochs):.1f} sec
        """
        ax3.text(0.1, 0.9, summary_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Learning rate
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(self.epochs, self.learning_rates, 'g-', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('LR')
        ax4.set_title('Learning Rate')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        # Loss components
        ax5 = fig.add_subplot(gs[2, 1:])
        if self.loss_components:
            for name, values in self.loss_components.items():
                if name != 'total':
                    ax5.plot(self.epochs, values, label=name, linewidth=1.5)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Loss')
            ax5.set_title('Loss Components')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'Single loss function used',
                    ha='center', va='center', transform=ax5.transAxes)
        
        plt.tight_layout()
        return self._save_figure(fig, 'final_summary')
    
    def save_metrics_csv(self):
        """Save training metrics to CSV."""
        import csv
        
        filepath = self.save_dir / f"{self.exp_id}_metrics_{self.timestamp}.csv"
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            headers = ['epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'lr']
            headers.extend(self.loss_components.keys())
            writer.writerow(headers)
            
            # Data
            for i, epoch in enumerate(self.epochs):
                row = [
                    epoch,
                    self.train_losses[i],
                    self.val_losses[i],
                    self.train_dices[i],
                    self.val_dices[i],
                    self.learning_rates[i]
                ]
                for key in self.loss_components.keys():
                    row.append(self.loss_components[key][i] if i < len(self.loss_components[key]) else '')
                writer.writerow(row)
        
        print(f"Saved metrics: {filepath}")
        return filepath


def visualize_sample_data(
    images: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
    num_samples: int = 4
) -> None:
    """Visualize sample data for EDA."""
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle("Sample Data Visualization", fontsize=14)
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        lbl = labels[i]
        
        mid_slice = img.shape[0] // 2
        
        axes[i, 0].imshow(img[mid_slice], cmap='gray')
        axes[i, 0].set_title(f'Image (z={mid_slice})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(lbl[mid_slice], cmap='tab10')
        axes[i, 1].set_title('Label')
        axes[i, 1].axis('off')
        
        # Label distribution
        unique, counts = np.unique(lbl, return_counts=True)
        axes[i, 2].bar(unique, counts / counts.sum())
        axes[i, 2].set_xlabel('Label')
        axes[i, 2].set_ylabel('Fraction')
        axes[i, 2].set_title('Label Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


if __name__ == "__main__":
    # Demo visualization
    vis = ExperimentVisualizer(
        save_dir="./test_vis",
        experiment_name="demo",
        architecture="UNet3D",
        loss_type="Dice"
    )
    
    # Simulate training
    for epoch in range(10):
        vis.log_epoch(
            epoch=epoch,
            train_loss=1.0 / (epoch + 1),
            val_loss=1.1 / (epoch + 1),
            train_dice=0.3 + epoch * 0.05,
            val_dice=0.25 + epoch * 0.05,
            lr=0.001 * (0.9 ** epoch)
        )
    
    vis.plot_training_curves()
    print("Demo visualization complete!")