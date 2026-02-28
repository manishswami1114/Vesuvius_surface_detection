#!/usr/bin/env python3
"""
Vesuvius Challenge Experiment Lab - Main Runner
===============================================
Run experiments with different architectures and loss functions.

Usage:
    python run_experiment.py --arch unet3d --loss dice
    python run_experiment.py --arch attention_unet3d --loss combined
    python run_experiment.py --run-all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import ExperimentConfig, DataConfig, ModelConfig, LossConfig, TrainingConfig


def create_experiment_config(
    arch: str,
    loss: str,
    data_root: str,
    max_samples: int = 100,
    num_epochs: int = 10,
    batch_size: int = 4,
    patch_size: tuple = (64, 64, 64)
) -> ExperimentConfig:
    """Create experiment configuration."""
    
    # Architecture-specific settings
    use_residual = arch == "resunet3d"
    use_attention = arch == "attention_unet3d"
    
    config = ExperimentConfig(
        experiment_name=f"vesuvius_{arch}_{loss}",
        seed=42,
        data=DataConfig(
            data_root=data_root,
            patch_size=patch_size,
            patches_per_volume=8,
            max_samples=max_samples,
            use_augmentation=True
        ),
        model=ModelConfig(
            architecture=arch,
            in_channels=1,
            out_channels=1,
            init_features=32,
            depth=4,
            use_residual=use_residual,
            use_attention=use_attention,
            use_deep_supervision=False,
            dropout_rate=0.1,
            norm_type="batch"
        ),
        loss=LossConfig(
            loss_type=loss,
            dice_weight=0.5,
            bce_weight=0.3,
            focal_weight=0.2,
            tversky_alpha=0.7,
            tversky_beta=0.3
        ),
        training=TrainingConfig(
            num_epochs=num_epochs,
            batch_size=batch_size,
            optimizer="adamw",
            learning_rate=1e-3,
            weight_decay=1e-4,
            scheduler="cosine",
            gradient_clip=1.0,
            use_amp=True,
            early_stopping_patience=5,
            device="auto",
            num_workers=0
        ),
        log_dir="./logs",
        vis_dir="./visualizations",
        vis_interval=2
    )
    
    return config


def run_single_experiment(
    arch: str,
    loss: str,
    data_root: str,
    **kwargs
) -> Dict[str, Any]:
    """Run a single experiment."""
    from trainer import run_experiment
    
    config = create_experiment_config(arch, loss, data_root, **kwargs)
    results = run_experiment(config)
    
    return {
        'architecture': arch,
        'loss_function': loss,
        'results': results,
        'config': config.get_experiment_id()
    }


def run_all_experiments(
    data_root: str,
    architectures: List[str] = None,
    losses: List[str] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Run grid of experiments."""
    
    if architectures is None:
        architectures = ["unet3d", "resunet3d", "attention_unet3d"]
    
    if losses is None:
        losses = ["dice", "bce", "focal", "tversky", "combined"]
    
    all_results = []
    
    print(f"\n{'='*60}")
    print("VESUVIUS EXPERIMENT LAB - GRID SEARCH")
    print(f"{'='*60}")
    print(f"Architectures: {architectures}")
    print(f"Loss Functions: {losses}")
    print(f"Total experiments: {len(architectures) * len(losses)}")
    print(f"{'='*60}\n")
    
    for arch in architectures:
        for loss in losses:
            print(f"\n{'='*60}")
            print(f"EXPERIMENT: {arch} + {loss}")
            print(f"{'='*60}")
            
            try:
                result = run_single_experiment(arch, loss, data_root, **kwargs)
                all_results.append(result)
                print(f"✓ Completed: {arch} + {loss}")
                print(f"  Best Dice: {result['results']['best_val_dice']:.4f}")
            except Exception as e:
                print(f"✗ Failed: {arch} + {loss}")
                print(f"  Error: {str(e)}")
                all_results.append({
                    'architecture': arch,
                    'loss_function': loss,
                    'error': str(e)
                })
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    # Sort by best dice
    successful = [r for r in all_results if 'results' in r]
    successful.sort(key=lambda x: x['results']['best_val_dice'], reverse=True)
    
    print(f"\nTop Results:")
    for i, r in enumerate(successful[:5], 1):
        print(f"  {i}. {r['architecture']} + {r['loss_function']}: "
              f"Dice = {r['results']['best_val_dice']:.4f}")
    
    # Save results
    results_path = Path("./experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'experiments': all_results
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Vesuvius Challenge Experiment Lab"
    )
    
    # Experiment selection
    parser.add_argument(
        '--arch', type=str, default='unet3d',
        choices=['unet3d', 'resunet3d', 'attention_unet3d'],
        help='Model architecture'
    )
    parser.add_argument(
        '--loss', type=str, default='dice',
        choices=['dice', 'bce', 'focal', 'tversky', 'combined'],
        help='Loss function'
    )
    parser.add_argument(
        '--run-all', action='store_true',
        help='Run all architecture + loss combinations'
    )
    
    # Data settings
    parser.add_argument(
        '--data-root', type=str,
        default='/Volumes/New_HDD2/KaggleCompetitions/Vesuvius_challenge/vesuvius-challenge-surface-detection',
        help='Path to data directory'
    )
    parser.add_argument(
        '--max-samples', type=int, default=100,
        help='Maximum number of training samples'
    )
    
    # Training settings
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--patch-size', type=int, default=64,
        help='Patch size (cubic)'
    )
    
    args = parser.parse_args()
    
    kwargs = {
        'max_samples': args.max_samples,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'patch_size': (args.patch_size, args.patch_size, args.patch_size)
    }
    
    if args.run_all:
        run_all_experiments(args.data_root, **kwargs)
    else:
        run_single_experiment(args.arch, args.loss, args.data_root, **kwargs)


if __name__ == "__main__":
    main()