# Vesuvius - Surface Detection ML Project

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-green)

This repository contains the training and inference pipelines for the **Vesuvius Surface Detection Challenge**. It has been refactored into a highly modular, clean, and production-ready Machine Learning project structure, providing robust 3D segmentation training, multi-GPU inference, and optimized morphology-based post-processing.

## 🏆 Competition Write-Up

Check out the full write-up for my **94th Place Solution** on Kaggle, detailing the training methodology, architectural decisions, and how the Topology-Preserving 3D U-Net was utilized to tackle the competition: 
👉 [**88th Solution with Topology-Preserving 3D U-Net**](https://kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/94-th-solution-with-topology-preserving-3d-u-net)

## 💻 Hardware & System Requirements

- **Training Hardware**: The model in this repository was trained on a machine equipped with **Dual RTX 3090 GPUs (24GB VRAM each)** and **64 GB of System RAM**.
- **Compatibility**: All code is fully **CUDA-compatible**. The pipelines are designed to execute seamlessly on either a **CPU** or **CUDA-enabled GPUs**. Multi-GPU setups are automatically detected and utilized via PyTorch's DataParallel.

## 📂 Project Structure

- `src/` - Core source code module:
  - `config.py` - Centralized configuration settings and hyperparameters.
  - `models.py` - Topology-Preserving 3D U-Net architecture.
  - `dataset.py` - Memory-efficient 3D volume data loading and caching logic.
  - `augmentations.py` - High-performance GPU-accelerated spatial augmentations.
  - `losses.py` - Custom loss functions (Dice, clDice, BCE).
  - `utils.py` - Helper functions, logging, and checkpointing.
  - `train.py` - Main training pipeline with cross-validation support.
  - `inference.py` - Multi-GPU sliding-window inference and submission generation.
  - `postprocessing.py` - Optimized 2D/3D morphology-based mask post-processing.
- `notebooks/` - The original source Jupyter notebooks (`v11_training.ipynb`, `V11_inference_final.ipynb`).
- `IMPROVEMENTS_GUIDE.md` - Detailed documentation of mathematical optimizations and architectural improvements.
- `related_paper/` - Reference materials and academic papers.

## 🚀 Quickstart

### 1. Install Dependencies
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run Training
To start the training process:
```bash
python -m src.train
```

### 3. Run Inference
To generate predictions using the trained model:
```bash
python -m src.inference
```
