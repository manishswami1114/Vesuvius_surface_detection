# Vesuvius - Surface Detection ML Project

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Compatible-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vesuvius-surface-detection.streamlit.app)

This repository contains the training, inference pipelines, and a fully functional Web App for the **Vesuvius Surface Detection Challenge**. It has been refactored into a highly modular, clean, and production-ready Machine Learning project structure, providing robust 3D segmentation training, multi-GPU inference, optimized morphology-based post-processing, and a Streamlit-based graphical user interface.

## 🌐 Web App Deployment
You can interact with the pre-trained model directly in your browser!
👉 **[Launch Streamlit App](https://vesuvius-surface-detection.streamlit.app)**

## 📥 Get Test Data
To test the model or the Streamlit app, you will need 3D `.tif` volumes. You can download the official test and training data directly from the Vesuvius Challenge data portal:
👉 **[Download Data from ScrollPrize.org](https://scrollprize.org/data)**

## 🏆 Competition Write-Up

Check out the full write-up for my **94th Place Solution** on Kaggle, detailing the training methodology, architectural decisions, and how the Topology-Preserving 3D U-Net was utilized to tackle the competition: 
👉 [**88th Solution with Topology-Preserving 3D U-Net**](https://kaggle.com/competitions/vesuvius-challenge-surface-detection/writeups/94-th-solution-with-topology-preserving-3d-u-net)

## 💻 Hardware & System Requirements

- **Training Hardware**: The model in this repository was trained on a machine equipped with **Dual RTX 3090 GPUs (24GB VRAM each)** and **64 GB of System RAM**.
- **Compatibility**: All code is fully **CUDA-compatible**. The pipelines are designed to execute seamlessly on either a **CPU** or **CUDA-enabled GPUs**. Multi-GPU setups are automatically detected and utilized via PyTorch's DataParallel.

## 📂 Project Structure

- `src/` - Core source code module:
  - `app.py` - **Streamlit Web Application** for easy model inference and visualization.
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
- `checkpoints_v11/` - Pre-trained model weights stored via Git LFS.
- `IMPROVEMENTS_GUIDE.md` - Detailed documentation of mathematical optimizations and architectural improvements.
- `related_paper/` - Reference materials and academic papers.

## 🚀 Quickstart

### 1. Install Dependencies
Clone the repository and install the required packages:
```bash
git clone https://github.com/manishswami1114/Vesuvius_surface_detection.git
cd Vesuvius_surface_detection
pip install -r requirements.txt
```
*(Note: To download the large checkpoint files, make sure you have [Git LFS](https://git-lfs.com/) installed before cloning!)*

### 2. Run the Streamlit Web App Locally
To start the web interface on your local machine:
```bash
streamlit run src/app.py
```

### 3. Run Training
To start the training process from the terminal:
```bash
python -m src.train
```

### 4. Run CLI Inference
To generate predictions using the trained model via the command line:
```bash
python -m src.inference
```
