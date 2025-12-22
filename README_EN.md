<div align="center">

# ✨ Diffusion Models Collection

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-DDPM%20%7C%20DDIM-7F78D2?style=flat)](https://arxiv.org/abs/2006.11239)
[![Backbones](https://img.shields.io/badge/Backbones-UNet%20%7C%20DiT%20%7C%20DiM-6CC7F6?style=flat)](https://github.com/facebookresearch/DiT)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)

**A comprehensive, engineering-oriented PyTorch implementation of modern Diffusion Models.**

[English](README_EN.md) | [Chinese](README.md)

</div>

---

## 📖 1 Introduction

This repository provides a modular and extensible implementation of Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM). It supports multiple backbone architectures including the classic **UNet**, the Transformer-based **DiT** (Diffusion Transformer), and the state-of-the-art **DiM** (Diffusion Mamba).

Designed for research and engineering, it features unified interfaces for training, sampling, and evaluation, along with support for distributed training and experiment tracking.

## 🚀 2 Features

- **🧠 Advanced Backbones**
  - **UNet**: Classic residual blocks with self-attention mechanisms.
  - **DiT**: Scalable Diffusion Transformer architecture.
  - **DiM**: Efficient Diffusion Mamba (State Space Models) backbone.

- **⚡ Sampling Strategies**
  - **DDPM**: Standard stochastic sampling for high-quality generation.
  - **DDIM**: Deterministic accelerated sampling.
  - **CFG**: Classifier-Free Guidance for conditional generation.

- **📦 Comprehensive Dataset Support**
  - Built-in support: CIFAR-10, CIFAR-100, MNIST, FashionMNIST, CelebA.
  - **Custom Datasets**: Flexible loading via folder structure or JSON label files.

- **🛠️ Engineering Features**
  - **Distributed Training**: Seamless DDP (Distributed Data Parallel) support.
  - **EMA**: Exponential Moving Average for stable model weights.
  - **Experiment Tracking**: Integrated with [SwanLab](https://swanlab.cn) for real-time monitoring.
  - **Metrics**: Built-in calculation for FID, Inception Score (IS), and LPIPS.

---

## 📂 3 Project Structure

```text
DDPM_structure/
├── configs/                # Configuration files
│   ├── cifar10_dim.py
│   ├── cifar10_dit.py
│   ├── cifar10_dit.yml
│   ├── cifar10_unet.py
│   └── cifar10_unet.yml
├── datasets/               # Dataset loading logic
│   ├── __init__.py
│   ├── base_dataset.py
│   └── custom_dataset.py
├── diffusion/              # Diffusion processes (DDPM/DDIM)
│   ├── __init__.py
│   ├── ddpm.py
│   └── ddim.py
├── metrics/                # Evaluation metrics
│   ├── __init__.py
│   ├── fid.py
│   ├── inception_score.py
│   └── lpips_score.py
├── models/                 # Backbone architectures
│   ├── __init__.py
│   ├── unet.py
│   ├── dit.py
│   └── dim.py
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── trainer.py
│   └── helpers.py
├── train.py                # Main training script
├── sample.py               # Inference/Sampling script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── LICENSE                 # License file
```

---

## 🛠️ 4 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sunyzhi55/Diffusion_Models_Collection.git
   cd Diffusion_Models_Collection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Install Mamba for DiM**
   If you intend to use the Mamba backbone:
   ```bash
   pip install mamba-ssm
   ```

---

## ⚡ 5 Quick Start

### 5.1 Training

The training script `train.py` supports both single-GPU and multi-GPU (DDP) configurations.

**On CFG Training (Classifier-Free Guidance)**
- When `conditional=True`, dataset labels are internally shifted by +1; index 0 is reserved as the unconditional (null) label. Ensure `num_classes` matches your data.
- `cfg_dropout_prob` (config, default 0.2) randomly sets a portion of labels to 0 to train the unconditional branch; set it to 0 to disable.
- With this setup, DDP does **not** require `find_unused_parameters=True` because the null label still flows through the same forward path.

1、**Single GPU:**

Modify the ` gpu_ids' in the configuration file` to specify the GPU ID.

```python
config = {
    # ...
    'gpu_ids': [0],  # Specify single GPU ID
    # ...
}
```

Then run the training command:

```bash
# Train UNet on CIFAR-10
python train.py --config configs/cifar10_unet.py

# Train DiT
python train.py --config configs/cifar10_dit.py
```

2、**Multi-GPU (DDP):**

Modify the ` gpu_ids' in the configuration file` to specify multiple GPU IDs.

```python
config = {
    # ...
    'gpu_ids': [0, 1, 2, 3],  # Specify multiple GPU IDs
    # ...
}
```

Then run the training command:

```bash
# Train on 4 GPUs
python train.py --config configs/cifar10_unet.py

# nohup example (run in background)
nohup setsid torchrun --nproc_per_node=2 --master_port=15355 train.py --config configs/cifar10_unet.py > result.out &
```

3、**Resume Training:**

Set `resume_path` in the configuration file to the checkpoint path:

```python
# configs/cifar10_unet.py
config = {
    # ...
    'resume_path': 'checkpoints/cifar10-unet-ddpm/model_epoch_0050.pth',
    # ...
}
```

> **If the epoch count in the current configuration file is less than or equal to the number of epochs already trained, training will automatically extend to the new total number of epochs.**
>
> **If the epoch count in the current configuration file is greater than the number of epochs already trained, training will continue according to the epoch count in the configuration file.**
Then run the training command as usual.

### 5.2 Sampling / Inference

Generate images using trained checkpoints with `sample.py`.

**On CFG Sampling**
- Provide labels in the real class index range `[0, num_classes-1]`; the script automatically shifts them by +1 so 0 remains the unconditional token and is never fed to the model.
- `cfg_scale` controls guidance strength; common values are 1.5–3.0. Too large can cause artifacts or collapse.
- Prefer `--use_ema` at sampling time for smoother quality.

**Standard DDPM Sampling:**
```bash
python sample.py \
    --checkpoint checkpoints/cifar10-unet-ddpm/best_model.pth \
    --sampling_method ddpm \
    --num_samples 64 \
    --batch_size 16 \
    --output_dir ./generated_images \
    --use_ema
```

**Accelerated DDIM Sampling:**
```bash
python sample.py \
    --checkpoint checkpoints/cifar10-unet-ddpm/best_model.pth \
    --sampling_method ddim \
    --num_inference_steps 50 \
    --num_samples 64 \
    --use_ema
```

**Conditional Sampling with CFG:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --sampling_method ddim \
    --cfg_scale 4.0 \
    --labels "0,1,2,3" \
    --use_ema
```
> Note: `--labels` accepts real class indices starting from 0; the script shifts them internally, so 0 is never passed to the model. If `--labels` is omitted, classes are sampled per row at random.

**Generate GIF and Save Intermediate Steps:**
```bash
# Generate GIF animation
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --create_gif \
    --output_dir ./results

# Generate GIF and save intermediate frames
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --create_gif \
    --save_intermediate \
    --output_dir ./results

# Accelerated sampling with DDIM and generate GIF (Recommended)
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --sampling_method ddim \
    --num_inference_steps 50 \
    --create_gif
```

### 5.3 Evaluation

Calculate FID, IS, and LPIPS metrics.

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 10000 \
    --batch_size 32 \
    --save_images_dir ./eval_generated_images \
    --use_ema \
    --output metrics_report.json
```

> [!note]
>
> Notes: If conditional diffusion is used during training, evaluation will automatically use conditional diffusion and reuse the real data labels (shifted by +1, with 0 as the null label) to match the true class distribution; ensure the evaluation dataset can return labels and that `num_classes` matches the data; use `--cfg_scale` (recommended 1.5–3.0) to enable/adjust CFG sampling. [The `--cfg_scale` parameter must be used.]

---

## ⚙️ 6 Configuration

Configurations are defined in Python files (e.g., `configs/cifar10_unet.py`) for maximum flexibility.

```python
config = {
    # Project Metadata
    'project_name': 'diffusion-models',
    'experiment_name': 'cifar10-unet',
    'resume_path': None,    # Path to checkpoint for resuming training

    # Model Architecture
    'model_type': 'unet',   # Options: 'unet', 'dit', 'dim'
    'model_params': {
        'image_size': (32, 32),
        'in_channels': 3,
        # ... specific model params
    },

    # Dataset
    'dataset': 'cifar10',
    'conditional': True,
    'num_classes': 10,

    # Training Hyperparameters
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 2e-4,
    
    # Diffusion Parameters
    'num_timesteps': 1000,
    'beta_schedule': 'linear',
}
```

---

## 🗂️ 7  Custom Datasets

You can train on your own data using the `custom` dataset mode.

**Option 1: Folder Structure (Class-conditional)**
```text
data/
├── dog/
│   ├── img1.jpg
│   └── ...
├── cat/
│   ├── img1.jpg
│   └── ...
```

**Option 2: Flat Folder + JSON Labels**
labels.json example content:
```json
{
  "img1.jpg": 0,
  "img2.jpg": 1
}
```

**Config Setup:**
```python
config = {
    'dataset': 'custom',
    'data_root': './path/to/data',
    'use_subdirs': True,  # Set False if using JSON
    'label_file': None,   # Path to JSON if not using subdirs
}
```

---

## 📈 8 Experiment Tracking

This project supports [SwanLab](https://swanlab.cn) for visualizing training metrics.

Enable it in your config:
```python
config = {
    'use_swanlab': True,
    'project_name': 'diffusion-models',
    # ...
}
```

---

## 📊 9 Experiment Logs

- CIFAR-10 conditional training/sampling/evaluation summary: see [docs/cifar10_runs.md](docs/cifar10_runs.md).

---

## 📜 10 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


✅ Get started now by cloning the project and embark on your diffusion model journey!

---
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=9580FF&text=✨%20Enjoy%20Building%20Your%20Model!%20✨&fontColor=FFC0FA&fontSize=25&height=80"/>  
</p>
