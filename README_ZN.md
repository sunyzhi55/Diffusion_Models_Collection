<div align="center">

# ✨ Diffusion Models Collection

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Models](https://img.shields.io/badge/Models-DDPM%20%7C%20DDIM-7F78D2?style=flat)](https://arxiv.org/abs/2006.11239)
[![Backbones](https://img.shields.io/badge/Backbones-UNet%20%7C%20DiT%20%7C%20DiM-6CC7F6?style=flat)](https://github.com/facebookresearch/DiT)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)

**一个全面、工程化、基于 PyTorch 的现代扩散模型实现库。**

[English](README_EN.md) | [中文](README_ZN.md)

</div>

---

## 📖 简介

本项目提供了一个模块化且易于扩展的去噪扩散概率模型 (DDPM) 和去噪扩散隐式模型 (DDIM) 实现。支持多种主流的主干网络架构，包括经典的 **UNet**、基于 Transformer 的 **DiT** (Diffusion Transformer) 以及最前沿的 **DiM** (Diffusion Mamba)。

项目专为研究和工程落地设计，提供了统一的训练、采样和评估接口，并支持多 GPU 分布式训练 (DDP) 和实验追踪。

## 🚀 功能特性

- **🧠 先进的主干网络 (Backbones)**
  - **UNet**: 经典的残差块 + 自注意力机制架构。
  - **DiT**: 可扩展的 Diffusion Transformer 架构。
  - **DiM**: 高效的 Diffusion Mamba (状态空间模型) 架构。

- **⚡ 采样策略**
  - **DDPM**: 标准随机采样，生成质量高。
  - **DDIM**: 确定性加速采样，推理速度快。
  - **CFG**: 无分类器引导 (Classifier-Free Guidance)，用于条件生成。

- **📦 全面的数据集支持**
  - 内置支持: CIFAR-10, CIFAR-100, MNIST, FashionMNIST, CelebA。
  - **自定义数据集**: 支持通过文件夹结构或 JSON 标签文件灵活加载私有数据。

- **🛠️ 工程化特性**
  - **分布式训练**: 无缝支持 DDP (Distributed Data Parallel) 多卡训练。
  - **EMA**: 指数移动平均 (Exponential Moving Average)，稳定模型权重。
  - **实验追踪**: 集成 [SwanLab](https://swanlab.cn) 实现实时训练监控。
  - **评估指标**: 内置 FID, Inception Score (IS), 和 LPIPS 计算工具。

---

## 📂 项目结构

```text
DDPM_structure/
├── configs/                # 配置文件
│   ├── cifar10_dim.py
│   ├── cifar10_dit.py
│   ├── cifar10_dit.yml
│   ├── cifar10_unet.py
│   └── cifar10_unet.yml
├── datasets/               # 数据集加载逻辑
│   ├── __init__.py
│   ├── base_dataset.py
│   └── custom_dataset.py
├── diffusion/              # 扩散过程核心 (DDPM/DDIM)
│   ├── __init__.py
│   ├── ddpm.py
│   └── ddim.py
├── metrics/                # 评估指标计算
│   ├── __init__.py
│   ├── fid.py
│   ├── inception_score.py
│   └── lpips_score.py
├── models/                 # 模型主干架构
│   ├── __init__.py
│   ├── unet.py
│   ├── dit.py
│   └── dim.py
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── trainer.py
│   └── helpers.py
├── train.py                # 训练主脚本
├── sample.py               # 推理/采样脚本
├── evaluate.py             # 评估脚本
├── requirements.txt        # 依赖列表
└── LICENSE                 # 许可证文件
```

---

## 🛠️ 安装指南

1. **克隆仓库**
   ```bash
   git clone https://github.com/sunyzhi55/Diffusion_Models_Collection.git
   cd Diffusion_Models_Collection
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **(可选) 安装 Mamba**
   如果你计划使用 DiM (Mamba) 模型：
   ```bash
   pip install mamba-ssm
   ```

---

## ⚡ 快速开始

### 1. 训练 (Training)

`train.py` 脚本同时支持单卡和多卡 (DDP) 训练。

**单 GPU 训练:**
```bash
# 在 CIFAR-10 上训练 UNet
python train.py --config configs/cifar10_unet.py --gpus 0

# 训练 DiT 模型
python train.py --config configs/cifar10_dit.py --gpus 0
```

**多 GPU 分布式训练 (DDP):**
```bash
# 使用 4 张 GPU 进行训练
python train.py --config configs/cifar10_unet.py --gpus 0,1,2,3
```

### 2. 采样 / 推理 (Sampling)

使用 `sample.py` 加载训练好的权重生成图像。

**标准 DDPM 采样:**
```bash
python sample.py \
    --checkpoint checkpoints/cifar10-unet-ddpm/best_model.pth \
    --sampling_method ddpm \
    --num_samples 64 \
    --batch_size 16 \
    --output_dir ./generated_images \
    --use_ema
```

**加速 DDIM 采样:**
```bash
python sample.py \
    --checkpoint checkpoints/cifar10-unet-ddpm/best_model.pth \
    --sampling_method ddim \
    --num_inference_steps 50 \
    --num_samples 64 \
    --use_ema
```

**带 CFG 的条件生成:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --sampling_method ddim \
    --cfg_scale 4.0 \
    --labels "0,1,2,3" \
    --use_ema
```

### 3. 评估 (Evaluation)

计算 FID, IS, 和 LPIPS 指标。

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 10000 \
    --batch_size 32 \
    --save_images_dir ./eval_generated_images \
    --use_ema \
    --output metrics_report.json
```

---

## ⚙️ 配置说明

配置文件采用 Python 格式 (例如 `configs/cifar10_unet.py`)，提供最大的灵活性。

```python
config = {
    # 项目元数据
    'project_name': 'diffusion-models',
    'experiment_name': 'cifar10-unet',

    # 模型架构
    'model_type': 'unet',   # 选项: 'unet', 'dit', 'dim'
    'model_params': {
        'image_size': (32, 32),
        'in_channels': 3,
        # ... 特定模型参数
    },

    # 数据集
    'dataset': 'cifar10',
    'conditional': True,
    'num_classes': 10,

    # 训练超参数
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 2e-4,
    
    # 扩散参数
    'num_timesteps': 1000,
    'beta_schedule': 'linear',
}
```

---

## 🗂️ 自定义数据集

你可以使用 `custom` 模式在自己的数据集上进行训练。

**方式 1: 文件夹结构 (按类别分文件夹)**
```text
data/
├── dog/
│   ├── img1.jpg
│   └── ...
├── cat/
│   ├── img1.jpg
│   └── ...
```

**方式 2: 扁平文件夹 + JSON 标签**
```json
// labels.json
{
  "img1.jpg": 0,
  "img2.jpg": 1
}
```

**配置设置:**
```python
config = {
    'dataset': 'custom',
    'data_root': './path/to/data',
    'use_subdirs': True,  # 如果使用 JSON 方式，设为 False
    'label_file': None,   # 如果不使用子目录方式，指定 JSON 路径
}
```

---

## 📈 实验追踪

本项目支持使用 [SwanLab](https://swanlab.cn) 进行训练指标的可视化监控。

在配置文件中启用:
```python
config = {
    'use_swanlab': True,
    'project_name': 'diffusion-models',
    # ...
}
```

---

## 📜 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

---
✅ 现在就克隆项目，开启你的扩散模型之旅吧！

---
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&color=9580FF&text=✨%20Enjoy%20Building%20Your%20Model!%20✨&fontColor=FFC0FA&fontSize=25&height=80"/>
</p>