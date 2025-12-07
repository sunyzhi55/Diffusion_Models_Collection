# Diffusion Models Collection (扩散模型大集锦)

A comprehensive PyTorch implementation of various diffusion models for image generation, supporting multiple backbone architectures and sampling strategies.

[English](#english) | [中文](#chinese)

---

## <a id="english"></a>English

### Features

- **Multiple Backbone Architectures**
  - UNet: Classic architecture with residual blocks and attention
  - DiT (Diffusion Transformer): Transformer-based diffusion model
  - DiM (Diffusion Mamba): Efficient sequence modeling with Mamba (optional)

- **Sampling Algorithms**
  - DDPM (Denoising Diffusion Probabilistic Models)
  - DDIM (Denoising Diffusion Implicit Models) - Faster sampling

- **Dataset Support**
  - Built-in: CIFAR-10, CIFAR-100, MNIST, FashionMNIST, CelebA
  - Custom datasets with flexible data loading

- **Training Features**
  - Single GPU and Multi-GPU (DDP) support
  - Exponential Moving Average (EMA) for stable generation
  - Conditional and unconditional generation
  - Classifier-Free Guidance (CFG)
  - Automatic checkpointing (current + best model)
  - Periodic image sampling during training
  - SwanLab integration for experiment tracking

- **Evaluation Metrics**
  - FID (Fréchet Inception Distance)
  - IS (Inception Score)
  - LPIPS (Learned Perceptual Image Patch Similarity)

### Project Structure

```
DDPM_structure/
├── models/                  # Model architectures
│   ├── unet.py             # UNet implementation
│   ├── dit.py              # DiT implementation
│   └── dim.py              # DiM implementation
├── diffusion/              # Diffusion processes
│   ├── ddpm.py             # DDPM implementation
│   └── ddim.py             # DDIM implementation
├── datasets/               # Dataset handling
│   ├── base_dataset.py     # Torchvision datasets wrapper
│   └── custom_dataset.py   # Custom dataset loader
├── metrics/                # Evaluation metrics
│   ├── fid.py              # FID score
│   ├── inception_score.py  # Inception Score
│   └── lpips_score.py      # LPIPS score
├── utils/                  # Utility functions
│   ├── trainer.py          # Training loop
│   └── helpers.py          # Helper functions
├── configs/                # Configuration files
│   ├── cifar10_unet.yml    # CIFAR-10 + UNet config (YAML)
│   └── cifar10_dit.yml     # CIFAR-10 + DiT config (YAML)
├── train.py                # Training script
├── sample.py               # Sampling script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Mamba for DiM support
pip install mamba-ssm
```

### Quick Start

#### 1. Training

**Single GPU:**
```bash
python train.py --config configs/cifar10_unet.yml --gpus 1
```

**Multi-GPU (e.g., 4 GPUs):**
```bash
python train.py --config configs/cifar10_unet.yml --gpus 4
```

**Training with DiT:**
```bash
python train.py --config configs/cifar10_dit.yml --gpus 1
```

#### 2. Sampling

**Generate samples from trained model:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 64 \
    --batch_size 16 \
    --output_dir ./samples \
    --use_ema
```

**Conditional generation with CFG:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 64 \
    --cfg_scale 3.0 \
    --labels "0,1,2,3,4,5,6,7,8,9" \
    --use_ema
```

#### 3. Evaluation

**Compute metrics (FID, IS, LPIPS):**
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 5000 \
    --batch_size 32 \
    --use_ema \
    --output metrics_results.json
```

### Configuration

Edit configuration files in `configs/` to customize:

- Model architecture and parameters
- Dataset selection
- Training hyperparameters
- Diffusion parameters
- Sampling settings

Example configuration structure:
```python
config = {
    'model_type': 'unet',  # 'unet', 'dit', 'dim'
    'diffusion_type': 'ddpm',  # 'ddpm', 'ddim'
    'dataset': 'cifar10',
    'conditional': True,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 2e-4,
    # ... more options
}
```

### Custom Dataset

For custom datasets, organize your data in one of two ways:

**Option 1: Subdirectories (for conditional)**
```
data/
├── class_0/
│   ├── img1.jpg
│   └── img2.jpg
├── class_1/
│   ├── img3.jpg
│   └── img4.jpg
```

**Option 2: JSON labels**
```json
{
    "img1.jpg": 0,
    "img2.jpg": 0,
    "img3.jpg": 1
}
```

Then set in config:
```python
config = {
    'dataset': 'custom',
    'data_root': './data',
    'use_subdirs': True,  # or set 'label_file': 'labels.json'
}
```

---

## <a id="chinese"></a>中文

### 功能特性

- **多种主干网络**
  - UNet: 经典架构，包含残差块和注意力机制
  - DiT (Diffusion Transformer): 基于Transformer的扩散模型
  - DiM (Diffusion Mamba): 使用Mamba的高效序列建模（可选）

- **采样算法**
  - DDPM (去噪扩散概率模型)
  - DDIM (去噪扩散隐式模型) - 更快的采样速度

- **数据集支持**
  - 内置数据集: CIFAR-10, CIFAR-100, MNIST, FashionMNIST, CelebA
  - 支持自定义数据集，灵活的数据加载

- **训练特性**
  - 支持单卡和多卡分布式训练（DDP）
  - 指数移动平均（EMA）用于稳定生成
  - 条件生成和无条件生成
  - 无分类器引导（CFG）
  - 自动保存检查点（当前模型 + 最佳模型）
  - 训练过程中定期生成样本图片
  - SwanLab实验追踪集成

- **评估指标**
  - FID (Fréchet Inception Distance)
  - IS (Inception Score)
  - LPIPS (学习感知图像块相似度)

### 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 可选：安装Mamba以支持DiM
pip install mamba-ssm
```

### 快速开始

#### 1. 训练

**单卡训练:**
```bash
python train.py --config configs/cifar10_unet.yml --gpus 1
```

**多卡训练（例如4卡）:**
```bash
python train.py --config configs/cifar10_unet.yml --gpus 4
```

**使用DiT训练:**
```bash
python train.py --config configs/cifar10_dit.yml --gpus 1
```

#### 2. 采样生成

**从训练好的模型生成样本:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 64 \
    --batch_size 16 \
    --output_dir ./samples \
    --use_ema
```

**使用CFG进行条件生成:**
```bash
python sample.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 64 \
    --cfg_scale 3.0 \
    --labels "0,1,2,3,4,5,6,7,8,9" \
    --use_ema
```

#### 3. 评估

**计算评估指标（FID, IS, LPIPS）:**
```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 5000 \
    --batch_size 32 \
    --use_ema \
    --output metrics_results.json
```

### 配置说明

在 `configs/` 目录下编辑配置文件以自定义：

- 模型架构和参数
- 数据集选择
- 训练超参数
- 扩散过程参数
- 采样设置

配置文件示例：
```python
config = {
    'model_type': 'unet',  # 'unet', 'dit', 'dim'
    'diffusion_type': 'ddpm',  # 'ddpm', 'ddim'
    'dataset': 'cifar10',
    'conditional': True,
    'epochs': 200,
    'batch_size': 128,
    'learning_rate': 2e-4,
    # ... 更多选项
}
```

### 自定义数据集

自定义数据集有两种组织方式：

**方式1: 子目录结构（用于条件生成）**
```
data/
├── class_0/
│   ├── img1.jpg
│   └── img2.jpg
├── class_1/
│   ├── img3.jpg
│   └── img4.jpg
```

**方式2: JSON标签文件**
```json
{
    "img1.jpg": 0,
    "img2.jpg": 0,
    "img3.jpg": 1
}
```

然后在配置中设置：
```python
config = {
    'dataset': 'custom',
    'data_root': './data',
    'use_subdirs': True,  # 或设置 'label_file': 'labels.json'
}
```

### 训练监控

项目集成了SwanLab进行实验追踪，训练过程中会自动记录：
- 训练损失
- 学习率变化
- 生成的样本图片
- 训练时间等

在配置文件中设置：
```python
config = {
    'use_swanlab': True,
    'project_name': 'diffusion-models',
    'experiment_name': 'my-experiment'
}
```

### 检查点管理

训练过程中会保存三种检查点：
1. `current_model.pth` - 最新的模型
2. `best_model.pth` - 损失最低的最佳模型
3. `model_epoch_xxxx.pth` - 定期保存的检查点

### 注意事项

1. **显存要求**: DiT和DiM模型相比UNet需要更多显存，建议根据GPU显存调整batch_size
2. **训练时间**: 建议在训练开始一定轮数后（如20轮）再开始生成样本图片，避免早期生成质量过差
3. **Mamba安装**: 如果不需要DiM模型，可以不安装mamba-ssm，DiM会自动回退到注意力机制
4. **分布式训练**: 多卡训练时会自动使用DistributedDataParallel（DDP）

### 引用

如果您觉得这个项目有用，欢迎star和fork！

### 许可证

MIT License
