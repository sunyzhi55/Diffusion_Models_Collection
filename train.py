"""
Main training script for diffusion models
Supports single GPU and multi-GPU training
"""
import time
import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, SequentialLR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models import UNet, DiT, DiM
from diffusion import DDPM, DDIM
from datasets import DiffusionDataset, CustomImageDataset
from utils.trainer import DiffusionTrainer
from utils.helpers import set_seed, setup_distributed, count_parameters, load_config, resolve_image_size


def get_model(config):
    """Create model based on config"""
    model_type = config['model_type'].lower()
    model_params = config['model_params'].copy()
    # Ensure image size fields reflect normalized (H, W)
    if model_type == 'unet':
        model_params['image_size'] = config['image_size']
    elif model_type == 'dit':
        if 'img_size' in model_params:
            model_params['img_size'] = config['image_size']
    elif model_type == 'dim':
        if 'img_size' in model_params:
            model_params['img_size'] = config['image_size']
    
    # Add conditional info
    if config.get('conditional', False):
        model_params['num_classes'] = config.get('num_classes')
    else:
        model_params['num_classes'] = None
    
    if model_type == 'unet':
        model = UNet(**model_params)
    elif model_type == 'dit':
        model = DiT(**model_params)
    elif model_type == 'dim':
        model = DiM(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def get_diffusion(config, device):
    """Create DDPM diffusion process for training"""
    # Training always uses DDPM for loss computation
    diffusion = DDPM(
        num_timesteps=config['num_timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        beta_schedule=config['beta_schedule'],
        device=device
    )
    
    return diffusion


def get_dataset(config, train=True):
    """Create dataset based on config"""
    dataset_name = config['dataset'].lower()
    img_size = resolve_image_size(config['image_size'])
    
    if dataset_name == 'custom':
        # Custom dataset
        transform = CustomImageDataset.get_default_transform(
            img_size, 'rgb', train=train
        )
        dataset = CustomImageDataset(
            root=config['data_root'],
            transform=transform,
            conditional=config.get('conditional', False),
            label_file=config.get('label_file'),
            use_subdirs=config.get('use_subdirs', False)
        )
    else:
        # Torchvision dataset
        transform = DiffusionDataset.get_default_transform(
            img_size, dataset_name, train=train
        )
        dataset = DiffusionDataset(
            dataset_name=dataset_name,
            root=config['data_root'],
            train=train,
            transform=transform,
            download=True,
            conditional=config.get('conditional', False)
        )
    
    return dataset


def get_dataloader(config, dataset, rank=0, world_size=1, train=True):
    """Create dataloader"""
    if world_size > 1 and train:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = train
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=train
    )
    
    return dataloader


def get_optimizer(config, model):
    """Create optimizer"""
    optimizer_type = config.get('optimizer', 'adamw').lower()
    
    if optimizer_type == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    elif optimizer_type == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.0)
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def get_scheduler(config, optimizer):
    """Create learning rate scheduler"""
    if not config.get('use_scheduler', False):
        return None
    
    scheduler_type = config.get('scheduler_type', 'cosine').lower()
    
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
    elif scheduler_type == 'linear':
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config['epochs']
        )
    elif scheduler_type == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=config.get('step_size', 50),
            gamma=config.get('gamma', 0.5)
        )
    elif scheduler_type == 'warmup_cosine':
        warmup_epochs = max(0, config.get('warmup_epochs', 0))
        warmup_start = config.get('warmup_start_factor', 0.01)
        total_epochs = config['epochs']
        cosine_epochs = max(1, total_epochs - warmup_epochs)

        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs)
        ) if warmup_epochs > 0 else None

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=1e-6
        )

        if warmup is not None:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


def train_worker(rank, world_size, config, gpu_ids=None, local_rank=None):
    """Worker function for distributed training"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU training requires CUDA.")

    # Setup GPU
    if world_size > 1:
        if gpu_ids is not None:
            device_id = gpu_ids[local_rank if local_rank is not None else rank]
        else:
            device_id = local_rank if local_rank is not None else rank
    else:
        if isinstance(gpu_ids, list):
            device_id = gpu_ids[rank]
        else:
            device_id = gpu_ids

    if device_id is None:
        raise RuntimeError("No GPU id provided for training.")

    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    # Setup distributed
    if world_size > 1:
        setup_distributed(rank, world_size, backend='nccl', port=config.get('port', '12355'))
    
    # Set seed
    set_seed(config['seed'] + rank)
    
    # Create model
    print(f"[Rank {rank}] Creating model...")
    model = get_model(config)
    if rank == 0:
        print(f"Model parameters: {count_parameters(model):,}")
    
    # Create diffusion
    diffusion = get_diffusion(config, device)
    
    # Create dataset and dataloader
    print(f"[Rank {rank}] Loading dataset...")
    train_dataset = get_dataset(config, train=True)
    train_loader = get_dataloader(config, train_dataset, rank, world_size, train=True)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    
    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        rank=rank,
        world_size=world_size,
        resume_path=config.get('resume_path')
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    trainer.cleanup()


def main():
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config (YAML/JSON)
    config = load_config(Path(args.config))
    # Normalize image_size to (H, W)
    config['image_size'] = resolve_image_size(config['image_size'])

    # Use config file GPU settings
    gpu_id_config = config.get('gpu_ids', 0)
    config_world_size = len(gpu_id_config) if isinstance(gpu_id_config, list) else 1

    # Torchrun environment detection
    env_world_size = int(os.environ.get('WORLD_SIZE', '1'))
    using_torchrun = env_world_size > 1

    if using_torchrun:
        world_size = env_world_size
        rank = int(os.environ.get('RANK', '0'))
        local_rank = int(os.environ.get('LOCAL_RANK', rank))

        # Respect user-specified port when torchrun didn't set one
        os.environ.setdefault('MASTER_PORT', str(config.get('port', '12355')))

        num_gpus = torch.cuda.device_count()
        if local_rank >= num_gpus:
            raise ValueError(f"Local rank {local_rank} not available. Only {num_gpus} GPU(s) detected.")

        # Optional explicit device mapping when launching with torchrun
        gpu_ids = None
        if isinstance(gpu_id_config, list):
            if len(gpu_id_config) != world_size:
                raise ValueError(
                    f"gpu_ids length ({len(gpu_id_config)}) must equal WORLD_SIZE ({world_size}) when using torchrun."
                )
            for gid in gpu_id_config:
                if gid >= num_gpus:
                    raise ValueError(f"GPU {gid} not available. Only {num_gpus} GPU(s) detected.")
            gpu_ids = gpu_id_config

        config['world_size'] = world_size
        config['distributed'] = True

        print(f"Starting distributed training with torchrun (rank {rank}/{world_size}, local_rank {local_rank}, gpu_ids={gpu_ids})")
        train_worker(rank, world_size, config, gpu_ids=gpu_ids, local_rank=local_rank)
        return

    # Non-torchrun path (single GPU only)
    if config_world_size > 1:
        expected_nprocs = config_world_size
        raise RuntimeError(
            "Multi-GPU training now relies on torchrun. "
            f"Launch with: torchrun --nproc_per_node={expected_nprocs} train.py --config {args.config}"
        )

    gpu_ids = gpu_id_config

    # Validate GPU availability
    num_gpus = torch.cuda.device_count()
    target_gpu = gpu_ids[0] if isinstance(gpu_ids, list) else gpu_ids
    if target_gpu >= num_gpus:
        raise ValueError(f"GPU {target_gpu} not available. Only {num_gpus} GPU(s) detected.")
    
    # Update config with actual settings
    config['world_size'] = 1
    config['distributed'] = False
    
    print(f"Starting single GPU training on GPU {target_gpu}...")
    train_worker(0, 1, config, gpu_ids=gpu_ids, local_rank=target_gpu)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_seconds = end_time - start_time
    # 计算小时、分钟和秒
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    # 打印总训练时间
    print(f"Total training time: {hours}h {minutes}m {seconds}s")
