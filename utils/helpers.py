"""
Utility functions
"""

import torch
import random
import numpy as np
import os
from pathlib import Path
import yaml


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_id=None):
    """Get device for training"""
    if device_id is not None:
        return torch.device(f'cuda:{device_id}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_config(config, save_path):
    """Save configuration to file (YAML preferred, JSON fallback)"""
    path = Path(save_path)
    if path.suffix in {'.yml', '.yaml'}:
        with path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    else:
        import json
        with path.open('w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)


def load_config(config_path):
    """Load configuration from YAML or JSON file"""
    path = Path(config_path)
    if path.suffix in {'.yml', '.yaml'}:
        with path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    import json
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_to_neg_one_to_one(img):
    """Normalize image from [0, 1] to [-1, 1]"""
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    """Unnormalize image from [-1, 1] to [0, 1]"""
    return (img + 1) * 0.5


def setup_distributed(rank, world_size, backend='nccl'):
    """Setup for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
