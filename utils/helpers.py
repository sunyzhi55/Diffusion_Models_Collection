"""
Utility functions
"""

import torch
import random
import numpy as np
import os
from pathlib import Path


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_image_size(image_size):
    """Normalize image_size to a (height, width) tuple.

    Accepts int (square), list/tuple of length 2. Raises ValueError otherwise.
    """
    if isinstance(image_size, int):
        return (image_size, image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        h, w = image_size
        if not (isinstance(h, int) and isinstance(w, int)):
            raise ValueError("image_size values must be integers")
        return (h, w)
    raise ValueError("image_size must be int or a pair (H, W)")


def count_parameters(model):
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_id=None):
    """Get device for training"""
    if device_id is not None:
        return torch.device(f'cuda:{device_id}')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_config(config, save_path):
    """Save configuration to JSON file"""
    import json
    path = Path(save_path)
    with path.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)


def load_config(config_path):
    """Load configuration from Python file"""
    import sys
    import importlib.util
    
    path = Path(config_path)
    
    # Load Python module dynamically
    spec = importlib.util.spec_from_file_location("config", path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config_module
    spec.loader.exec_module(config_module)
    
    return config_module.config


def normalize_to_neg_one_to_one(img):
    """Normalize image from [0, 1] to [-1, 1]"""
    return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    """Unnormalize image from [-1, 1] to [0, 1]"""
    return (img + 1) * 0.5


def setup_distributed(rank, world_size, backend='nccl', port='12355'):
    """Setup for distributed training"""
    # Respect existing env (e.g., torchrun) while keeping a sane default
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', str(port))
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    # Note: Do not call torch.cuda.set_device() here
    # It should be called before this function in train_worker with the correct GPU ID


def create_gif(images_list, save_path, fps=20):
    """
    Create GIF from a list of images
    
    Args:
        images_list: List of images (tensor or numpy array), each shape (C, H, W) or (H, W, C)
                     Values should be in [0, 1] or [0, 255]
        save_path: Path to save the GIF
        fps: Frames per second
    """
    from PIL import Image
    import numpy as np
    
    frames = []
    for img in images_list:
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Handle (C, H, W) format
        if img.ndim == 3 and (img.shape[0] == 1 or img.shape[0] == 3):
            img = np.transpose(img, (1, 2, 0))
        
        # Normalize to [0, 255]
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Handle grayscale
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
            
        frames.append(Image.fromarray(img))
        
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/fps,
        loop=0
    )
