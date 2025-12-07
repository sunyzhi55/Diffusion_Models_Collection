"""
Evaluation script for computing metrics (FID, IS, LPIPS)
"""

import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models import UNet, DiT, DiM
from diffusion import DDPM, DDIM
from datasets import DiffusionDataset, CustomImageDataset
from metrics.lpips_score import calculate_all_metrics
from utils.helpers import set_seed, load_config


def get_model(config):
    """Create model based on config"""
    model_type = config['model_type'].lower()
    model_params = config['model_params'].copy()
    
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
    """Create diffusion process based on config"""
    diffusion_type = config['diffusion_type'].lower()
    
    common_params = {
        'num_timesteps': config['num_timesteps'],
        'beta_start': config['beta_start'],
        'beta_end': config['beta_end'],
        'beta_schedule': config['beta_schedule'],
        'device': device
    }
    
    if diffusion_type == 'ddpm':
        diffusion = DDPM(**common_params)
    elif diffusion_type == 'ddim':
        ddim_params = common_params.copy()
        ddim_params['num_inference_steps'] = config.get('num_inference_steps', 50)
        ddim_params['eta'] = config.get('ddim_eta', 0.0)
        diffusion = DDIM(**ddim_params)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")
    
    return diffusion


def get_dataset(config, train=False):
    """Create dataset based on config"""
    dataset_name = config['dataset'].lower()
    
    if dataset_name == 'custom':
        transform = CustomImageDataset.get_default_transform(
            config['image_size'], 'rgb'
        )
        dataset = CustomImageDataset(
            root=config['data_root'],
            transform=transform,
            conditional=False  # Don't need labels for evaluation
        )
    else:
        transform = DiffusionDataset.get_default_transform(
            config['image_size'], dataset_name
        )
        dataset = DiffusionDataset(
            dataset_name=dataset_name,
            root=config['data_root'],
            train=train,
            transform=transform,
            download=True,
            conditional=False
        )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate diffusion models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model')
    parser.add_argument('--output', type=str, default='./metrics_results.json', help='Output file for metrics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = checkpoint['config']
    
    # Create model
    print("Creating model...")
    model = get_model(config)
    
    # Load weights
    if args.use_ema and 'ema_model_state_dict' in checkpoint:
        print("Using EMA model")
        model.load_state_dict(checkpoint['ema_model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Create diffusion
    diffusion = get_diffusion(config, device)
    
    # Load real images
    print("Loading real images...")
    dataset = get_dataset(config, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    real_images = []
    for i, batch in enumerate(tqdm(dataloader, desc='Loading real images')):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
        else:
            imgs = batch
        
        # Denormalize from [-1, 1] to [0, 1]
        imgs = (imgs + 1) / 2
        real_images.append(imgs)
        
        if len(real_images) * args.batch_size >= args.num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:args.num_samples]
    print(f"Loaded {len(real_images)} real images")
    
    # Generate fake images
    print(f"Generating {args.num_samples} fake images...")
    fake_images = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        start = i * args.batch_size
        end = min(start + args.batch_size, args.num_samples)
        batch_size = end - start
        
        shape = (batch_size, config['model_params']['in_channels'],
                config['image_size'], config['image_size'])
        
        print(f"Generating batch {i+1}/{num_batches}...")
        samples = diffusion.sample(model, shape)
        
        # Denormalize
        samples = (samples + 1) / 2
        fake_images.append(samples.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)[:args.num_samples]
    print(f"Generated {len(fake_images)} fake images")
    
    # Compute metrics
    print("\n" + "="*50)
    print("Computing metrics...")
    print("="*50)
    
    metrics = calculate_all_metrics(real_images, fake_images, device=device)
    
    # Save results
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Save to file
    output_path = Path(args.output)
    with output_path.open('w', encoding='utf-8') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
