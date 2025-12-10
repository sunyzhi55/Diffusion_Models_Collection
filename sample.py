"""
Sampling/Inference script for trained diffusion models
"""

import sys
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models import UNet, DiT, DiM
from diffusion import DDPM, DDIM
from utils.helpers import set_seed, load_config, resolve_image_size


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


def get_diffusion(config, device, sampling_method='ddpm'):
    """Create diffusion process for sampling"""
    sampling_method = sampling_method.lower()
    
    common_params = {
        'num_timesteps': config['num_timesteps'],
        'beta_start': config['beta_start'],
        'beta_end': config['beta_end'],
        'beta_schedule': config['beta_schedule'],
        'device': device
    }
    
    if sampling_method == 'ddpm':
        diffusion = DDPM(**common_params)
    elif sampling_method == 'ddim':
        ddim_params = common_params.copy()
        ddim_params['num_inference_steps'] = config.get('num_inference_steps', 50)
        ddim_params['eta'] = config.get('ddim_eta', 0.0)
        diffusion = DDIM(**ddim_params)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}. Use 'ddpm' or 'ddim'")
    
    return diffusion


def main():
    parser = argparse.ArgumentParser(description='Sample from trained diffusion models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (if not in checkpoint)')
    parser.add_argument('--sampling_method', type=str, default='ddpm', choices=['ddpm', 'ddim'], 
                        help='Sampling method: ddpm (1000 steps) or ddim (50 steps, faster)')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for sampling')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Output directory')
    parser.add_argument('--output_name', type=str, default='samples.png', help='Output filename')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model if available')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale (0 = no CFG)')
    parser.add_argument('--labels', type=str, default=None, help='Comma-separated labels for conditional generation')
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

    # Normalize image_size to (H, W)
    config['image_size'] = resolve_image_size(config['image_size'])
    
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
    
    # Create diffusion for sampling
    print(f"Using sampling method: {args.sampling_method.upper()}")
    diffusion = get_diffusion(config, device, sampling_method=args.sampling_method)
    
    # Prepare labels for conditional generation
    conditional = config.get('conditional', False)
    num_classes = config.get('num_classes')
    
    if conditional and args.labels:
        # Parse labels
        labels = [int(x) for x in args.labels.split(',')]
        if len(labels) < args.num_samples:
            # Repeat labels
            labels = (labels * ((args.num_samples // len(labels)) + 1))[:args.num_samples]
        labels = torch.tensor(labels[:args.num_samples], device=device)
    elif conditional and num_classes:
        # Generate samples from each class
        labels = torch.arange(args.num_samples, device=device) % num_classes
    else:
        labels = None
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        start = i * args.batch_size
        end = min(start + args.batch_size, args.num_samples)
        batch_size = end - start
        
        h, w = config['image_size']
        shape = (batch_size, config['model_params']['in_channels'], h, w)
        
        batch_labels = labels[start:end] if labels is not None else None
        
        # Sample with or without CFG
        if args.cfg_scale > 0 and conditional:
            print(f"Sampling batch {i+1}/{num_batches} with CFG scale {args.cfg_scale}...")
            samples = diffusion.sample_with_cfg(
                model, shape, batch_labels, cfg_scale=args.cfg_scale
            )
        else:
            print(f"Sampling batch {i+1}/{num_batches}...")
            samples = diffusion.sample(model, shape, batch_labels)
        
        all_samples.append(samples.cpu())
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Denormalize
    all_samples = (all_samples + 1) / 2
    all_samples = torch.clamp(all_samples, 0, 1)
    
    # Save
    output_path = output_dir / args.output_name
    print(f"Saving samples to {output_path}...")
    nrow = int(args.num_samples ** 0.5)
    save_image(all_samples, str(output_path), nrow=nrow)
    
    print("Done!")


if __name__ == '__main__':
    main()
