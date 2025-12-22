"""
Sampling/Inference script for trained diffusion models
"""
import time
import sys
import argparse
from pathlib import Path
import math
import torch
from torchvision.utils import save_image, make_grid

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models import UNet, DiT, DiM
from diffusion import DDPM, DDIM
from utils.helpers import set_seed, load_config, resolve_image_size, create_gif


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
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps for DDIM sampling')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for sampling')
    parser.add_argument('--output_dir', type=str, default='./samples', help='Output directory')
    parser.add_argument('--output_name', type=str, default='samples.png', help='Output filename')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model if available')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='Classifier-free guidance scale (0 = no CFG)')
    parser.add_argument('--labels', type=str, default=None, help='Comma-separated labels for conditional generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate denoising steps')
    parser.add_argument('--create_gif', action='store_true', help='Create GIF of the denoising process')
    parser.add_argument('--gif_fps', type=int, default=20, help='FPS for the GIF')
    parser.add_argument('--gif_final_seconds', type=float, default=2.0, help='Seconds to hold the final denoised frame in GIF')
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

    # num_inference_steps override
    if args.sampling_method.lower() == 'ddim' and args.num_inference_steps:
        config['num_inference_steps'] = args.num_inference_steps
    
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
    nrow = max(1, int(math.sqrt(args.num_samples)))
    num_rows = math.ceil(args.num_samples / nrow)
    
    if conditional:
        if args.labels:
            # Parse user-provided labels (e.g., "0,1,2" or "3")
            row_labels = [int(x.strip()) for x in args.labels.split(',')]
            
            # Validate labels are in valid range (optional but recommended)
            if num_classes is not None:
                for lbl in row_labels:
                    if not (0 <= lbl < num_classes):
                        raise ValueError(f"Label {lbl} is out of range [0, {num_classes})")
            
            # Repeat/truncate row labels to cover all rows, then expand per row
            if len(row_labels) < num_rows:
                row_labels = (row_labels * ((num_rows // len(row_labels)) + 1))[:num_rows]
            else:
                row_labels = row_labels[:num_rows]
            # Shift by +1 so 0 is reserved for unconditional
            labels = torch.tensor(row_labels, dtype=torch.long, device=device)
            labels = (labels + 1).repeat_interleave(nrow)[:args.num_samples]
        
        elif num_classes is not None:
            # Default: sample a class per row, then expand across the row
            row_labels = torch.randint(0, num_classes, (num_rows,), device=device, dtype=torch.long)
            labels = (row_labels + 1).repeat_interleave(nrow)[:args.num_samples]
        
        else:
            raise ValueError("Conditional generation requires either --labels or known num_classes.")
        print(f"Using conditional generation with labels: {labels.tolist()}")
    else:
        labels = None
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    all_intermediates = [] # List of lists of intermediates for each batch
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    return_all = args.save_intermediate or args.create_gif
    
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
                model, shape, batch_labels, cfg_scale=args.cfg_scale, return_all_timesteps=return_all
            )
        else:
            print(f"Sampling batch {i+1}/{num_batches}...")
            samples = diffusion.sample(model, shape, batch_labels, return_all_timesteps=return_all)
        
        if return_all:
            # samples: (T, B, C, H, W)
            all_intermediates.append(samples.cpu())
            all_samples.append(samples[-1].cpu())
        else:
            # samples: (B, C, H, W)
            all_samples.append(samples.cpu())
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    # Denormalize
    all_samples = (all_samples + 1) / 2
    all_samples = torch.clamp(all_samples, 0, 1)
    
    # Save
    output_path = output_dir / args.output_name
    print(f"Saving samples to {output_path}...")
    save_image(all_samples, str(output_path), nrow=nrow)
    
    # Process intermediates
    if return_all:
        print("Processing intermediate steps...")
        # Combine intermediates from all batches
        # all_intermediates is list of (T, B_i, C, H, W)
        # We want (T, Total_B, C, H, W)
        # Assuming T is same for all batches
        T = all_intermediates[0].shape[0]
        combined_intermediates = []
        for t in range(T):
            # Concatenate along batch dim for timestep t
            batch_t = torch.cat([batch[t] for batch in all_intermediates], dim=0)
            combined_intermediates.append(batch_t)
        
        # combined_intermediates is list of T tensors of shape (Total_B, C, H, W)
        
        # Denormalize
        combined_intermediates = [(img + 1) / 2 for img in combined_intermediates]
        combined_intermediates = [torch.clamp(img, 0, 1) for img in combined_intermediates]
        
        # Create GIF
        if args.create_gif:
            gif_path = output_dir / args.output_name.replace('.png', '.gif')
            print(f"Creating GIF at {gif_path}...")
            
            # Create grid for each timestep
            grid_frames = []
            for img in combined_intermediates:
                grid = make_grid(img, nrow=nrow, padding=2)
                grid_frames.append(grid)
            # Extend final frame to hold longer in the GIF
            if len(grid_frames) > 0 and args.gif_final_seconds and args.gif_final_seconds > 0:
                extra_frames = max(1, int(args.gif_fps * float(args.gif_final_seconds)))
                grid_frames.extend([grid_frames[-1]] * extra_frames)

            create_gif(grid_frames, str(gif_path), fps=args.gif_fps)
            
        # Save intermediate frames
        if args.save_intermediate:
            intermediate_dir = output_dir / 'intermediate'
            intermediate_dir.mkdir(exist_ok=True)
            print(f"Saving intermediate frames to {intermediate_dir}...")
            
            # Save every 10th frame or so to avoid too many files? 
            # Or just save all if user asked. Let's save all but maybe with step index.
            # If T is large (1000), this is too many.
            # If DDIM (50), it's fine.
            
            step_interval = 1
            if T > 100:
                step_interval = T // 50 # Limit to ~50 frames
            
            for t in range(0, T, step_interval):
                save_path = intermediate_dir / f'step_{t:04d}.png'
                save_image(combined_intermediates[t], str(save_path), nrow=nrow)
            
            # Always save last frame
            save_image(combined_intermediates[-1], str(intermediate_dir / f'step_{T-1:04d}.png'), nrow=nrow)
    
    print("Done!")


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
    print(f"Total sampling time: {hours}h {minutes}m {seconds}s")
