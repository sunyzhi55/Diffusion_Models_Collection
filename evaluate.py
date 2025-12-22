"""
Evaluation script for computing metrics (FID, IS, LPIPS)
"""
import time
import sys
import argparse
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models import UNet, DiT, DiM
from diffusion import DDPM, DDIM
from datasets import DiffusionDataset, CustomImageDataset
from metrics.lpips_score import calculate_all_metrics
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


def get_diffusion(config, device):
    """Create DDPM diffusion process for evaluation"""
    # Evaluation always uses DDPM for consistency with training
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate diffusion models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA model')
    parser.add_argument('--output', type=str, default='./metrics_results.json', help='Output file for metrics')
    parser.add_argument('--save_images_dir', type=str, default='./eval', help='Directory to save PNG images (real/generate subfolders)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--cfg_scale', type=float, default=0.0, help='CFG guidance scale (0 = no CFG)')
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
    
    # Create diffusion
    diffusion = get_diffusion(config, device)
    
    # Load real images
    print("Loading real images...")
    dataset = get_dataset(config, train=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    real_images = []
    real_labels = []  # collect labels to match class distribution during conditional sampling
    for i, batch in enumerate(tqdm(dataloader, desc='Loading real images')):
        if isinstance(batch, (list, tuple)):
            imgs = batch[0]
            if len(batch) > 1 and torch.is_tensor(batch[1]):
                real_labels.append(batch[1])
        else:
            imgs = batch
        
        # Denormalize from [-1, 1] to [0, 1]
        imgs = (imgs + 1) / 2
        real_images.append(imgs)
        
        if len(real_images) * args.batch_size >= args.num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:args.num_samples]
    if real_labels:
        real_labels = torch.cat(real_labels, dim=0)[:args.num_samples]
    else:
        real_labels = None
    print(f"Loaded {len(real_images)} real images")
    
    # Generate fake images
    print(f"Generating {args.num_samples} fake images...")
    fake_images = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    # Prepare label schedule for conditional models (shift by +1, 0 is null)
    conditional = config.get('conditional', False)
    num_classes = config.get('num_classes')
    if conditional:
        if real_labels is None or num_classes is None:
            raise ValueError("Conditional evaluation requires labels from the real dataset and known num_classes.")
        # 进行类别分布匹配采样
        # hist = torch.bincount(real_labels.to(device), minlength=num_classes).float()
        # if hist.sum() == 0:
        #     probs = torch.full((num_classes,), 1.0 / num_classes, device=device)
        # else:
        #     probs = hist / hist.sum()
        # sampled = torch.multinomial(probs, args.num_samples, replacement=True)
        # labels_all = sampled + 1  # shift to avoid 0

        # 直接使用真实标签进行评估
        labels_all = real_labels.to(device) + 1  # shift to avoid 0
    else:
        labels_all = None

    for i in range(num_batches):
        start = i * args.batch_size
        end = min(start + args.batch_size, args.num_samples)
        batch_size = end - start
        
        h, w = config['image_size']
        shape = (batch_size, config['model_params']['in_channels'], h, w)
        batch_labels = labels_all[start:end] if labels_all is not None else None
        
        print(f"Generating batch {i+1}/{num_batches}...")
        if args.cfg_scale > 0 and conditional:
            print(f"Sampling with CFG scale {args.cfg_scale}, labels: {batch_labels}")
            samples = diffusion.sample_with_cfg(model, shape, batch_labels, cfg_scale=args.cfg_scale)
        else:
            samples = diffusion.sample(model, shape, batch_labels)
        
        # Denormalize
        samples = (samples + 1) / 2
        fake_images.append(samples.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)[:args.num_samples]
    print(f"Generated {len(fake_images)} fake images")

    # Optionally save all images as PNGs in a single root folder
    if args.save_images_dir:
        save_root = Path(args.save_images_dir)
        real_dir = save_root / 'real'
        gen_dir = save_root / 'generate'
        real_dir.mkdir(parents=True, exist_ok=True)
        gen_dir.mkdir(parents=True, exist_ok=True)

        num_digits = len(str(max(len(real_images), len(fake_images), 1)))

        for idx, img in enumerate(tqdm(real_images, desc='Saving real images')):
            save_image(img, real_dir / f"real_{idx + 1:0{num_digits}d}.png")

        for idx, img in enumerate(tqdm(fake_images, desc='Saving generated images')):
            save_image(img, gen_dir / f"generate_{idx + 1:0{num_digits}d}.png")

        # Save all images as grid PNGs in batches (default 64 per grid)
        def _save_grids(tensor_imgs, prefix, out_dir):
            grid_size = 64
            total = len(tensor_imgs)
            if total == 0:
                return
            # ensure tensor is on CPU
            imgs = tensor_imgs.cpu()
            num_digits_grid = len(str((total + grid_size - 1) // grid_size))
            for i in range(0, total, grid_size):
                chunk = imgs[i:i + grid_size]
                # compute nrow for a near-square grid (prefer 8 for 64)
                n = len(chunk)
                nrow = min(8, max(1, int(n ** 0.5)))
                grid_idx = i // grid_size + 1
                out_name = f"{prefix}_grid_{grid_idx:0{num_digits_grid}d}.png"
                save_image(chunk, out_dir / out_name, nrow=nrow)

        # Save real images grids
        _save_grids(real_images, 'real', save_root)
        # Save generated images grids
        _save_grids(fake_images, 'generate', save_root)

        print(f"Saved real images to {real_dir} and generated images to {gen_dir}")
        print(f"Also saved image grids in {save_root}")
    
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
    
    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    metrics_serializable = {k: _to_serializable(v) for k, v in metrics.items()}

    with output_path.open('w', encoding='utf-8') as f:
        import json
        json.dump(metrics_serializable, f, indent=4)
    
    print(f"\nResults saved to {args.output}")


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

"""
nohup python evaluate.py --checkpoint checkpoints/best_model.pth --num_samples 10000 --batch_size 512 --use_ema --output metrics_report.json --cfg_scale 1.4 > result_eval_cfg.out &
"""
