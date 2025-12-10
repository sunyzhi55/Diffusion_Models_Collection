"""
LPIPS (Learned Perceptual Image Patch Similarity) metric
"""

import torch
import lpips
from tqdm import tqdm
import numpy as np


class LPIPSScore:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity)
    
    LPIPS measures perceptual similarity between images
    Lower is better (more similar)
    """
    
    def __init__(self, net='alex', device='cuda'):
        """
        Args:
            net: Network to use ('alex', 'vgg', 'squeeze')
            device: Device to run on
        """
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()
    
    @torch.no_grad()
    def compute_lpips(self, images1, images2, batch_size=32):
        """
        Compute LPIPS distance between two sets of images
        
        Args:
            images1: First set of images (N, C, H, W) in range [0, 1]
            images2: Second set of images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for processing
        
        Returns:
            Mean LPIPS distance
        """
        assert len(images1) == len(images2), "Number of images must match"
        
        n_samples = len(images1)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        distances = []
        
        for i in tqdm(range(n_batches), desc='Computing LPIPS'):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            batch1 = images1[start:end].to(self.device)
            batch2 = images2[start:end].to(self.device)
            
            # LPIPS expects images in range [-1, 1]
            batch1 = 2 * batch1 - 1
            batch2 = 2 * batch2 - 1
            
            dist = self.loss_fn(batch1, batch2)
            distances.append(dist.cpu().numpy())
        
        distances = np.concatenate(distances, axis=0)
        
        return np.mean(distances)
    
    @torch.no_grad()
    def compute_lpips_diversity(self, images, num_pairs=1000, batch_size=32):
        """
        Compute LPIPS diversity within a set of images
        Measures average distance between random pairs
        
        Args:
            images: Set of images (N, C, H, W) in range [0, 1]
            num_pairs: Number of random pairs to sample
            batch_size: Batch size for processing
        
        Returns:
            Mean LPIPS distance between random pairs
        """
        n_samples = len(images)
        
        # Sample random pairs
        idx1 = torch.randint(0, n_samples, (num_pairs,))
        idx2 = torch.randint(0, n_samples, (num_pairs,))
        
        # Ensure pairs are different
        while (idx1 == idx2).any():
            mask = idx1 == idx2
            idx2[mask] = torch.randint(0, n_samples, (mask.sum(),))
        
        images1 = images[idx1]
        images2 = images[idx2]
        
        return self.compute_lpips(images1, images2, batch_size)


def calculate_all_metrics(real_images, fake_images, device='cuda'):
    """
    Calculate all metrics (FID, IS, LPIPS)
    
    Args:
        real_images: Real images (N, C, H, W) in range [0, 1]
        fake_images: Generated images (N, C, H, W) in range [0, 1]
        device: Device to run on
    
    Returns:
        Dictionary with all metrics
    """
    from .fid import FIDScore
    from .inception_score import InceptionScore
    
    metrics = {}
    
    # FID
    print("\n=== Computing FID ===")
    fid_calculator = FIDScore(device=device)
    fid = fid_calculator.compute_fid(real_images, fake_images)
    metrics['FID'] = float(fid)
    print(f"FID: {fid:.4f}")
    
    # IS
    print("\n=== Computing IS ===")
    is_calculator = InceptionScore(device=device)
    is_mean, is_std = is_calculator.compute_inception_score(fake_images)
    metrics['IS_mean'] = float(is_mean)
    metrics['IS_std'] = float(is_std)
    print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
    
    # LPIPS
    print("\n=== Computing LPIPS ===")
    lpips_calculator = LPIPSScore(device=device)
    
    # Diversity
    lpips_div = lpips_calculator.compute_lpips_diversity(fake_images)
    metrics['LPIPS_diversity'] = float(lpips_div)
    print(f"LPIPS Diversity: {lpips_div:.4f}")
    
    return metrics
