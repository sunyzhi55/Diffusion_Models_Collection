"""
FID (Fréchet Inception Distance) metric
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from torchvision import models
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network for FID calculation"""
    
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling features
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }
    
    def __init__(self, output_blocks=[DEFAULT_BLOCK_INDEX], resize_input=True, normalize_input=True):
        super().__init__()
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        
        inception = models.inception_v3(pretrained=True, transform_input=False)
        
        # Block 0: input to first max pool
        self.block0 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 1: first max pool to second max pool
        self.block1 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Block 2: second max pool to aux classifier
        self.block2 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        
        # Block 3: aux classifier to final avgpool
        self.block3 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            List of feature tensors
        """
        output = []
        
        # Resize to 299x299 for Inception
        if self.resize_input:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1]
        if self.normalize_input:
            x = 2 * x - 1
        
        # Forward through blocks
        x = self.block0(x)
        if 0 in self.output_blocks:
            output.append(adaptive_avg_pool2d(x, output_size=(1, 1)))
        
        x = self.block1(x)
        if 1 in self.output_blocks:
            output.append(adaptive_avg_pool2d(x, output_size=(1, 1)))
        
        x = self.block2(x)
        if 2 in self.output_blocks:
            output.append(adaptive_avg_pool2d(x, output_size=(1, 1)))
        
        x = self.block3(x)
        if 3 in self.output_blocks:
            output.append(x)
        
        return output


class FIDScore:
    """
    Calculate Fréchet Inception Distance (FID)
    
    FID measures the distance between distributions of real and generated images
    Lower is better
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.inception.eval()
    
    @torch.no_grad()
    def compute_statistics(self, images, batch_size=50):
        """
        Compute mean and covariance of features
        
        Args:
            images: Tensor of images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for processing
        
        Returns:
            Mean and covariance matrix
        """
        n_samples = len(images)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        features_list = []
        
        for i in tqdm(range(n_batches), desc='Computing features'):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = images[start:end].to(self.device)
            
            features = self.inception(batch)[0]
            features = features.squeeze(-1).squeeze(-1).cpu().numpy()
            features_list.append(features)
        
        features = np.concatenate(features_list, axis=0)
        
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Calculate Fréchet distance between two Gaussians
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small value for numerical stability
        
        Returns:
            FID score
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    
    def compute_fid(self, real_images, fake_images, batch_size=50):
        """
        Compute FID score
        
        Args:
            real_images: Real images (N, C, H, W) in range [0, 1]
            fake_images: Generated images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for processing
        
        Returns:
            FID score
        """
        print("Computing statistics for real images...")
        mu_real, sigma_real = self.compute_statistics(real_images, batch_size)
        
        print("Computing statistics for generated images...")
        mu_fake, sigma_fake = self.compute_statistics(fake_images, batch_size)
        
        print("Calculating FID score...")
        fid = self.calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid
