"""
Inception Score (IS) metric
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from torch.nn.functional import softmax
from tqdm import tqdm


class InceptionScore:
    """
    Calculate Inception Score (IS)
    
    IS measures quality and diversity of generated images
    Higher is better
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        # Use InceptionV3 for classification
        self.inception = models.inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
    
    @torch.no_grad()
    def compute_inception_score(self, images, batch_size=32, splits=10):
        """
        Compute Inception Score
        
        Args:
            images: Generated images (N, C, H, W) in range [0, 1]
            batch_size: Batch size for processing
            splits: Number of splits for computing score
        
        Returns:
            Mean and std of IS across splits
        """
        n_samples = len(images)
        
        # Get predictions
        preds = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc='Computing predictions'):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch = images[start:end].to(self.device)
            
            # Resize to 299x299
            batch = nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Normalize to [-1, 1]
            batch = 2 * batch - 1
            
            # Get predictions
            pred = self.inception(batch)
            pred = softmax(pred, dim=1).cpu().numpy()
            preds.append(pred)
        
        preds = np.concatenate(preds, axis=0)
        
        # Compute IS for each split
        split_scores = []
        split_size = n_samples // splits
        
        for k in range(splits):
            part = preds[k * split_size: (k + 1) * split_size]
            
            # p(y|x)
            py_given_x = part
            
            # p(y) = E_x[p(y|x)]
            py = np.mean(py_given_x, axis=0)
            
            # KL divergence
            kl_div = py_given_x * (np.log(py_given_x + 1e-10) - np.log(py + 1e-10))
            kl_div = np.mean(np.sum(kl_div, axis=1))
            
            split_scores.append(np.exp(kl_div))
        
        return np.mean(split_scores), np.std(split_scores)
