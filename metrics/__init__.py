"""
Evaluation metrics for generated images
Includes FID, IS, LPIPS
"""

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
import lpips
from torchvision import models
from torch.nn.functional import adaptive_avg_pool2d


__all__ = ['FIDScore', 'InceptionScore', 'LPIPSScore', 'calculate_all_metrics']
