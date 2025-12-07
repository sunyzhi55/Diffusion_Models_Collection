"""
Datasets package
Supports torchvision datasets and custom datasets
"""

from .base_dataset import DiffusionDataset
from .custom_dataset import CustomImageDataset

__all__ = ['DiffusionDataset', 'CustomImageDataset']
