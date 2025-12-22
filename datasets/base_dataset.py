"""
Base dataset wrapper for diffusion models
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import Optional, Callable


class DiffusionDataset(Dataset):
    """
    Wrapper dataset for diffusion models
    Supports common torchvision datasets
    
    Supported datasets:
        - CIFAR10
        - CIFAR100
        - MNIST
        - FashionMNIST
        - CelebA
        - ImageNet (requires manual download)
    """
    
    SUPPORTED_DATASETS = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
        'mnist': datasets.MNIST,
        'fashionmnist': datasets.FashionMNIST,
        'celeba': datasets.CelebA,
    }
    
    def __init__(
        self,
        dataset_name: str,
        root: str = './data',
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True,
        conditional: bool = False
    ):
        """
        Args:
            dataset_name: Name of the dataset (e.g., 'cifar10', 'mnist')
            root: Root directory to store dataset
            train: Whether to use training set
            transform: Transform to apply to images
            download: Whether to download dataset if not present
            conditional: Whether to return labels for conditional generation
        """
        super().__init__()
        
        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                           f"Supported datasets: {list(self.SUPPORTED_DATASETS.keys())}")
        
        dataset_class = self.SUPPORTED_DATASETS[dataset_name]
        
        # Special handling for CelebA
        if dataset_name == 'celeba':
            split = 'train' if train else 'test'
            self.dataset = dataset_class(
                root=root,
                split=split,
                transform=transform,
                download=download
            )
        else:
            self.dataset = dataset_class(
                root=root,
                train=train,
                transform=transform,
                download=download
            )
        
        self.conditional = conditional
        self.dataset_name = dataset_name
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.dataset_name == 'celeba':
            img, _ = self.dataset[idx]
            # CelebA doesn't have class labels, return dummy label
            if self.conditional:
                return img, torch.tensor(0)
            return img
        else:
            img, label = self.dataset[idx]
            if self.conditional:
                return img, label
            return img
    
    @staticmethod
    def get_default_transform(image_size=32, dataset_name='cifar10', train=True):
        """Get default transform for a dataset.

        - train=True  : include simple augmentation (flip for RGB)
        - train=False : deterministic preprocessing only
        """
        dataset_name = dataset_name.lower()
        # torchvision Resize/CenterCrop accept int or (h, w)
        
        if dataset_name in ['mnist', 'fashionmnist']:
            # Grayscale datasets
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            if train:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
            else:
                return transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    
    @staticmethod
    def get_num_classes(dataset_name):
        """Get number of classes for a dataset"""
        dataset_name = dataset_name.lower()
        num_classes_map = {
            'cifar10': 10,
            'cifar100': 100,
            'mnist': 10,
            'fashionmnist': 10,
            'celeba': 0,  # No class labels
        }
        return num_classes_map.get(dataset_name, 0)
    
    @staticmethod
    def get_image_channels(dataset_name):
        """Get number of image channels for a dataset"""
        dataset_name = dataset_name.lower()
        if dataset_name in ['mnist', 'fashionmnist']:
            return 1
        return 3
