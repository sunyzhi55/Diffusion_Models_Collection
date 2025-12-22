"""
Custom dataset for user-provided images
"""

from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, List
import json


class CustomImageDataset(Dataset):
    """
    Custom dataset for loading images from a directory
    
    Supports two modes:
    1. Unconditional: Just load images from a directory
    2. Conditional: Load images with labels from a JSON file or subdirectory structure
    
    Directory structure for conditional (subdirectory mode):
        root/
            class_0/
                img1.jpg
                img2.jpg
            class_1/
                img3.jpg
                img4.jpg
    
    JSON format for conditional:
        {
            "img1.jpg": 0,
            "img2.jpg": 0,
            "img3.jpg": 1,
            ...
        }
    """
    
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        conditional: bool = False,
        label_file: Optional[str] = None,
        use_subdirs: bool = False
    ):
        """
        Args:
            root: Root directory containing images
            transform: Transform to apply to images
            conditional: Whether to return labels
            label_file: Path to JSON file with labels (if conditional=True)
            use_subdirs: Use subdirectory names as labels (if conditional=True)
        """
        super().__init__()
        
        self.root = Path(root)
        self.transform = transform
        self.conditional = conditional
        self.use_subdirs = use_subdirs
        
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        if self.conditional and not (use_subdirs or label_file):
            raise ValueError(
                "CustomImageDataset with conditional=True requires either use_subdirs=True or a label_file."
            )
        
        # if conditional and use_subdirs:
        #     self._load_with_subdirs()
        # elif conditional and label_file:
        #     self._load_with_json(label_file)
        # else:
        #     self._load_images_only()
        if use_subdirs:
            self._load_with_subdirs()
        elif label_file:
            self._load_with_json(label_file)
        else:
            self._load_images_only()
    
    def _load_images_only(self):
        """Load images without labels"""
        for path in self.root.iterdir():
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                self.images.append(path)
    
    def _load_with_subdirs(self):
        """Load images from subdirectories, using directory names as labels"""
        classes = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {cls.name: idx for idx, cls in enumerate(classes)}
        
        for class_dir in classes:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in class_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    self.images.append(img_path)
                    self.labels.append(class_idx)
    
    def _load_with_json(self, label_file):
        """Load images with labels from JSON file"""
        label_path = Path(label_file)
        with label_path.open('r', encoding='utf-8') as f:
            labels_dict = json.load(f)
        
        for filename, label in labels_dict.items():
            img_path = self.root / filename
            if img_path.exists():
                self.images.append(img_path)
                self.labels.append(label)
        
        # Build class_to_idx
        unique_labels = sorted(set(self.labels))
        self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        # Remap labels to consecutive indices [0, num_classes-1]
        self.labels = [self.class_to_idx[l] for l in self.labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        # Return with or without label
        if self.conditional:
            label = self.labels[idx]
            return img, label
        return img
    
    @property
    def num_classes(self):
        """Get number of classes"""
        if self.conditional:
            return len(self.class_to_idx)
        return 0
    @staticmethod
    def get_default_transform(image_size=32, dataset_type='rgb', train=True):
        """Get default transform for a dataset.

        - train=True  : keep light augmentation (flip)
        - train=False : deterministic preprocessing only
        """
        dataset_type = dataset_type.lower()
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
