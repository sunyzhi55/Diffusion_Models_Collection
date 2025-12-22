"""
Configuration file for CIFAR-10 training with UNet
"""

config = {
    # Project
    'project_name': 'diffusion-models',
    'experiment_name': 'cifar10-unet-ddpm',
    
    # Model
    'model_type': 'unet',  # 'unet', 'dit', 'dim'
    'model_params': {
        'image_size': (32, 32),
        'in_channels': 3,
        'model_channels': 128,
        'out_channels': 3,
        'num_res_blocks': 2,
        'attention_resolutions': (16, 8),
        'dropout': 0.1,
        'channel_mult': (1, 2, 2, 2),
        'use_attention': True,
    },
    
    # Dataset
    'dataset': 'custom',  # 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'celeba', 'custom'
    'data_root': '/data3/wangchangmiao/shenxy/PublicDataset/oxfordFlowers/prepare_pic/test',
    'image_size': (32, 32),
    'conditional': False,  # Whether to use labels
    'num_classes': 102,
    'use_subdirs': True,  # For custom dataset
    'label_file': None,    # For custom dataset
    
    # Diffusion
    # Note: Training always uses DDPM for loss computation
    # Sampling method is selected via --sampling_method argument in sample.py
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'linear',  # 'linear', 'cosine', 'quadratic'
    'loss_type': 'l2',  # 'l1', 'l2', 'huber'
    'cfg_scale': 1.3,  # Classifier-Free Guidance scale
    
    # For DDIM sampling
    'num_inference_steps': 50,
    'ddim_eta': 0.0,
    
    # Training
    'epochs': 200,
    'batch_size': 128,
    'num_workers': 4,
    'optimizer': 'adamw',
    'learning_rate': 2e-4,
    'weight_decay': 1e-4,
    'gradient_accumulation_steps': 1,
    'use_ema': True,
    'ema_decay': 0.9999,
    'cfg_dropout_prob': 0.2,
    
    # Learning rate schedule
    'use_scheduler': True,
    'scheduler_type': 'cosine',  # 'cosine', 'linear', 'step', 'warmup_cosine'
    'warmup_epochs': 10, # For 'warmup_cosine' scheduler, about 0.5% ~ 2% of total epochs
    'warmup_start_factor': 0.01,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_interval': 10,
    'resume_path': None, # Path to checkpoint to resume training from
    
    # Sampling
    'sample_dir': './generated_images',
    'sample_interval': 20,
    'sample_start_epoch': 200,
    'num_samples': 16,
    
    # Monitoring
    'use_swanlab': False,
    
    # GPU settings
    # Single GPU: specify GPU ID (e.g., 0, 1, 2, etc.)
    # Multi-GPU: specify list of GPU IDs (e.g., [0, 1, 2, 3])
    # Note: --gpus command line argument will override this setting
    'gpu_ids': [0],  # Single GPU ID, or list for multi-GPU
    'port': '12355',  # Port for distributed training
    
    # Random seed
    'seed': 42,
}
