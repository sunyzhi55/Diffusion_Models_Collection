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
        'image_size': 32,
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
    'dataset': 'cifar10',  # 'cifar10', 'cifar100', 'mnist', 'fashionmnist', 'celeba', 'custom'
    'data_root': './data',
    'image_size': 32,
    'conditional': True,  # Whether to use labels
    'num_classes': 10,
    
    # Diffusion
    'diffusion_type': 'ddpm',  # 'ddpm' or 'ddim'
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'linear',  # 'linear', 'cosine', 'quadratic'
    'loss_type': 'l2',  # 'l1', 'l2', 'huber'
    
    # For DDIM sampling
    'num_inference_steps': 50,
    'ddim_eta': 0.0,
    
    # Training
    'epochs': 200,
    'batch_size': 128,
    'num_workers': 4,
    'learning_rate': 2e-4,
    'weight_decay': 0.0,
    'gradient_accumulation_steps': 1,
    'use_ema': True,
    'ema_decay': 0.9999,
    
    # Learning rate schedule
    'use_scheduler': True,
    'scheduler_type': 'cosine',  # 'cosine', 'linear', 'step'
    'warmup_epochs': 10,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_interval': 10,
    
    # Sampling
    'sample_dir': './generated_images',
    'sample_interval': 5,
    'sample_start_epoch': 20,
    'num_samples': 16,
    
    # Monitoring
    'use_swanlab': True,
    
    # Distributed training
    'distributed': False,
    'world_size': 1,
    
    # Random seed
    'seed': 42,
}
