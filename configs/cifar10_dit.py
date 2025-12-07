"""
Configuration file for CIFAR-10 training with DiT
"""

config = {
    # Project
    'project_name': 'diffusion-models',
    'experiment_name': 'cifar10-dit-ddpm',
    
    # Model
    'model_type': 'dit',  # 'unet', 'dit', 'dim'
    'model_params': {
        'img_size': 32,
        'patch_size': 2,
        'in_channels': 3,
        'hidden_size': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
    },
    
    # Dataset
    'dataset': 'cifar10',
    'data_root': './data',
    'image_size': 32,
    'conditional': True,
    'num_classes': 10,
    
    # Diffusion
    'diffusion_type': 'ddim',  # 'ddpm' or 'ddim'
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'beta_schedule': 'linear',
    'loss_type': 'l2',
    
    # For DDIM sampling
    'num_inference_steps': 50,
    'ddim_eta': 0.0,
    
    # Training
    'epochs': 300,
    'batch_size': 128,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 0.0,
    'gradient_accumulation_steps': 1,
    'use_ema': True,
    'ema_decay': 0.9999,
    
    # Learning rate schedule
    'use_scheduler': True,
    'scheduler_type': 'cosine',
    'warmup_epochs': 10,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_interval': 10,
    
    # Sampling
    'sample_dir': './generated_images',
    'sample_interval': 5,
    'sample_start_epoch': 30,
    'num_samples': 16,
    
    # Monitoring
    'use_swanlab': True,
    
    # Distributed training
    'distributed': False,
    'world_size': 1,
    
    # Random seed
    'seed': 42,
}
