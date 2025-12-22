"""
Trainer for diffusion models
Supports single-GPU and multi-GPU training with DDP
"""

from pathlib import Path
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import swanlab
from torchvision.utils import save_image
import time

from utils.helpers import resolve_image_size


class DiffusionTrainer:
    """
    Trainer for diffusion models
    
    Args:
        model: Diffusion model (UNet, DiT, or DiM)
        diffusion: Diffusion process (DDPM or DDIM)
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        config: Training configuration
        rank: Process rank for DDP
        world_size: Number of processes for DDP
    """
    
    def __init__(
        self,
        model,
        diffusion,
        train_loader,
        optimizer,
        scheduler=None,
        device='cuda',
        config=None,
        rank=0,
        world_size=1,
        resume_path=None
    ):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.is_main_process = rank == 0
        
        # Model
        self.model = model.to(device)
        if self.is_distributed:
            # DDP will automatically use the device set by torch.cuda.set_device()
            # Do not specify device_ids to avoid GPU mapping issues
            self.model = DDP(model)
        
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # Config
        self.config = config or {}
        self.epochs = self.config.get('epochs', 100)
        self.save_dir = Path(self.config.get('save_dir', './checkpoints'))
        self.sample_dir = Path(self.config.get('sample_dir', './generated_images'))
        self.loss_type = self.config.get('loss_type', 'l2')
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        self.save_interval = self.config.get('save_interval', 10)
        self.sample_interval = self.config.get('sample_interval', 5)
        self.sample_start_epoch = self.config.get('sample_start_epoch', 20)
        self.num_samples = self.config.get('num_samples', 16)
        self.cfg_dropout_prob = self.config.get('cfg_dropout_prob', 0.2)
        self.cfg_scale = self.config.get('cfg_scale', 1.8)
        self.use_ema = self.config.get('use_ema', False)
        self.ema_decay = self.config.get('ema_decay', 0.9999)
        # Default to False to avoid requiring swanlab when not configured
        self.use_swanlab = self.config.get('use_swanlab', False)
        self.conditional = self.config.get('conditional', False)
        self.num_classes = self.config.get('num_classes', None)
        self.image_size = resolve_image_size(self.config.get('image_size', 32))
        self.model_type = self.config.get('model_type', 'unet').lower()
        self.model_params = self.config.get('model_params', {}).copy()
        # Prefer model_params in_channels; fall back to top-level config and then default.
        self.in_channels = self.model_params.get('in_channels',  3)
        
        # Create directories
        if self.is_main_process:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # EMA model
        if self.use_ema and self.is_main_process:
            self.ema_model = self._create_ema_model()
        else:
            self.ema_model = None
        
        # Best loss tracking
        self.best_loss = float('inf')
        self.start_epoch = 1
        
        # Resume from checkpoint
        if resume_path:
            self.load_checkpoint(resume_path)
        
        # Initialize SwanLab
        if self.use_swanlab and self.is_main_process:
            swanlab.init(
                project=self.config.get('project_name', 'diffusion-models'),
                experiment_name=self.config.get('experiment_name', 'experiment'),
                config=self.config
            )
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA model state
        if 'ema_model_state_dict' in checkpoint and self.ema_model:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        
        # Load other info
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"Resuming training from epoch {self.start_epoch}")
        
        # Check if we need to extend training
        if self.start_epoch > self.epochs:
            print(f"Checkpoint epoch ({self.start_epoch-1}) is greater than configured epochs ({self.epochs}).")
            print(f"Extending training by {self.config.get('epochs', 100)} epochs...")
            self.epochs = self.start_epoch + self.config.get('epochs', 100)
            print(f"New target epochs: {self.epochs}")

    def _create_ema_model(self):
        """Create EMA model"""
        if self.is_distributed:
            ema_model = type(self.model.module)(
                **self._get_model_params()
            ).to(self.device)
            ema_model.load_state_dict(self.model.module.state_dict())
        else:
            ema_model = type(self.model)(
                **self._get_model_params()
            ).to(self.device)
            ema_model.load_state_dict(self.model.state_dict())
        
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _get_model_params(self):
        """Get model initialization parameters based on model type"""
        params = self.model_params.copy()
        
        # Add conditional info
        if self.conditional:
            params['num_classes'] = self.num_classes
        else:
            params['num_classes'] = None
        
        return params
    
    def _update_ema(self):
        """Update EMA model"""
        if self.ema_model is None:
            return
        
        if self.is_distributed:
            model_params = self.model.module.state_dict()
        else:
            model_params = self.model.state_dict()
        
        ema_params = self.ema_model.state_dict()
        
        for name in ema_params:
            ema_params[name].mul_(self.ema_decay).add_(
                model_params[name], alpha=1 - self.ema_decay
            )
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            disable=not self.is_main_process
        )
        
        self.optimizer.zero_grad()
        
        for i, batch in enumerate(progress_bar):
            # Parse batch
            if self.conditional:
                images, labels = batch
                labels = labels.to(self.device)
                # Shift labels by +1; reserve 0 as null for CFG
                labels_for_loss = labels + 1
                if self.cfg_dropout_prob > 0 and self.num_classes is not None:
                    drop_mask = torch.rand_like(labels.float()) < self.cfg_dropout_prob
                    labels_for_loss = labels_for_loss.clone()
                    labels_for_loss[drop_mask] = 0  # null class index
            else:
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                labels = None
                labels_for_loss = None
            
            images = images.to(self.device)
            
            # Sample timesteps
            batch_size = images.shape[0]
            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch_size,), device=self.device
            ).long()
            
            # Compute loss
            loss = self.diffusion.p_losses(
                self.model, images, t, labels_for_loss, loss_type=self.loss_type
            )
            
            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            if (i + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update EMA
                if self.use_ema:
                    self._update_ema()
            
            # Track loss
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            if self.is_main_process:
                progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
        
        avg_loss = total_loss / num_batches
        
        # Synchronize loss across processes
        if self.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        return avg_loss
    
    @torch.no_grad()
    def sample_images(self, epoch, num_samples=None):
        """Generate sample images"""
        if num_samples is None:
            num_samples = self.num_samples
        
        model = self.ema_model if self.ema_model is not None else self.model
        if self.is_distributed:
            # EMA model is not wrapped in DDP; only unwrap when attribute exists
            model = model.module if hasattr(model, 'module') else model
        
        model.eval()
        
        # Generate samples
        h, w = self.image_size
        shape = (num_samples, self.in_channels, h, w)
        nrow = max(1, int(math.sqrt(num_samples)))
        
        if self.conditional and self.num_classes:
            num_rows = (num_samples + nrow - 1) // nrow
            row_labels = torch.arange(num_rows, device=self.device) % self.num_classes
            labels = (row_labels + 1).repeat_interleave(nrow)[:num_samples]  # shift to avoid 0
            # Ensure each row in the grid shares a single class label
            print(f"Sampling with labels: {labels.cpu().numpy()}")
            samples = self.diffusion.sample_with_cfg(model, shape, labels, cfg_scale=self.cfg_scale)
        else:
            labels = None
            samples = self.diffusion.sample(model, shape, labels)
        
        # Denormalize
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        # Save
        save_path = self.sample_dir / f'epoch_{epoch:04d}.png'
        save_image(samples, str(save_path), nrow=nrow)
        
        # Log to SwanLab
        if self.use_swanlab:
            swanlab.log({'samples': swanlab.Image(samples)}, step=epoch)
        
        return samples
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        if not self.is_main_process:
            return
        
        # Get model state dict
        if self.is_distributed:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema_model is not None:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        
        # Save current checkpoint
        current_path = self.save_dir / 'current_model.pth'
        torch.save(checkpoint, current_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Save periodic checkpoint
        if epoch % self.save_interval == 0:
            epoch_path = self.save_dir / f'model_epoch_{epoch:04d}.pth'
            torch.save(checkpoint, epoch_path)
    
    def train(self):
        """Main training loop"""
        if self.is_main_process:
            print(f"Starting training for {self.epochs} epochs")
            print(f"Device: {self.device}")
            print(f"Distributed: {self.is_distributed} (World size: {self.world_size})")
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log to console
            if self.is_main_process:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f} - "
                      f"LR: {lr:.6f} - Time: {epoch_time:.2f}s")
                
                # Log to SwanLab
                if self.use_swanlab:
                    swanlab.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'train/epoch_time': epoch_time
                    }, step=epoch)
            
            # Save checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            if self.is_main_process:
                self.save_checkpoint(epoch, is_best)
            
            # Generate samples
            if self.is_main_process and epoch >= self.sample_start_epoch and epoch % self.sample_interval == 0:
                print(f"Generating samples at epoch {epoch}...")
                self.sample_images(epoch)
        
        if self.is_main_process:
            print("Training completed!")
            if self.use_swanlab:
                swanlab.finish()
    
    def cleanup(self):
        """Cleanup for distributed training"""
        if self.is_distributed:
            dist.destroy_process_group()
