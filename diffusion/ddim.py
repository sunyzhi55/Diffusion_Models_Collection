"""
DDIM (Denoising Diffusion Implicit Models)
Based on "Denoising Diffusion Implicit Models" (Song et al., 2020)
Faster sampling than DDPM
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


class DDIM:
    """
    DDIM diffusion process with accelerated sampling
    
    Args:
        num_timesteps: Number of diffusion timesteps (for training)
        num_inference_steps: Number of steps for inference (can be much smaller)
        beta_start: Start value for beta schedule
        beta_end: End value for beta schedule
        beta_schedule: Type of beta schedule ('linear', 'cosine', 'quadratic')
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
        device: Device to run on
    """
    
    def __init__(
        self,
        num_timesteps=1000,
        num_inference_steps=50,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        eta=0.0,
        device='cuda'
    ):
        self.num_timesteps = num_timesteps
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.device = device
        
        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, device)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, device=device) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Create inference timesteps
        self._setup_inference_timesteps()
    
    def _cosine_beta_schedule(self, timesteps, s=0.008, device='cuda'):
        """Cosine schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def _setup_inference_timesteps(self):
        """Setup timesteps for inference"""
        # Use uniform spacing
        step = self.num_timesteps // self.num_inference_steps
        self.inference_timesteps = torch.arange(0, self.num_timesteps, step, device=self.device)
        self.inference_timesteps = torch.flip(self.inference_timesteps, [0])
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Original images (B, C, H, W)
            t: Timesteps (B,)
            noise: Noise to add (B, C, H, W)
        
        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, model, x_start, t, y=None, noise=None, loss_type='l2'):
        """
        Compute training loss (same as DDPM)
        
        Args:
            model: Denoising model
            x_start: Original images (B, C, H, W)
            t: Timesteps (B,)
            y: Class labels (B,)
            noise: Noise to add
            loss_type: 'l1' or 'l2' or 'huber'
        
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = model(x_noisy, t, y)
        
        # Compute loss
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == 'huber':
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
    
    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_next, y=None, clip_denoised=True):
        """
        Sample x_{t_next} from x_t using DDIM
        
        Args:
            model: Denoising model
            x: Current noisy image (B, C, H, W)
            t: Current timestep (B,)
            t_next: Next timestep (B,)
            y: Class labels (B,)
            clip_denoised: Whether to clip predicted x_0
        
        Returns:
            Image at next timestep
        """
        # Predict noise
        predicted_noise = model(x, t, y)
        
        # Get alpha values
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
        
        if t_next[0] >= 0:
            alpha_cumprod_t_next = self._extract(self.alphas_cumprod, t_next, x.shape)
        else:
            alpha_cumprod_t_next = torch.ones_like(alpha_cumprod_t)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        # Compute direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_next - self.eta ** 2 * (
            (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_next)
        )) * predicted_noise
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + dir_xt
        
        # Add stochasticity
        if self.eta > 0:
            noise = torch.randn_like(x)
            sigma_t = self.eta * torch.sqrt(
                (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_next)
            )
            x_prev = x_prev + sigma_t * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample(self, model, shape, y=None, return_all_timesteps=False):
        """
        Generate samples using DDIM
        
        Args:
            model: Denoising model
            shape: Shape of samples (B, C, H, W)
            y: Class labels (B,) for conditional generation
            return_all_timesteps: Whether to return all intermediate timesteps
        
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        device = self.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        timesteps = self.inference_timesteps.tolist()
        
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling')):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Get next timestep
            if i < len(timesteps) - 1:
                t_next = torch.full((batch_size,), timesteps[i + 1], device=device, dtype=torch.long)
            else:
                t_next = torch.full((batch_size,), -1, device=device, dtype=torch.long)
            
            img = self.p_sample(model, img, t_batch, t_next, y)
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=0)
        return img
    
    @torch.no_grad()
    def sample_with_cfg(self, model, shape, y, cfg_scale=3.0, return_all_timesteps=False):
        """
        Generate samples using Classifier-Free Guidance with DDIM
        
        Args:
            model: Denoising model
            shape: Shape of samples (B, C, H, W)
            y: Class labels (B,)
            cfg_scale: Guidance scale
            return_all_timesteps: Whether to return all intermediate timesteps
        
        Returns:
            Generated samples
        """
        batch_size = shape[0]
        device = self.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        timesteps = self.inference_timesteps.tolist()
        
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Sampling with CFG')):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict with and without conditioning
            noise_pred_cond = model(img, t_batch, y)
            noise_pred_uncond = model(img, t_batch, None)
            
            # Apply classifier-free guidance
            predicted_noise = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Get alpha values
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t_batch, img.shape)
            
            if i < len(timesteps) - 1:
                alpha_cumprod_t_next = self._extract(
                    self.alphas_cumprod, 
                    torch.full((batch_size,), timesteps[i + 1], device=device, dtype=torch.long),
                    img.shape
                )
            else:
                alpha_cumprod_t_next = torch.ones_like(alpha_cumprod_t)
            
            # Predict x_0
            pred_x0 = (img - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
            # Compute direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_next) * predicted_noise
            
            # Compute x_{t-1}
            img = torch.sqrt(alpha_cumprod_t_next) * pred_x0 + dir_xt
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=0)
        return img
    
    def set_inference_steps(self, num_inference_steps):
        """Update number of inference steps"""
        self.num_inference_steps = num_inference_steps
        self._setup_inference_timesteps()
