"""DDPM (Denoising Diffusion Probabilistic Models).

Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).

This implementation also supports Classifier-Free Guidance (CFG) sampling and an optional
Dynamic Thresholding step (Imagen-style) applied to the predicted x0.
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DDPM:
    """
    DDPM diffusion process
    
    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Start value for beta schedule
        beta_end: End value for beta schedule
        beta_schedule: Type of beta schedule ('linear', 'cosine', 'quadratic')
        device: Device to run on
    """
    
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='linear',
        device='cuda'
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Create beta schedule
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, device=device)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, device=device) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008, device='cuda'):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Original images (B, C, H, W)
            t: Timesteps (B,)
            noise: Noise to add (B, C, H, W), if None, sample from N(0, I)
        
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
        Compute training loss
        
        Args:
            model: Denoising model
            x_start: Original images (B, C, H, W)
            t: Timesteps (B,)
            y: Class labels (B,) for conditional generation
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
        # Ensure 'a' is on the same device as 't' for multi-GPU training
        a = a.to(t.device)
        # Use direct indexing for clarity and correct broadcasting
        out = a[t]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def p_mean_variance(self, model, x, t, y=None, clip_denoised=True, eps=None, x0_pred=None):
        """
        Compute mean and variance of posterior distribution p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x: Current noisy image (B, C, H, W)
            t: Current timestep (B,)
            y: Class labels (B,)
            clip_denoised: Whether to clip denoised image to [-1, 1]
        
        Returns:
            Posterior mean, posterior variance, posterior log variance
        """
        # Predict noise (epsilon)
        if eps is None:
            eps = model(x, t, y)
        
        # Compute x_0 from x_t and predicted noise
        sqrt_recip_alphas_cumprod_t = self._extract(
            torch.sqrt(1.0 / self.alphas_cumprod), t, x.shape
        )
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x.shape
        )
        
        if x0_pred is None:
            x0_pred = sqrt_recip_alphas_cumprod_t * x - sqrt_recipm1_alphas_cumprod_t * eps

        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # Compute posterior mean
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x.shape)
        
        posterior_mean = posterior_mean_coef1_t * x0_pred + posterior_mean_coef2_t * x
        
        # Posterior variance
        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        posterior_log_variance_clipped_t = self._extract(
            self.posterior_log_variance_clipped, t, x.shape
        )
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t
    
    @torch.no_grad()
    def p_sample(self, model, x, t, y=None, clip_denoised=True, eps=None, x0_pred=None):
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x: Current noisy image (B, C, H, W)
            t: Current timestep (B,)
            y: Class labels (B,)
            clip_denoised: Whether to clip denoised image
        
        Returns:
            Previous timestep image
        """
        posterior_mean, _, posterior_log_variance = self.p_mean_variance(
            model, x, t, y, clip_denoised, eps=eps, x0_pred=x0_pred
        )
        
        noise = torch.randn_like(x)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
    
    @torch.no_grad()
    def sample(self, model, shape, y=None, return_all_timesteps=False):
        """
        Generate samples using DDPM
        
        Args:
            model: Denoising model
            shape: Shape of samples (B, C, H, W)
            y: Class labels (B,) for conditional generation
            return_all_timesteps: Whether to return all intermediate timesteps
        
        Returns:
            Generated samples, optionally with all timesteps
        """
        batch_size = shape[0]
        device = self.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, y)
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=0)
        return img
    
    @torch.no_grad()
    def sample_with_cfg(
        self,
        model,
        shape,
        y,
        cfg_scale=3.0,
        p_threshold=0.995,
        return_all_timesteps=False,
    ):
        """
        Generate samples using Classifier-Free Guidance (CFG) with optional Dynamic Thresholding.

        Notes:
        - CFG acts on epsilon only.
        - Dynamic Thresholding acts on x0 only.
        - DDPM posterior sampling is reused via p_sample().
        """

        if y is None:
            raise ValueError("CFG sampling requires class labels y.")
        if p_threshold is not None and not (0.0 < float(p_threshold) < 1.0):
            raise ValueError("p_threshold must be in (0, 1) or None")

        batch_size = shape[0]
        device = self.device
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = []

        y = y.to(device)
        y_uncond = torch.zeros_like(y)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc=f'DDPM Sampling with CFG scale {cfg_scale}', total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # 1) CFG epsilon
            eps_cond = model(img, t, y)
            eps_uncond = model(img, t, y_uncond)
            eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # 2) Predict x0 from guided epsilon
            sqrt_recip_alphas_cumprod_t = self._extract(
                torch.sqrt(1.0 / self.alphas_cumprod), t, img.shape
            )
            sqrt_recipm1_alphas_cumprod_t = self._extract(
                self.sqrt_recipm1_alphas_cumprod, t, img.shape
            )

            x0_pred = sqrt_recip_alphas_cumprod_t * img - sqrt_recipm1_alphas_cumprod_t * eps_guided

            # 3) Dynamic Thresholding (Imagen-style)
            if p_threshold is not None:
                flat = x0_pred.reshape(batch_size, -1)
                s = torch.quantile(flat.abs(), float(p_threshold), dim=1)
                s = torch.maximum(s, torch.ones_like(s))
                s = s.view(batch_size, 1, 1, 1)
                x0_pred = torch.clamp(x0_pred, -s, s) / s
            else:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # 4) Reuse DDPM posterior via p_sample()
            img = self.p_sample(
                model,
                img,
                t,
                y=None,
                clip_denoised=False,
                eps=eps_guided,
                x0_pred=x0_pred,
            )
            
            if return_all_timesteps:
                imgs.append(img.cpu())
        
        if return_all_timesteps:
            return torch.stack(imgs, dim=0)
        return img
