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
            self.betas = self._cosine_beta_schedule(num_timesteps, device=device)
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
        # # Use uniform spacing
        # step = self.num_timesteps // self.num_inference_steps
        # self.inference_timesteps = torch.arange(0, self.num_timesteps, step, device=self.device)
        # self.inference_timesteps = torch.flip(self.inference_timesteps, [0])
        # 生成从 0 到 num_timesteps-1 的均匀浮点间隔
        timesteps = torch.linspace(
            self.num_timesteps - 1,  # 起始（最大 t）
            0,                       # 结束（最小 t）
            self.num_inference_steps,
            device=self.device
        )
        # 转为整数（四舍五入或向下取整）
        self.inference_timesteps = timesteps.round().long()
    
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
        # Ensure 'a' is on the same device as 't' for multi-GPU training
        a = a.to(t.device)
        # Use direct indexing for clarity and correct broadcasting
        out = a[t]
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    @torch.no_grad()
    def p_sample(self, model, x, t, t_next, y=None, clip_denoised=True, eps=None, x0_pred=None):
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

        if eps is None:
            eps = model(x, t, y)

        alpha_t = self._extract(self.alphas_cumprod, t, x.shape)

        if t_next.min() >= 0:
            alpha_t_next = self._extract(self.alphas_cumprod, t_next, x.shape)
        else:
            alpha_t_next = torch.ones_like(alpha_t)

        # alpha_t_next = torch.where(t_next >= 0,
        #                    self._extract(self.alphas_cumprod, t_next, x.shape),
        #                    torch.ones_like(alpha_t))

        # predict x0
        if x0_pred is None:
            x0_pred = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

        if clip_denoised:
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # DDIM sigma
        sigma_t = self.eta * torch.sqrt(
            torch.clamp(
                (1 - alpha_t_next) / (1 - alpha_t) *
                (1 - alpha_t / alpha_t_next),
                min=0.0
            )
        )
        # direction pointing to x_t
        dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_next - sigma_t ** 2, min=0.0)) * eps

        x_prev = torch.sqrt(alpha_t_next) * x0_pred + dir_xt

        if self.eta > 0:
            x_prev = x_prev + sigma_t * torch.randn_like(x)

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
        DDIM sampling with Classifier-Free Guidance and Dynamic Thresholding
        (Imagen / Stable Diffusion style)

        - CFG acts on epsilon only
        - Dynamic Thresholding acts on x0 only
        - DDIM update is untouched
        """

        if y is None:
            raise ValueError("CFG sampling requires class labels y.")
        if p_threshold is not None and not (0.0 < float(p_threshold) < 1.0):
            raise ValueError("p_threshold must be in (0, 1) or None")

        batch_size = shape[0]
        device = self.device

        img = torch.randn(shape, device=device)
        imgs = []

        timesteps = self.inference_timesteps.tolist()

        y = y.to(device)
        y_uncond = torch.zeros_like(y)

        for i, t in enumerate(tqdm(timesteps, desc=f"DDIM sampling with CFG scale {cfg_scale}")):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if i < len(timesteps) - 1:
                t_next = torch.full(
                    (batch_size,),
                    timesteps[i + 1],
                    device=device,
                    dtype=torch.long,
                )
            else:
                t_next = torch.full((batch_size,), -1, device=device, dtype=torch.long)

            # 1) CFG epsilon
            eps_cond = model(img, t_batch, y)
            eps_uncond = model(img, t_batch, y_uncond)
            eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

            # ------------------------------------------------
            # 2. Predict x0 from epsilon
            # ------------------------------------------------
            alpha_t = self._extract(self.alphas_cumprod, t_batch, img.shape)

            # if t_next.min() >= 0:
            #     alpha_t_next = self._extract(self.alphas_cumprod, t_next, img.shape)
            # else:
            #     alpha_t_next = torch.ones_like(alpha_t)


            x0_pred = (img - torch.sqrt(1 - alpha_t) * eps_guided) / torch.sqrt(alpha_t)

            # ------------------------------------------------
            # 3. Dynamic Thresholding (Imagen)
            # ------------------------------------------------
            if p_threshold is not None:
                flat = x0_pred.view(batch_size, -1)
                s = torch.quantile(flat.abs(), p_threshold, dim=1)
                s = torch.maximum(s, torch.ones_like(s))
                s = s.view(batch_size, 1, 1, 1)
                x0_pred = torch.clamp(x0_pred, -s, s) / s
            else:
                x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # 4) Reuse DDIM update via p_sample()
            img = self.p_sample(
                model,
                img,
                t_batch,
                t_next,
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
    
    def set_inference_steps(self, num_inference_steps):
        """Update number of inference steps"""
        self.num_inference_steps = num_inference_steps
        self._setup_inference_timesteps()
