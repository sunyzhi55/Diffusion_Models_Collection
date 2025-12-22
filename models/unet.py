"""
UNet architecture for diffusion models
Supports both conditional and unconditional generation
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time and label conditioning"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes=None, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.label_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels, bias=False)
        ) if num_classes is not None else None
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, t_emb, y_emb=None):
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        # Add shared label embedding if provided
        if self.label_proj is not None and y_emb is not None:
            h = h + self.label_proj(y_emb)[:, :, None, None]
        
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(C // self.num_heads)
        attn = torch.softmax(attn, dim=-1)
        
        h = torch.matmul(attn, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet model for diffusion
    
    Args:
        image_size: Image resolution
        in_channels: Number of input channels
        model_channels: Base channel count
        out_channels: Number of output channels
        num_res_blocks: Number of residual blocks per level
        attention_resolutions: Resolutions to apply attention
        dropout: Dropout rate
        channel_mult: Channel multiplier for each level
        num_classes: Number of classes for conditional generation (None for unconditional)
        use_attention: Whether to use attention blocks
    """
    def __init__(
        self,
        image_size: Tuple[int, int] = (32, 32),  # 修改这里，指定默认值为一个元组
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        dropout=0.1,
        channel_mult=(1, 2, 2, 2),
        num_classes=None,
        use_attention=True
    ):
        super().__init__()
        
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Shared label embedding table (index 0 is null / CFG)
        if num_classes is not None:
        #     self.label_embed = nn.Sequential(
        #     # 类别为0时表示无条件生成
        #     nn.Embedding(num_embeddings=num_classes + 1, embedding_dim=model_channels, padding_idx=0),
        #     nn.Linear(model_channels, time_emb_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_emb_dim, time_emb_dim),
        # )
            self.label_embed = nn.Embedding(num_embeddings=num_classes + 1, embedding_dim=time_emb_dim, padding_idx=0)
        else:
            self.label_embed = None
            
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_channels = [ch]
        resolution = list(image_size)
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, out_ch, time_emb_dim, num_classes, dropout)]
                ch = out_ch
                # Apply attention if the current resolution matches any attention_resolutions
                if use_attention and (resolution[0] in attention_resolutions or resolution[1] in attention_resolutions):
                    layers.append(AttentionBlock(ch))
                self.down_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)
            
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_block_channels.append(ch)
                resolution[0] //= 2
                resolution[1] //= 2
        
        # Middle
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim, num_classes, dropout),
            AttentionBlock(ch) if use_attention else nn.Identity(),
            ResidualBlock(ch, ch, time_emb_dim, num_classes, dropout)
        ])
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for level, mult in enumerate(reversed(channel_mult)):
            for i in range(num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [ResidualBlock(ch + ich, model_channels * mult, time_emb_dim, num_classes, dropout)]
                ch = model_channels * mult
                # During upsampling check the current resolution similarly
                if use_attention and (resolution[0] in attention_resolutions or resolution[1] in attention_resolutions):
                    layers.append(AttentionBlock(ch))
                if level != len(channel_mult) - 1 and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    resolution[0] *= 2
                    resolution[1] *= 2
                self.up_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1)
        )
    
    def forward(self, x, t, y=None):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            t: Timestep tensor (B,)
            y: Class labels (B,) for conditional generation
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Label embedding (shared)
        if self.num_classes is not None and y is not None:
            y = torch.clamp(y, 0, self.num_classes)
            y_emb = self.label_embed(y)
        else:
            y_emb = None
        
        # Input
        h = self.input_conv(x)
        hs = [h]
        
        # Downsampling
        for block in self.down_blocks:
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb, y_emb)
                else:
                    h = layer(h)
            hs.append(h)
        
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb, y_emb)
            else:
                h = layer(h)
        
        # Upsampling
        for block in self.up_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                if isinstance(layer, ResidualBlock):
                    h = layer(h, t_emb, y_emb)
                else:
                    h = layer(h)
        
        # Output
        return self.output(h)
