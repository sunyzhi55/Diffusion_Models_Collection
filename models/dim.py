"""
DiM (Diffusion Mamba) architecture
Combining Mamba's efficient sequence modeling with diffusion models
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    # Fallback implementation using attention
    pass


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size: Tuple[int, int] = (32, 32), patch_size=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.h_tokens = img_size[0] // patch_size
        self.w_tokens = img_size[1] // patch_size
        self.num_patches = self.h_tokens * self.w_tokens
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size
        
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations"""
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size, padding_idx=0)  # 0 is null/unconditional
        
        # add more layers for better capacity
        # self.embedding_table = nn.Sequential(
        #     # 类别为0时表示无条件生成
        #     nn.Embedding(num_embeddings=num_classes + 1, embedding_dim=hidden_size, padding_idx=0),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        # self.num_classes = num_classes
        # self.dropout_prob = dropout_prob
        
    def forward(self, labels, train=True):
        """
        The trainer handles CFG label dropout, so disable internal dropout here.
        """
        # if self.dropout_prob > 0 and train:
        #     mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        #     labels = torch.where(mask, torch.zeros_like(labels), labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class MambaBlock(nn.Module):
    """
    Mamba block for sequence modeling
    Falls back to attention if mamba-ssm is not available
    """
    def __init__(self, hidden_size, state_size=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=hidden_size,
                d_state=state_size,
                d_conv=4,
                expand=2
            )
        else:
            # Fallback to multi-head attention
            self.mamba = nn.MultiheadAttention(
                hidden_size, 
                num_heads=8, 
                dropout=dropout, 
                batch_first=True
            )
        
        # Conditioning modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        """
        Args:
            x: (B, N, hidden_size)
            c: (B, hidden_size) conditioning
        """
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        
        h = self.norm(x)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        if MAMBA_AVAILABLE:
            h = self.mamba(h)
        else:
            h = self.mamba(h, h, h)[0]
        
        x = x + gate.unsqueeze(1) * h
        
        return x


class FeedForward(nn.Module):
    """Feed-forward network with conditioning"""
    def __init__(self, hidden_size, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        
        h = self.norm(x)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate.unsqueeze(1) * h
        
        return x


class DiMBlock(nn.Module):
    """DiM Block combining Mamba and Feed-forward"""
    def __init__(self, hidden_size, state_size=16, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.mamba_block = MambaBlock(hidden_size, state_size, dropout)
        self.ff_block = FeedForward(hidden_size, mlp_ratio, dropout)
        
    def forward(self, x, c):
        x = self.mamba_block(x, c)
        x = self.ff_block(x, c)
        return x


class FinalLayer(nn.Module):
    """Final layer to project back to image space"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiM(nn.Module):
    """
    Diffusion Mamba (DiM)
    
    Args:
        img_size: Image resolution
        patch_size: Patch size for patch embedding
        in_channels: Number of input channels
        hidden_size: Hidden dimension
        depth: Number of DiM blocks
        state_size: State size for Mamba
        mlp_ratio: MLP hidden dimension ratio
        num_classes: Number of classes for conditional generation (None for unconditional)
        dropout: Dropout rate
    """
    def __init__(
        self,
        img_size: Tuple[int, int] = (32, 32),
        patch_size=2,
        in_channels=3,
        hidden_size=768,
        depth=12,
        state_size=16,
        mlp_ratio=4.0,
        num_classes=None,
        dropout=0.1
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Patch embedding
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.x_embedder.num_patches
        self.h_tokens = self.x_embedder.h_tokens
        self.w_tokens = self.x_embedder.w_tokens
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # Time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Label embedding
        if num_classes is not None:
            # IMPORTANT:
            # - Reserve label index 0 as unconditional / null token.
            # - CFG label dropout is handled by the trainer (cfg_dropout_prob),
            #   so disable internal label dropout to avoid double-dropping.
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.0)
        else:
            self.y_embedder = None
        
        # DiM blocks
        self.blocks = nn.ModuleList([
            DiMBlock(hidden_size, state_size, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Zero-out modulation layers
        for block in self.blocks:
            nn.init.constant_(block.mamba_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.mamba_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.ff_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.ff_block.adaLN_modulation[-1].bias, 0)
            
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
    
    def unpatchify(self, x):
        """
        x: (B, N, patch_size^2 * C)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        h = self.h_tokens
        w = self.w_tokens
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_channels, h * p, w * p))
        return imgs
    
    def forward(self, x, t, y=None):
        """
        Forward pass
        
        Args:
            x: (B, C, H, W) input images
            t: (B,) timesteps
            y: (B,) class labels for conditional generation
        """
        # Patch embedding
        x = self.x_embedder(x) + self.pos_embed
        
        # Time embedding
        t_emb = self.t_embedder(t)
        
        # Label embedding
        if self.y_embedder is not None and y is not None:
            # Clamp to include null class index for CFG masking
            y = torch.clamp(y, 0, self.num_classes)
            y_emb = self.y_embedder(y, self.training)
            c = t_emb + y_emb
        else:
            c = t_emb
        
        # DiM blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x
