"""
DiT (Diffusion Transformer) architecture
Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


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
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
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
        The dropout for classifier-free guidance is handled externally in the trainer.
        """
        # if self.dropout_prob > 0 and train:
        #     # Randomly drop labels for classifier-free guidance
        #     mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        #     labels = torch.where(mask, torch.zeros_like(labels), labels)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        
        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x, c):
        """
        Args:
            x: (B, N, hidden_size)
            c: (B, hidden_size) conditioning
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Attention with modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h, h, h)[0]
        x = x + gate_msa.unsqueeze(1) * h
        
        # MLP with modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        
        return x


class FinalLayer(nn.Module):
    """Final layer to project back to image space"""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
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


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)
    
    Args:
        img_size: Image resolution (int or (H, W))
        patch_size: Patch size for patch embedding
        in_channels: Number of input channels
        hidden_size: Hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
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
        num_heads=12,
        mlp_ratio=4.0,
        num_classes=None,
        dropout=0.1
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_h = img_w = img_size
        else:
            img_h, img_w = img_size
        self.img_size = (img_h, img_w)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
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
            # - We reserve label index 0 as the unconditional / null token.
            # - Classifier-free guidance label dropout is handled in the trainer via cfg_dropout_prob,
            #   so we disable any internal label dropout here to avoid double-dropping.
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.0)
        else:
            self.y_embedder = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
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
        
        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            
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
            # Clamp to include null class index (0) and max class index (num_classes)
            y = torch.clamp(y, 0, self.num_classes)
            y_emb = self.y_embedder(y, self.training)
            c = t_emb + y_emb
        else:
            c = t_emb
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x
