import math
from typing import Tuple, List, Dict, Sized, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings for Transformers.

    This module applies rotary positional embeddings to input tensors, allowing the model to utilize 
    continuous position information in a more flexible manner compared to traditional learned embeddings.

    :param d: Dimension of the embeddings. Should be even.
    :param base: Base used for calculating positional encodings (default: 10,000).
    """
    
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        if d % 2 != 0:
            raise ValueError("Dimension `d` for Rotary Positional Embedding must be even.")
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, max_pos: int, device: torch.device, dtype: torch.dtype):
        """Builds the cache for cosine and sine values."""
        if self.cos_cached is not None and max_pos <= self.cos_cached.shape[0]:
            return

        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(device)
        seq_idx = torch.arange(max_pos, device=device, dtype=dtype)
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor):
        """Rotates half of the embedding dimension."""
        x1, x2 = x.chunk(2, dim=-1)
        
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor=None):
        """
        Forward pass for the Rotary Positional Embeddings.

        This method applies the rotary positional embeddings to the input tensor.

        :param x: Input tensor of shape [batch, seq_len, d] 
        :param pos: Optional tensor of position indices, shape [batch, seq_len]. If None, indices are inferred as range(seq_len).
        :return: Tensor with applied rotary embeddings of the same shape as x.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        if pos is None:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        max_pos = pos.max().item() + 1
        
        self._build_cache(max_pos, device=x.device, dtype=x.dtype)
        
        x_rotated = self._rotate_half(x)
       
        x_rope = x * self.cos_cached[pos] + x_rotated * self.sin_cached[pos]

        return x_rope

class RopeEmbeddingXY(nn.Module):

    def __init__(self, emb_dim: int, max_patches_xy: int = 128, freezed=True):
        '''
        :param emb_dim: 
        :param max_patches_xy:
        :param freezed:
        :param separate: 
        '''
        super(RopeEmbeddingXY, self).__init__()
        assert emb_dim % 4 == 0, f'Embedding dimension must be divisible by 4'
        assert emb_dim > 4, f'Embedding dimension must be greater than 4'
   
        self.emb_dim = emb_dim
        # self.ax_dim = emb_dim // 2 if separate else emb_dim
        self.h_emb = RotaryPositionalEmbedding(emb_dim)
        self.w_emb = RotaryPositionalEmbedding(emb_dim)        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        :param x: input patches [B, D, H, W]
        :returns: output sequence [B, L, D]
        '''
        batch_size, emb_dim, patch_h, patch_w = x.shape
        device = x.device
        
        h_pos, w_pos = torch.meshgrid(
            torch.arange(0, patch_h), torch.arange(0, patch_w), indexing='ij',
        )

        x = x.flatten(-2).swapaxes(1, 2)
        h_pos =  h_pos.flatten().unsqueeze(0)
        w_pos =  w_pos.flatten().unsqueeze(0)

        x = self.h_emb(x, h_pos)
        x = self.w_emb(x, w_pos)
            
        return x 
