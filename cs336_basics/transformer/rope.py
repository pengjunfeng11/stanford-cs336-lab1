from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, einsum
from jaxtyping import Float, Int


class RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device="cuda"):
        theta: float  # ΘvaluefortheRoPE
        d_k: int  # dimensionofqueryandkeyvectors
        max_seq_len: int  # Maximumsequencelengththatwillbeinputted
        device: torch.device | None = None  #  Devicetostorethebufferon
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        self.freqs_cis = self.precompute_freqs_cis(d_k, max_seq_len, theta).to(device)

    def precompute_freqs_cis(self, dim: int, sql_len: int, theta: float = 10000.0):
        """
        Precompute frequency tensors for RoPE.
        Args:
            dim (int): Feature dimension (d_k).
            sql_len (int): Maximum sequence length.
            theta (float): Base for frequency computation.
        Returns:
            freqs_cis: Complex tensor of shape (sql_len, dim // 2).
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(sql_len, device=freqs.device)  # (seq_len)
        freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, dim // 2)
        return freqs_cis

    def forward(
        self,
        x: torch.Tensor,  # batch_size, seq_len, dim
        token_positions: Int[torch.Tensor, " ... sequence_length"] = -1,
    ) -> torch.Tensor:
        """Process an input tensor of shape (..., seq_len, d_k) and return a tensor
        of the same shape.
        Note that you should tolerate x with an arbitrary number
        of batch dimensions. You should assume that the token positions are a tensor
        of shape (..., seq_len) specifying the token positions of x along the sequence
        dimension. You should use the token positions to slice your (possibly precomputed)
        cos and sin tensors along the sequence dimension."""
        
        # 使用预计算的freqs_cis而不是重新计算
        freqs_cis = self.freqs_cis
    
        # 将token_positions展平成一维张量,方便后续索引操作
        flat_positions = token_positions.reshape(-1)
    
        # 确保索引不超出范围
        max_pos = torch.max(flat_positions)
        if max_pos >= freqs_cis.shape[0]:
            raise ValueError(f"Token position {max_pos} exceeds maximum sequence length {freqs_cis.shape[0]-1}")
    
        # 根据展平后的位置索引从预计算的freqs_cis中获取对应位置的旋转编码
        freqs_cis_pos = freqs_cis[flat_positions]
    
        # 将获取到的旋转编码重新调整为与token_positions相同的形状,并在最后添加一个维度用于存储编码值
        freqs_cis_pos = freqs_cis_pos.reshape(*token_positions.shape, -1)
        
        # 将最后一维度拆分为二维向量
        x_re, x_im = x[..., ::2], x[..., 1::2]
    
        # 将 freqs_cis 分为实部和虚部
        freqs_cos = torch.cos(torch.angle(freqs_cis_pos))
        freqs_sin = torch.sin(torch.angle(freqs_cis_pos))
    
        x_out = torch.zeros_like(x)
        # 对偶数位置的元素进行旋转变换: x_re * cos(θ) - x_im * sin(θ)
        x_out[..., ::2] = x_re * freqs_cos - x_im * freqs_sin
        # 对奇数位置的元素进行旋转变换: x_re * sin(θ) + x_im * cos(θ)
        x_out[..., 1::2] = x_re * freqs_sin + x_im * freqs_cos
    
        return x_out
