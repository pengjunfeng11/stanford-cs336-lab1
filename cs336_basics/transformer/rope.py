import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotaryEmbedding(nn.Module):
    def __init__(self,theta: float,d_k:int,max_seq_len:int,device=None):
        theta:float # ΘvaluefortheRoPE
        d_k:int # dimensionofqueryandkeyvectors
        max_seq_len: int # Maximumsequencelengththatwillbeinputted
        device: torch.device | None=None #  Devicetostorethebufferon

    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        """
        Precompute frequency tensors for RoPE.
        Args:
            dim (int): Feature dimension (d_k).
            end (int): Maximum sequence length.
            theta (float): Base for frequency computation.
        Returns:
            freqs_cis: Complex tensor of shape (end, dim // 2).
        """
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # (seq_len)
        freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, dim // 2)
        return freqs_cis

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)-> torch.Tensor:
        '''Process an input tensor of shape (..., seq_len, d_k) and return a tensor
        of the same shape. 
        Note that you should tolerate x with an arbitrary number 
        of batch dimensions. You should assume that the token positions are a tensor 
        of shape (..., seq_len) specifying the token positions of x along the sequence 
        dimension. You should use the token positions to slice your (possibly precomputed) 
        cos and sin tensors along the sequence dimension.'''
        pass

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute frequency tensors for RoPE.
    Args:
        dim (int): Feature dimension (d_k).
        end (int): Maximum sequence length.
        theta (float): Base for frequency computation.
    Returns:
        freqs_cis: Complex tensor of shape (end, dim // 2).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # (seq_len)
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, dim // 2)
    return freqs_cis

precompute_freqs_cis(1024, 128)