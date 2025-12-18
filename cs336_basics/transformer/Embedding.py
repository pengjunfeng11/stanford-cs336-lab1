import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: vocab_size
        embedding_dim: d_model
        """
        super().__init__()
        self.d_model = embedding_dim
        self.vocab_size = num_embeddings
        # 创建一个空的参数矩阵,形状为(词表大小 x 词嵌入维度)
        self.weight = torch.nn.Parameter(
            torch.empty((self.vocab_size, self.d_model), device=device, dtype=dtype)
        )
        # 使用截断正态分布初始化参数矩阵
        self.weight = torch.nn.init.trunc_normal_(
            self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch_size, seq_len)
        """
        return self.weight[token_ids]
