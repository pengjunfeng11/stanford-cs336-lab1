import torch
import sys

sys.path.append(
    "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/transformer/"
)
from Linear import Linear
from util import softmax


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    # Q.shape = torch.Size([4, 12, 64])
    # K.shape = torch.Size([4, 16, 64])
    # V.shape = torch.Size([4, 16, 64])
    # mask.shape = torch.Size([4, 12, 16])
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask):
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask
        attn_output = self.Q @ self.K.transpose(-2, -1)

        if mask is not None:
            attn_output = attn_output.masked_fill(~mask, float("-inf"))

        attn_output = attn_output / torch.sqrt(
            torch.tensor(self.K.size(-1), dtype=torch.float32)
        )
        attn_output = softmax(attn_output, dim=-1)
        attn_output = attn_output @ self.V

        return attn_output


class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        # dk = dv = d_model/n_heads
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        d_k = d_v = d_model // n_heads
        # As a stretch goal, try combining the key, query, and value projections into a single weight matrix so you only need a single matrix multiply.
        self.q_proj = Linear(d_model, d_k * n_heads)
        self.k_proj = Linear(d_model, d_k * n_heads)
        self.v_proj = Linear(d_model, d_v * n_heads)
        self.o_proj = Linear(d_model, d_v)
        self.attn = ScaledDotProductAttention()

    def forward(
        self,
        in_features: torch.Tensor,
    ):
        """
        Given the key, query, and value projection weights of a naive unbatched
        implementation of multi-head attention, return the output of an optimized batched
        implementation. This implementation should handle the key, query, and value projections
        for all heads in a single matrix multiply.
        This function should not use RoPE.
        See section 3.2.2 of Vaswani et al., 2017.

        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
            q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
            k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
            v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
            o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
            in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
            implementation with the given QKV projection weights and input features.
        """
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        # 获取序列长度
        seq_len = in_features.size(1)

        # 创建下三角矩阵作为mask
        # 使用torch.triu创建上三角矩阵，然后取反得到下三角矩阵
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask

        # 扩展mask维度以匹配注意力的形状
        mask = mask.unsqueeze(0)  # 添加batch维度
        mask = mask.expand(in_features.size(0), -1, -1)  # 扩展到所有batch
        self.attn(Q, K, V, mask)
        O = self.o_proj(self.attn(Q, K, V, mask))
        return O

        pass
