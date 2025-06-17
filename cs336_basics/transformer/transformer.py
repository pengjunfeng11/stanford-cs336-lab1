import torch
import sys

sys.path.append(
    "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/transformer/"
)
from Linear import Linear
from util import softmax
from rope import RotaryEmbedding
from RMSNorm import RMSNorm
from FFN import SwiGLU


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
    def __init__(
        self, d_model, n_heads, max_len=None, theta=None, token_positions=None
    ):
        super(CausalMultiHeadSelfAttention, self).__init__()
        self.token_positions = token_positions
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)  # 修复：输出维度应该是d_model
        self.attn = ScaledDotProductAttention()
        self.rope = RotaryEmbedding(theta, self.d_k, max_len)

    def forward(self, in_features: torch.Tensor, token_positions=None):
        batch_size, seq_len, d_model = in_features.shape

        # 投影Q、K、V
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)

        # 重塑为多头格式: [batch, seq_len, n_heads, d_k]
        Q = Q.contiguous().view(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.contiguous().view(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.contiguous().view(batch_size, seq_len, self.n_heads, self.d_k)

        # 在重塑后应用RoPE到Q和K（此时维度是正确的d_k）
        # 需要对每个头分别应用RoPE
        for head in range(self.n_heads):
            Q[:, :, head, :] = self.rope(Q[:, :, head, :], self.token_positions)
            K[:, :, head, :] = self.rope(K[:, :, head, :], self.token_positions)

        # 转置为: [batch, n_heads, seq_len, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 创建因果mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask  # 下三角为True
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.n_heads, -1, -1)

        # 计算注意力（只调用一次）
        attn_output = self.attn(Q, K, V, mask)

        # 转置回来并合并头: [batch, seq_len, n_heads, d_k]
        attn_output = attn_output.transpose(1, 2)

        # 重塑为: [batch, seq_len, d_model]
        # contiguous()确保张量在内存中是连续的，这样可以提高view操作的效率
        # 在某些情况下(如转置操作后)，张量在内存中可能不连续，contiguous()可以解决这个问题
        attn_output = attn_output.contiguous().view(batch_size, seq_len, d_model)

        # 输出投影
        output = self.o_proj(attn_output)

        return output


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        max_seq_len,
        theta,
    ):
        super(TransformerBlock, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.attn = CausalMultiHeadSelfAttention(d_model, n_heads, max_seq_len, theta)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, in_features):
        # 获取序列长度并创建位置编码
        seq_len = in_features.shape[-2]
        positions = torch.arange(seq_len, device=in_features.device)

        # 更新注意力机制的token_positions
        self.attn.token_positions = positions

        # Pre-LN Transformer结构：先LayerNorm，再注意力，最后残差连接
        ln1_output = self.ln1(in_features)
        attn_output = self.attn(ln1_output)
        # 第一个残差连接
        residual1 = in_features + attn_output

        # 第二个残差连接：先LayerNorm，再FFN，最后残差连接
        ln2_output = self.ln2(residual1)
        ffn_output = self.ffn(ln2_output)
        # 第二个残差连接
        return residual1 + ffn_output
