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
