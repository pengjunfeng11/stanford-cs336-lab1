import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(
            torch.ones(d_model)
        )  # 初始化为全 1 的向量, shape = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        example :

        input:
        """

        def rms_norm_single(tensor):
            # tensor.shape: (d_model,)
            rms = torch.sqrt(torch.sum(tensor**2) / self.d_model + self.eps)
            normalized = tensor / rms
            scaled = normalized * self.gain
            return scaled

        # 对每个序列中的每个向量进行归一化
        for i in range(x.shape[0]):  # batch维度
            for j in range(x.shape[1]):  # sequence维度
                x[i, j] = rms_norm_single(x[i, j])
        return x
