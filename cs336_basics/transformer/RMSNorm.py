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
        x1 = x
        square = x**2
        mean = torch.mean(square, dim=-1, keepdim=True)
        rms = torch.sqrt(mean + self.eps)
        return x1 / rms * self.gain

        # 这两段代码因该完全等效，无非是效率的差别。但很奇怪，会有实际的误差。他们都能通过rsmNorm的测试，但在进行transformer block的测试时，下面的代码会不通过

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
