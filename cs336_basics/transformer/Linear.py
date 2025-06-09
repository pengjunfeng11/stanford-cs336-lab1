import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        # 在PyTorch中，线性层的权重矩阵维度为(out_features, in_features)是为了计算效率
        # 1. 这样在前向传播时可以直接使用矩阵乘法，不需要转置
        # 2. 符合PyTorch的内存布局优化
        # 3. 便于并行计算
        self.weight = torch.empty(
            (out_features, in_features), device=device, dtype=dtype
        )
        std_linear = math.sqrt(2 / (in_features + out_features))
        self.weight = torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std_linear,
            a=-3.0 * std_linear,
            b=3.0 * std_linear,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch 中的线性层的权重矩阵维度为(out_features, in_features)
        # 因此，在进行矩阵乘法时，需要将输入张量的维度调整为(in_features, out_features)
        # 这样才能与输入x进行乘法运算
        return x @ self.weight.T  # OK
