import torch
from Linear import Linear


def softmax(in_features, dim=-1):
    max_vals = torch.max(in_features, dim=dim, keepdim=True).values
    sum_tensor = torch.sum(torch.exp(in_features - max_vals), dim=dim, keepdim=True)
    # 使用torch的除法运算直接在tensor上进行操作，避免Python列表推导式
    return torch.exp(in_features - max_vals) / sum_tensor


# x * sigmoid(x)
def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.e ** (-x))
