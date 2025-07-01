import torch

# from Linear import Linear


def softmax(in_features, dim=-1):
    max_vals = torch.max(in_features, dim=dim, keepdim=True).values
    sum_tensor = torch.sum(torch.exp(in_features - max_vals), dim=dim, keepdim=True)
    # 使用torch的除法运算直接在tensor上进行操作，避免Python列表推导式
    return torch.exp(in_features - max_vals) / sum_tensor


# x * sigmoid(x)
def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.e ** (-x))


import numpy as np
import torch


def cross_entropy_loss(inputs, targets):
    from cs336_basics.transformer.util import softmax

    def logSoftmax(inputs: torch.Tensor):
        sm = softmax(inputs, dim=1)  # 确保 softmax 按行归一化
        return torch.log(sm)

    def MyNLLLoss(x: torch.Tensor, y: torch.Tensor):
        loss = []
        for n in range(len(y)):
            l_n = -x[n][y[n]]  # 提取第 n 个样本的真实标签对应的对数概率
            loss.append(l_n)
        return torch.mean(torch.stack(loss))  # 使用 PyTorch 的 mean 和 stack

    from torch.nn.functional import softmax, log_softmax

    # return loss
    return MyNLLLoss(log_softmax(inputs), targets)


def cross_entropy_gradient(y_pred, y_true):
    """
    计算交叉熵损失的梯度

    参数:
    y_pred: 预测概率分布 (N, C)
    y_true: 真实标签的one-hot编码 (N, C)

    返回:
    gradient: 交叉熵损失对预测值的梯度
    """
    N = y_true.shape[0]
    gradient = -y_true / (y_pred * N)
    return gradient
