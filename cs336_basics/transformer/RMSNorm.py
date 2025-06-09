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

        #  x.shape (batch_size, sequence_length, d_model)
        # return a tensor of the same shape.
        # 计算 RMS 范数
        # 将计算过程抽出来便于调试
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

        # 计算均方根值 RMS(a)
        # rms = torch.sqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)

        # # 应用 RMSNorm 公式
        # normalized_x = x / rms

        # # 乘以可学习的增益参数 g_i
        # output = normalized_x * self.gain

        # return output


def test_rms_norm():
    # 创建一个RMSNorm实例
    d_model = 4
    rms_norm = RMSNorm(d_model)

    # 创建测试输入张量
    batch_size = 2
    seq_len = 3
    x = torch.tensor(
        [
            # batch 1
            torch.randn(512, 1024),  # 生成第一个batch的随机向量序列
            # batch 2
            torch.randn(512, 1024),  # 生成第二个batch的随机向量序列
        ]
    )

    # 执行RMSNorm
    output = rms_norm(x)

    # 验证输出形状是否正确
    assert output.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"输出形状错误: 期望 {(batch_size, seq_len, d_model)}, 得到 {output.shape}"

    print("RMSNorm测试通过！")


if __name__ == "__main__":
    test_rms_norm()
