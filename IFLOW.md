# CS336 Assignment 1: 基础项目指南

## 项目概述

这是斯坦福大学CS336课程（2025年春季）的第一个作业项目，目标是**从头开始构建一个完整的Transformer语言模型**。项目涵盖了从分词器到训练循环的所有组件实现。

### 主要组件

1. **BPE分词器** - 字节对编码分词器实现
2. **Transformer模型** - 包含所有核心组件的完整Transformer架构
3. **训练框架** - 包括优化器、学习率调度、梯度裁剪等
4. **评估与生成** - 文本生成和模型评估工具

## 技术栈

- **语言**: Python 3.11+
- **深度学习框架**: PyTorch 2.7.1+
- **环境管理**: uv (推荐) 或 pip
- **测试框架**: pytest
- **代码质量**: ruff (linting)
- **类型检查**: jaxtyping

### 核心依赖

```toml
einops>=0.8.1          # 张量操作
torch>=2.7.1           # PyTorch
numpy                  # 数值计算
pytest>=8.3.4          # 测试框架
regex>=2024.11.6       # 正则表达式（用于分词）
tiktoken>=0.9.0        # 分词工具
wandb>=0.19.7          # 实验跟踪
```

## 项目结构

```
assignment1-basics/
├── cs336_basics/                 # 主要代码目录
│   ├── __init__.py
│   ├── bpe.py                   # BPE分词器实现
│   ├── pretokenization_example.py # 分词示例
│   ├── transformer/               # Transformer组件
│   │   ├── __init__.py
│   │   ├── Embedding.py         # 嵌入层
│   │   ├── Linear.py            # 线性层
│   │   ├── RMSNorm.py           # RMS归一化
│   │   ├── FFN.py               # 前馈网络
│   │   ├── rope.py              # RoPE位置编码
│   │   ├── transformer.py       # 主Transformer实现
│   │   └── util.py              # 工具函数
│   └── util/
│       ├── __init__.py
│       └── cross_entropy.py     # 交叉熵损失
├── tests/                       # 测试文件
│   ├── adapters.py              # 测试适配器（需要实现）
│   ├── test_*.py               # 各个组件的测试
│   └── fixtures/               # 测试数据
├── data/                       # 数据集目录（需下载）
├── pyproject.toml              # 项目配置
├── uv.lock                     # 依赖锁定
├── README.md                   # 作业说明
└── cs336_spring2025_assignment1_basics.pdf  # 详细作业说明
```

## 环境设置

### 使用 uv（推荐）

```bash
# 安装 uv
pip install uv

# 运行任何Python文件（自动管理环境）
uv run <python_file>

# 运行测试
uv run pytest

# 安装依赖
uv sync
```

### 使用 pip

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -e .
```

## 开发工作流

### 1. 实现适配器函数

所有测试通过 `tests/adapters.py` 中的适配器函数连接到你的实现。这是**胶水代码**，不应包含实质性逻辑。

**重要**: 先实现适配器，再运行对应测试。

示例适配器模式：
```python
def run_linear(d_in, d_out, weights, in_features):
    from cs336_basics.transformer.Linear import Linear
    
    linear = Linear(d_in, d_out)
    linear.load_state_dict({'weight': weights})
    return linear(in_features)
```

### 2. 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest -k test_linear
uv run pytest tests/test_model.py

# 带覆盖率运行
uv run pytest --cov=cs336_basics
```

### 3. 代码质量检查

```bash
# 运行 linting
uv run ruff check .

# 自动修复
uv run ruff check --fix .

# 格式化代码
uv run ruff format .
```

## 核心实现任务

### 第一阶段：基础组件

1. **Linear层** (`cs336_basics/transformer/Linear.py`)
   - 实现无偏置的线性变换
   - 使用截断正态分布初始化

2. **Embedding层** (`cs336_basics/transformer/Embedding.py`)
   - 词嵌入查找
   - 形状: (vocab_size, d_model)

3. **RMSNorm** (`cs336_basics/transformer/RMSNorm.py`)
   - 均方根归一化
   - 公式: `x_i / RMS(x) * gain_i`

### 第二阶段：注意力机制

4. **RoPE** (`cs336_basics/transformer/rope.py`)
   - 旋转位置编码
   - 支持任意批处理维度

5. **Scaled Dot-Product Attention**
   - 缩放点积注意力
   - 支持掩码

6. **Multi-Head Self-Attention**
   - 因果掩码
   - 集成RoPE

### 第三阶段：Transformer块

7. **SwiGLU FFN** (`cs336_basics/transformer/FFN.py`)
   - SiLU激活 + GLU门控
   - d_ff ≈ 8/3 * d_model

8. **Transformer Block**
   - Pre-norm架构
   - 残差连接

9. **完整Transformer LM**
   - 嵌入层 + N个Transformer块 + 输出层

### 第四阶段：训练框架

10. **损失函数** (`cs336_basics/util/cross_entropy.py`)
    - 交叉熵损失
    - 数值稳定性处理

11. **AdamW优化器**
    - 状态ful优化器
    - 权重衰减

12. **学习率调度**
    - 余弦退火 + 预热

13. **梯度裁剪**
    - L2范数裁剪

14. **数据加载**
    - 内存高效的批处理
    - 支持大文件（mmap）

15. **Checkpointing**
    - 模型和优化器状态保存/加载

### 第五阶段：BPE分词器

16. **BPE训练** (`cs336_basics/bpe.py`)
    - 字节级BPE
    - 并行化预分词
    - 特殊token处理

17. **Tokenizer类**
    - 编码/解码
    - 支持迭代器接口

### 第六阶段：训练脚本

18. **训练循环**
    - 完整训练流程
    - 验证和日志记录
    - 超参数配置

19. **文本生成**
    - 解码（贪婪/采样）
    - 温度缩放
    - Top-p采样

## 数据集

### TinyStories
- **训练**: 2.1M文档
- **验证**: 22K文档
- **用途**: 快速实验和调试

### OpenWebText
- **训练**: 约8GB文本
- **验证**: 约0.4GB文本  
- **用途**: 大规模训练

### 下载数据

```bash
mkdir -p data
cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## 实验配置

### 基础模型配置（TinyStories）

```python
{
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1344,  # ~8/3 * d_model, 64的倍数
    "rope_theta": 10000.0,
    "total_tokens": 327_680_000
}
```

### 训练超参数

```python
{
    "batch_size": 32,  # 根据GPU内存调整
    "learning_rate": 3e-4,  # 需要调优
    "warmup_steps": 1000,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "max_grad_norm": 1.0
}
```

## 性能优化技巧

### 1. 内存效率
- 使用 `np.memmap` 加载大数据集
- 在GPU上预分配张量
- 使用梯度检查点（如果需要）

### 2. 计算效率
- 批处理所有头矩阵乘法
- 使用 `torch.compile()` (PyTorch 2.0+)
- 在A100上使用TF32精度

### 3. 调试技巧
- 先过拟合单个小批次
- 监控梯度/激活范数
- 使用VSCode/PyCharm调试器

### 4. 开发策略
- **从小开始**: 先用TinyStories验证实现
- **逐步扩展**: 成功后再用OpenWebText
- **频繁测试**: 每实现一个组件就运行测试
- **记录实验**: 使用wandb跟踪超参数和指标

## 常见命令

```bash
# 运行测试
uv run pytest -v

# 运行特定测试文件
uv run pytest tests/test_model.py -v

# 运行带关键字的测试
uv run pytest -k "test_linear or test_embedding"

# 训练BPE分词器
uv run python cs336_basics/bpe.py

# 训练Transformer
uv run python train.py

# 生成文本
uv run python generate.py

# 创建提交包
./make_submission.sh
```

## 实现检查清单

### 必需组件
- [ ] Linear层实现
- [ ] Embedding层实现  
- [ ] RMSNorm实现
- [ ] RoPE实现
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Self-Attention
- [ ] SwiGLU FFN
- [ ] Transformer Block
- [ ] Transformer LM
- [ ] Cross-Entropy Loss
- [ ] AdamW Optimizer
- [ ] Learning Rate Schedule
- [ ] Gradient Clipping
- [ ] Data Loading
- [ ] Checkpointing
- [ ] BPE Tokenizer Training
- [ ] Tokenizer Encode/Decode
- [ ] Training Script
- [ ] Text Generation
- [ ] 所有测试通过

### 实验
- [ ] 在TinyStories上训练基础模型
- [ ] 学习率调优实验
- [ ] Batch size实验
- [ ] 生成样本文本
- [ ] RMSNorm消融实验
- [ ] Pre/Post-Norm比较
- [ ] NoPE vs RoPE实验
- [ ] SwiGLU vs SiLU实验

## 资源需求

### TinyStories训练
- **时间**: ~30-40分钟（1x H100 GPU）
- **内存**: < 30GB
- **令牌处理**: 327M tokens

### OpenWebText训练
- **时间**: ~12小时（BPE训练）
- **内存**: < 100GB
- **数据**: 825GB文本

### 低资源选项（CPU/Apple Silicon）
- **令牌处理**: 40M tokens（减少10倍）
- **时间**: 1-2小时（M3 Max）
- **验证损失目标**: 2.00（替代1.45）

## 参考资源

- **作业PDF**: `cs336_spring2025_assignment1_basics.pdf`
- **GitHub仓库**: https://github.com/stanford-cs336/assignment1-basics
- **Leaderboard**: https://github.com/stanford-cs336/assignment1-basics-leaderboard
- **uv文档**: https://docs.astral.sh/uv/
- **einops文档**: https://einops.rocks/

## 常见问题

### 测试失败
1. 检查适配器实现是否正确
2. 确认张量形状匹配
3. 验证初始化参数
4. 检查设备（CPU/GPU）一致性

### 训练不稳定
1. 降低学习率
2. 增加warmup步骤
3. 检查梯度裁剪
4. 验证RMSNorm实现

### 内存不足
1. 减小batch size
2. 使用梯度累积
3. 使用内存映射加载数据
4. 减少模型大小

---

*最后更新: 2025-12-15*
*项目版本: 1.0.3*
