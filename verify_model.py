import torch
from cs336_basics.transformer.transformer import TransformerLM

def verify_transformer_lm():
    # 1. 定义超参数 (小规模以便快速测试)
    vocab_size = 1000
    context_length = 32
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128
    rope_theta = 10000.0
    
    print("Initializing TransformerLM...")
    # 2. 实例化模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    
    print("Model initialized successfully.")
    
    # 3. 创建假数据 (Batch Size=2, Sequence Length=16)
    batch_size = 2
    seq_len = 16
    in_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {in_indices.shape}")
    
    # 4. 前向传播
    print("Running forward pass...")
    try:
        logits = model(in_indices)
        print("Forward pass successful!")
        print(f"Output logits shape: {logits.shape}")
        
        # 5. 验证输出形状
        expected_shape = (batch_size, seq_len, vocab_size)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, but got {logits.shape}"
        print("Shape verification passed.")
        
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_transformer_lm()
