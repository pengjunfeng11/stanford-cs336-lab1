import torch
import torch.nn as nn

class BadRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loop implementation that does in-place modification
        def rms_norm_single(tensor):
            rms = torch.sqrt(torch.sum(tensor**2) / self.d_model + self.eps)
            normalized = tensor / rms
            scaled = normalized * self.gain
            return scaled

        # This modifies x in-place!
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j] = rms_norm_single(x[i, j])
        return x

def test_residual_connection():
    d_model = 4
    batch_size = 1
    seq_len = 1
    
    rmsnorm = BadRMSNorm(d_model)
    
    # Input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Simulate a residual connection: y = x + RMSNorm(x)
    # In a real Transformer, this is often: x = x + Attention(RMSNorm(x))
    # But checking if 'x' changes after RMSNorm(x) is enough to prove the point.
    
    print("Original x:", x)
    x_clone = x.clone() # Keep a copy of the original
    
    print("\nRunning BadRMSNorm(x)...")
    _ = rmsnorm(x)
    
    print("x after BadRMSNorm:", x)
    
    # Check if x has changed
    diff = torch.abs(x - x_clone).sum().item()
    if diff > 1e-6:
        print(f"\n[CRITICAL ISSUE CONFIRMED] Input tensor 'x' was modified in-place! Diff: {diff}")
        print("This corrupts the residual connection because the 'original x' is lost.")
    else:
        print("\nInput tensor 'x' was preserved.")

if __name__ == "__main__":
    test_residual_connection()
