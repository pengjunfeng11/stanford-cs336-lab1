import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.weight = torch.nn.init.trunc_normal_(self.weight)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return x.multiply(self.weight)
