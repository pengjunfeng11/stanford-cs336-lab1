import torch
import torch.nn as nn
import sys

sys.path.extend(
    [
    "/mnt/d/WorkSpace/cs336/lab1/assignment1-basics/cs336_basics/transformer/",
    '/Users/berrypeng/Desktop/workSpace/berry_workSpace/Python/GitHub/stanford-cs336-lab1/cs336_basics/transformer/'
    ]
)

from util import SiLU



class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, dtype=torch.float32):
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    # follow formula 7
    def forward(self, x):
        SiLU_out = SiLU(self.w1(x))
        w3_x = self.w3(x)
        return self.w2(SiLU_out * w3_x)
