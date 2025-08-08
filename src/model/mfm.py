import torch
from torch import nn


class MFM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        c = x.size(1)
        x1, x2 = torch.split(x, c // 2, dim=1)
        return torch.max(x1, x2)
