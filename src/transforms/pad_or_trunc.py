import torch
from torch import nn


class PadOrTrunc:
    def __call__(self, x: torch.Tensor):
        width = x.shape[-1]
        if width > 600:
            x = x[:, :, :600]
        elif width < 600:
            x = nn.functional.pad(x, (0, 600 - width), "constant", 0)
        return x
