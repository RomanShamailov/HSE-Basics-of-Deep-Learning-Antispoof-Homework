import torch


class AbsLog:
    def __call__(self, x: torch.Tensor):
        return torch.log(x.abs() + 1e-8)
