import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """
    From https://browse.arxiv.org/pdf/1910.07467.pdf

    output = x / rms(x) * weight
    """

    def __init__(self, n_embd: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.variance_epsilon = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
