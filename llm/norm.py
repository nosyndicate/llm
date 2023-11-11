import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """
    From https://browse.arxiv.org/pdf/1910.07467.pdf

    output = input / rms(input) * weight
    """
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.variance_epsilon = 1e-6

    def forward(self, input: torch.Tensor):
        input_dtype = input.dtype
        hidden_states = input.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)