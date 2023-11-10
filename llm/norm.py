import torch
from torch import nn
from llm.config import Config
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """
    From https://browse.arxiv.org/pdf/1910.07467.pdf

    output = input / rms(input) * weight
    """
    def __init__(self, config: Config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.norm_eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
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