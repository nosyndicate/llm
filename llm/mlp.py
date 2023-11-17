from torch import Tensor, nn

from llm.config import Llama2Config


class SiLUActivation(nn.Module):
    """
    See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
    Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
    Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
    Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
    later.
    """

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)


class GLU(nn.Module):
    """
    Implemented the Gated Linear Unit.

    https://arxiv.org/pdf/2302.13971.pdf

    FFN(x, W, V, W_2) = (Activation(xW) * xV) * W_2

    The default value for intermediate_size is 2/3 * 4 * n_embd
    - 2/3 come from the original llama paper. 
        - because we have 3 layers instead of 2 in original FFN, so we should scale it by 2/3
    - 4 comes from the old practice of original attention is all you need paper.
    - n_embd is the embedding size.
    """
    def __init__(self, config: Llama2Config) -> None:
        super().__init__()
 
        if config.ffn_hidden is None:
            ffn_hidden = int(2 * 4 * config.n_embd / 3)
        else:
            ffn_hidden = config.ffn_hidden

        self.gate_proj = nn.Linear(config.n_embd, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, config.n_embd, bias=False)
        self.up_proj = nn.Linear(config.n_embd, ffn_hidden, bias=False)
        self.act_fn = SiLUActivation()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
