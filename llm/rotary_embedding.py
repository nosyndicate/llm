import torch
from torch import nn


def build_rope(base: int, d: int, max_seq_len: int, ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the rotary embedding.

    From Section 3.2.2
    \theta_i = 1 / (base ^ {\frac{2(i - 1)}{d}}) for i in [1, 2, ..., \frac{d}{2}]


    Args:
        base: The base used to compute theta
        d : The number of dimensions of a token's embedding
        max_seq_len: Max sequence length, we will create the position embedding for each position

    Returns:
        cos: Cosine part of the position embedding 
        sin: Sine part of the position embedding
    """
    # Compute theta, this theta is universal for all positions
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    seq_idx = torch.arange(max_seq_len)

    # Compute m\theta_i in the original paper, this determines the angles of rotation for each position
    # TODO idx_theta and idx_theta can use a better name
    idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
    idx_theta2 = torch.cat((idx_theta, idx_theta), dim=-1)

    cos = idx_theta2.cos()  # (max_seq_len, d)
    sin = idx_theta2.sin()  # (max_seq_len, d)

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dims of the input.

    Transform a embedding vector of
        (x_1, x_2, x_3, ..., x_{d-1}, x_d)
    into
        (-x_{d/2 + 1}, -x_{d/2 + 2}, ..., -x_{d}, x_{1}, ..., x_{d/2})

    Note, this rotation is different from original paper.

    TODO, write more about this difference according to here: https://github.com/jzhang38/TinyLlama/blob/e77d2d0eb1ef8c5f368027a40cbe005faece559e/lit_gpt/fused_rotary_embedding.py#L16
    The gist which dimensions we pick to form the pair
    origianl paper pick i and i + 1 where i = 0, 2, 4, ...
    a different style picked i and i + d/2 where i = 1, 2, 3, ...
    """
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]  # (B, n_head, T, head_dim/2)
    x2 = x[..., head_dim // 2 :]  # (B, n_head, T, head_dim/2)
    return torch.cat((-x2, x1), dim=-1)  # (B, n_head, T, head_dim)


def apply_rotary_embedding(
    tok_embedding: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply the Rotary Embedding to the query or key of the tokens.

    Args:
        tok_embedding: Query projection or key project of the tokens, of shape [B, n_head, T, head_dim]
        cos: (T, head_dim)
        sin: (T, head_dim)
    """
    # Apply the position embedding on the last two dimensions
    roped = (tok_embedding * cos) + (rotate_half(tok_embedding) * sin)
    return roped.type_as(tok_embedding)
