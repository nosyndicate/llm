import torch
from torch import nn


# class RotaryEmbedding(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         max_position_embeddings: int = 2048,
#         base: int = 10000,
#         device=None,
#     ):
#         super().__init__()
#         inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
#         self.register_buffer("inv_freq", inv_freq)

#         # Build here to make `torch.jit.trace` work.
#         self.max_seq_len_cached = max_position_embeddings
#         t = torch.arange(
#             self.max_seq_len_cached,
#             device=self.inv_freq.device,
#             dtype=self.inv_freq.dtype,
#         )
#         freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#         # Different from paper, but it uses a different permutation in order to obtain the same calculation
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer(
#             "cos_cached", emb.cos()[None, None, :, :], persistent=False
#         )
#         self.register_buffer(
#             "sin_cached", emb.sin()[None, None, :, :], persistent=False
#         )

#     def forward(self, x: torch.Tensor, seq_len: int | None = None):
#         """
#         x should of size  [batch_size, num_attention_heads, seq_len, head_size]
#         """
#         # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
#         if seq_len > self.max_seq_len_cached:
#             self.max_seq_len_cached = seq_len
#             t = torch.arange(
#                 self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype
#             )
#             freqs = torch.einsum("i,j->ij", t, self.inv_freq)
#             # Different from paper, but it uses a different permutation in order to obtain the same calculation
#             emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
#             self.register_buffer(
#                 "cos_cached", emb.cos()[None, None, :, :], persistent=False
#             )
#             self.register_buffer(
#                 "sin_cached", emb.sin()[None, None, :, :], persistent=False
#             )
#         return (
#             self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#             self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
#         )


# def rotate_half(x):
#     """
#     Rotates half the hidden dims of the input.

#     Transform a embedding vector of
#         (x_1, x_2, x_3, ..., x_{d-1}, x_d)
#     into
#         (-x_2, x_1, -x_4, ..., -x_d, x_{d-1})
#     """
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
#     """
#     Apply position embedding on query and key embeddings
#     Equation (34) of https://arxiv.org/pdf/2104.09864.pdf
#     """
#     # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
#     cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
#     sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
#     cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

RopeEmbedding = tuple[torch.Tensor, torch.Tensor]


def build_rope_cache(
    base: int, d: int, max_seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the rotary embedding.

    From Section 3.2.2
    \theta_i = 1 / (base ^ {\frac{2(i - 1)}{d}}) for i in [1, 2, ..., \frac{d}{2}]


    Args:
        base (int): The base used to compute theta
        d (int): The number of dimensions of a token's embedding
        seq_len (int): Max sequence length, we will create the position embedding for each position

    Returns:
        return_type: Description of the return value.

    """
    # Compute theta, this theta is universal for all positions
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d)).to(device)

    # Create position indexes `[0, 1, ..., max_seq_len - 1]`
    seq_idx = torch.arange(max_seq_len, device=device)

    # Compute m\theta_i in the original paper, this determines the angles of rotation for each position
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
        tok_embedding (torch.Tensor): Query projection or key project of the tokens, of shape [B, n_head, T, head_dim]
        cos (torch.Tensor): (T, head_dim)
        sin (torch.Tensor): (T, head_dim)
    """

    roped = (tok_embedding * cos) + (rotate_half(tok_embedding) * sin)
    return roped.type_as(tok_embedding)
