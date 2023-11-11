import math

import torch
from flash_attn import flash_attn_func  # type:ignore[import-untyped]
from torch import LongTensor, Tensor, nn
from torch.nn import functional as F

from llm.rotary_embedding import RotaryEmbedding, apply_rotary_pos_emb


class Attention(nn.Module):
    """
    Attention layer, using ROPE position embedding.

    It supoort three type of attentions:
    - if num_key_value_heads == num_heads, Mutlihead attention (https://arxiv.org/pdf/1706.03762.pdf)
    - if num_key_value_heads == 1, MultiQuery attention (https://arxiv.org/pdf/1911.02150v1.pdf)
    - otherwise, Grouped Query Attention (https://arxiv.org/pdf/2305.13245.pdf)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        attention_bias: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = num_key_value_heads

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention_bias = attention_bias
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: LongTensor | None = None,
    ):
        batch_size, q_len, _ = hidden_states.size()

        # Project into multiple QKV subspaces
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Split into multiple heads if necessary
        query_states = query_states.view(
            batch_size, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        # Apply Rotary embedding
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash_attention = config.flash_attention
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        q = q.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)
        v = v.view(B, T, self.n_head, C // self.n_head)  # (B, T, nh, hs)

        # first try flash attention
        if self.flash_attention:
            scale = 1.0 / math.sqrt(k.size(-1))
            y = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0,
                softmax_scale=scale,
                causal=True,
            )
            y = y.contiguous().view(B, T, C)
        else:
            # if not using flash attention, use original implementation from nanogpt
            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            q = q.transpose(1, 2)  # (B, nh, T, hs)
            k = k.transpose(1, 2)  # (B, nh, T, hs)
            v = v.transpose(1, 2)  # (B, nh, T, hs)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout if self.training else 0,
                    is_causal=True,
                )
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
