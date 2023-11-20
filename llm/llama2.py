import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F

from llm.attention import Attention
from llm.config import Llama2Config
from llm.mlp import GLU
from llm.norm import RMSNorm
from llm.rotary_embedding import build_rope


class Block(nn.Module):
    def __init__(self, config: Llama2Config) -> None:
        super().__init__()

        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)
        self.attention = Attention(config)
        self.ffn = GLU(config)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: token embedding, (B, T, n_embd)
            rope_cos: cos part of the rope embedding, (T, head_dim)
            rope_sin: sin part of the rope embedding, (T, head_dim)
        """

        # (B, T, n_embd) + (B, T, n_embd) -> (B, T, n_embd)
        x = x + self.attention(self.ln1(x), rope_cos, rope_sin)

        # (B, T, n_embd) + (B, T, n_embd) -> (B, T, n_embd)
        x = x + self.ffn(self.ln2(x))

        return x


class Llama2(nn.Module):
    def __init__(self, config: Llama2Config) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len

        self.n_layer = config.n_layer
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.norm = RMSNorm(config.n_embd)
        self.n_embd = config.n_embd
        self.n_head = config.n_head

        cos, sin = build_rope(
            base=10000, d=(self.n_embd // self.n_head), max_seq_len=self.max_seq_len
        )
        # Register buffer there so rotary embedding can move along with
        # other parameters to different devices with `.to(device)`
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layer):
            self.layers.append(Block(config))

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        tokens: The input tokens, of shape [B, T]
        target: The prediction target of each token [B, T]

        target should be a shift of tokens.
        """
        # TODO Need to validate the comment here to make sure the size matches
        device = tokens.device
        b, t = tokens.shape
        assert t <= self.max_seq_len, "Sequence is too long"

        # Create the position for each token, do
        position_idx = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.embedding(tokens)  # (B, T, n_embd)

        # Only take the necessary length of cached embedding
        rope_cos = self.rope_cos[:t]  # (T, head_dim)
        rope_sin = self.rope_sin[:t]  # (T, head_dim)

        for block in self.layers:
            x = block(x, rope_cos, rope_sin)  # (B, T, n_embd)

        # Llama has this normalization before lm_head
        # But gpt2 doesn't, when it become a convention?
        # TODO Need to investigate
        x = self.norm(x)  # (B, T, n_embd)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # Flatten the array to (B * T, vocab_size) then compute cross entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # TODO I will worry about the inference later
            raise NotImplementedError()

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        TODO Following functions borrowed from nanogpt, still need to rewrite or refactor
        """
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # N = self.get_num_params()
        # cfg = self.config
        # L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        # flops_per_token = 6 * N + 12 * L * H * Q * T
        # flops_per_fwdbwd = flops_per_token * T
        # flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # # express our flops throughput as ratio of A100 bfloat16 peak flops
        # flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # mfu = flops_achieved / flops_promised
        # return mfu
        return 0.1
