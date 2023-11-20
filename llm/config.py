from dataclasses import dataclass
from typing import Any, Literal, Mapping, TypeAlias

NormOption: TypeAlias = Literal["layernorm", "rmsnorm"]
NormPlacementOption: TypeAlias = Literal["prenorm", "postnorm", "gpt2norm", "deepnorm"]
ActivationOption: TypeAlias = Literal["swish", "relu"]


@dataclass
class ModelConfig:
    vocab_size: int
    # Dimension of the embedding
    dim: int
    n_layer: int
    n_head: int
    attention_bias: bool

    ffn: str
    # activation function for the ffn
    # valid choice are
    ffn_act: ActivationOption

    norm: NormOption
    norm_placement: NormPlacementOption
    norm_eps: float = 1e-6


@dataclass
class WandBConfig:
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "llama2"


@dataclass
class TrainingConfig:
    seed: int = 1337
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 5
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = (
        6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    )
    backend: str = "nccl"
    device: str = "cpu"
    dtype: str = "bfloat16"
    compile: bool = True


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash_attention: bool = False
    rms_norm: bool = False


@dataclass
class Llama2Config:
    max_seq_len: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768  # TODD: should I rename to n_dim or embd_dim?
    ffn_hidden: int | None = None
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash_attention: bool = False
