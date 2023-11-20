import math
import os
from dataclasses import asdict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.config import GPTConfig, Llama2Config
from llm.llama2 import Llama2
from llm.model import GPT


def get_lr(
    it: int, warmup_iters: int, learning_rate: float, lr_decay_iters: int, min_lr: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


class DataLoader:
    "Poor"

    def __init__(
        self,
        dataset: str,
        batch_size: int,
        block_size: int,
        device_type: str,
        device: str,
    ) -> None:
        data_dir = os.path.join("data", dataset)
        self.train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )
        self.val_data = np.memmap(
            os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r"
        )
        self.batch_size = batch_size
        self.block_size = block_size
        self.device_type = device_type
        self.device = device

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(
                self.device, non_blocking=True
            )
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


def get_model(
    n_layer,
    n_head,
    n_embd,
    block_size,
    bias,
    dropout,
    flash_attention,
    rms_norm,
    init_from,
    meta_vocab_size,
    out_dir,
    device,
    device_type,
    dtype,
    compile,
    ddp,
    ddp_local_rank,
    weight_decay,
    learning_rate,
    beta1,
    beta2,
):
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
        flash_attention=flash_attention,
        rms_norm=rms_norm,
    )  # start with model_args from command line

    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
        model_args["vocab_size"] = (
            meta_vocab_size if meta_vocab_size is not None else 50304
        )
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args[
            "block_size"
        ] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model, requires PyTorch 2.0
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # type: ignore[arg-type, assignment]

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model, model_args, optimizer, scaler


def get_llama2(
    n_layer: int,
    n_head: int,
    n_embd: int,
    max_seq_len: int,
    bias: bool,
    dropout: float,
    flash_attention: bool,
    init_from,
    meta_vocab_size,
    out_dir,
    device,
    device_type,
    dtype,
    compile,
    ddp,
    ddp_local_rank,
    weight_decay,
    learning_rate,
    beta1,
    beta2,
):
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print(
                "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
            )
            meta_vocab_size = 50304
        llama2_config = Llama2Config(
            max_seq_len=max_seq_len,
            vocab_size=meta_vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            ffn_hidden=None,
            dropout=dropout,
        )
        model_args = asdict(llama2_config)

        model = Llama2(llama2_config)
        print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))
    elif init_from == "resume":
        raise NotImplementedError()

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if init_from == "resume":
        pass
    checkpoint = None  # free up memory

    # compile the model, requires PyTorch 2.0
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # type: ignore[arg-type, assignment]

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    return model, model_args, optimizer, scaler
