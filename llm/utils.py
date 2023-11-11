import math
import os
import torch
import numpy as np


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
