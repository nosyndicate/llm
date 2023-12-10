from typing import Any, Iterator, Sized

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path: str, max_seq_len: int) -> None:
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.max_seq_len = max_seq_len

        # len(self.data) return number of tokens
        # we compute how many sequence we get from these tokens
        # and drop the last sequence if it is not long enough
        self.num_seq = len(self.data) // max_seq_len

    def __len__(self):
        # return self.num_seq
        return 20

    def __getitem__(self, idx: int) -> Any:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        x = torch.from_numpy(self.data[start:end].astype(np.int64))
        y = torch.from_numpy(self.data[start + 1 : end + 1].astype(np.int64))
        return x, y



class RandomSampler(Sampler):
    def __init__(self, data_source: Sized, shuffle: bool = False, seed=0) -> None:
        self.size = len(data_source)
        self.seed = seed
        self.epoch = 0
        self.shuffle = shuffle
        self.epoch_checkpoint : dict[int, int] = {}

    def __iter__(self) -> Iterator[int]:
        # if epoch_start has value for self.epoch, that means
        # all previous batch have been consumed, we would start with
        # that epoch
        checkpoint_start = self.epoch_checkpoint.get(self.epoch, 0)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.size, generator=g).tolist()  # type: ignore[arg-type]
            indices = indices[checkpoint_start:]
        else:
            indices = range(checkpoint_start, self.size)
        self.epoch += 1
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def state_dict(self, step: int, batch_size: int) -> dict[str, int]:
        index = step * batch_size
        return {"index": index}
        
    def load_state_dict(self, epoch: int, state: dict[str, int]) -> None:
        index = state.get("index", 0)
        self.epoch_checkpoint[epoch] = index


# TODO (ewei) this is for testing purpose, should move to a test file
class SimpleDataset(Dataset):
    def __init__(self, count):
        self._samples = list(range(count))
        
    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


if __name__ == "__main__":
    dataset = Dataset("data/openwebtext/train.bin", max_seq_len=1024)

    current_epoch = 1
    # ds = SimpleDataset(100)
    sampler = RandomSampler(dataset, shuffle=True)
    sampler.set_epoch(current_epoch)
    sampler.load_state_dict(current_epoch, {"index": 6})

    train_loader = DataLoader(dataset=dataset, batch_size=2, sampler=sampler)


    for epoch in range(current_epoch, 3):
        batch_iter = iter(train_loader)
        print(next(batch_iter))
        # for batch in batch_iter:
        #     print(batch)
        # for i, batch in enumerate(train_loader):
        #     print(f"epoch {epoch} batch {i} : {batch}")