import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Any

def main():
    num_proc = 8
    seed = 0
    test_size = 0.05
    max_length = 2048

    # let's start with a small dataset to test things
    dataset = load_dataset("stas/openwebtext-10k", num_proc=num_proc)
    split_dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, model_max_length=8192)
    print(tokenizer)

    print(tokenizer.bos_token_id)
    print(tokenizer.eos_token_id)

    def process(example: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(example['text'], truncation=False, padding=False)
        iids = encoded['input_ids']
        # only add the eos_token here, since it for sequence completion task, it doesn't feel natural
        # that we need to add bos_token in front of all tokens
        # However, Mosaic ML seems have different idea
        # https://github.com/mosaicml/llm-foundry/blob/390951639a4b38a0c2de6ae3fac200eeab675941/llmfoundry/data/data.py#L110
        return {"id": iids + [tokenizer.eos_token_id]}

    print(split_dataset["train"][0])

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        num_proc=4,
    )

    print(tokenized["train"][0])






if __name__ == "__main__":
    main()