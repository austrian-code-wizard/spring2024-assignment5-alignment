import json
import gzip
import torch
import random
from tqdm import tqdm

random.seed(42)

prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path: str,
        seq_length: int,
        shuffle: bool,
        num_samples: int = -1,
    ):
        if str(dataset_path).endswith(".gz"):
            assert str(dataset_path).endswith(".jsonl.gz"), "Only support .jsonl files"
            with gzip.open(dataset_path, "r") as f:
                data = [json.loads(line) for line in f]
        else:
            assert str(dataset_path).endswith(".jsonl"), "Only support .jsonl files"
            with open(dataset_path, "r") as f:
                data = [json.loads(line) for line in f]
        if shuffle:
            random.shuffle(data)
        if num_samples > 0:
            data = data[:num_samples]
        data = [prompt_format.format(**d) for d in data]
        data = [tokenizer.encode(d) + [tokenizer.eos_token_id] for d in tqdm(data)]
        data = [token for sequence in data for token in sequence]

        # We need a valid label for the last token, so if the length of the data is divisible by the sequence length,
        # we remove the last token to get rid of the last full sequence.
        if len(data) % seq_length == 0:
            data = data[:-1]

        self._data = torch.tensor(data, dtype=torch.long)
        self._seq_length = seq_length

    def __len__(self):
        return len(self._data) // self._seq_length

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        return {
            "input_ids": self._data[
                idx * self._seq_length : (idx + 1) * self._seq_length
            ],
            "labels": self._data[
                idx * self._seq_length + 1 : (idx + 1) * self._seq_length + 1
            ],
        }


def iterate_batches(dataset: SFTDataset, batch_size: int, shuffle: bool):
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        yield {
            "input_ids": torch.stack(
                [dataset[idx]["input_ids"] for idx in batch_indices]
            ),
            "labels": torch.stack([dataset[idx]["labels"] for idx in batch_indices]),
        }
