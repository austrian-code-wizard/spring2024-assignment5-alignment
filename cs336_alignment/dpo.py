import os
import json
import gzip
import torch
import random
from transformers import PreTrainedTokenizer, PreTrainedModel


prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{response}"""


def dpo_loss(
    pi: PreTrainedModel,
    pi_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    beta: float,
    prompt: str,
    resp_chosen: str,
    resp_rejected: str
) -> torch.Tensor:
    
    def get_total_logprob(policy, input, labels):
        logits = policy(input).logits[:, :-1, :]
        logits = torch.nn.functional.log_softmax(logits, dim=-1)
        return torch.gather(logits, 2, labels.unsqueeze(-1)).squeeze(-1).sum(-1)

    input_chosen = torch.tensor(tokenizer.encode(prompt_format.format(prompt=prompt, response=resp_chosen)) + [tokenizer.eos_token_id]).unsqueeze(0)
    labels_chosen = input_chosen.clone()[:, 1:]
    input_rejected = torch.tensor(tokenizer.encode(prompt_format.format(prompt=prompt, response=resp_rejected)) + [tokenizer.eos_token_id]).unsqueeze(0)
    labels_rejected = input_rejected.clone()[:, 1:]

    with torch.no_grad():
        pi_ref_chosen = get_total_logprob(pi_ref, input_chosen.to(pi_ref.device), labels_chosen.to(pi_ref.device))
        pi_ref_rejected = get_total_logprob(pi_ref, input_rejected.to(pi_ref.device), labels_rejected.to(pi_ref.device))
    pi_chosen = get_total_logprob(pi, input_chosen.to(pi.device), labels_chosen.to(pi.device))
    pi_rejected = get_total_logprob(pi, input_rejected.to(pi.device), labels_rejected.to(pi.device))
    return -torch.nn.functional.logsigmoid(beta * (pi_chosen - pi_rejected + pi_ref_rejected.to(pi.device) - pi_ref_chosen.to(pi.device)))
    

def get_dpo_dataset(base_path: str = "/home/shared/hh", num_val: int = 200) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    data = []
    for path in ["harmless-base.jsonl.gz", "helpful-base.jsonl.gz", "helpful-online.jsonl.gz", "helpful-rejection-sampled.jsonl.gz"]:
        with gzip.open(os.path.join(base_path, path), "r") as f:
            new_data = [json.loads(line) for line in f]
            new_data = [d for d in new_data if d["chosen"].count("\n\nHuman:") == 1 or d["rejected"].count("\n\nHuman:") == 1]
            new_data = [{
                "prompt": d["chosen"].split("\n\nAssistant:", maxsplit=1)[0].removeprefix("\n\nHuman: ").strip(),
                "chosen": d["chosen"].split("\n\nAssistant: ", maxsplit=1)[1].split("\n\nHuman: ", maxsplit=1)[0].strip(),
                "rejected": d["rejected"].split("\n\nAssistant: ", maxsplit=1)[1].split("\n\nHuman: ", maxsplit=1)[0].strip(),
                "dataset": path

            } for d in new_data]
            data += new_data
    random.shuffle(data)
    return data[num_val:], data[:num_val]