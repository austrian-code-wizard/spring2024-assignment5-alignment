#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import compute_per_instance_dpo_loss
from .common import FIXTURES_PATH


def test_per_instance_dpo_loss():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2")
    model_ref = AutoModelForCausalLM.from_pretrained(FIXTURES_PATH / "tiny-gpt2-ref")

    prompt = "The quick brown fox jumps over"
    good_response = "the lazy dog."
    bad_response = "their crazy frog."

    loss = compute_per_instance_dpo_loss(
        model=model,
        pi_ref=model_ref,
        beta=0.5,
        tokenizer=tokenizer,
        prompt=prompt,
        good_response=good_response,
        bad_response=bad_response,
    )

    assert torch.isclose(loss, torch.tensor(0.5785), atol=1e-4)
