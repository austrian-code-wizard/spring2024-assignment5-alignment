import os

# Increase timeout to prevent wandb errors
os.environ["WANDB__SERVICE_WAIT"] = "300"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

import math
import torch
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from cs336_alignment.dpo import get_dpo_dataset, dpo_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    model_name_or_path: str = "/data/Meta-Llama-3-8B"
    output_dir: str = "dpo_results"
    num_checkpoints: int = 4
    seq_length: int = 512
    batch_size: int = 1
    grad_accum_steps: int = 64
    lr: float = 1e-6
    warmup_ratio: float = 0.03
    log_every: int = 25
    log_to_wandb: bool = True
    use_lr_schedule: bool = True
    num_samples: int = -1
    num_val_batches: int = 200
    val_every: int = 250
    dpo_beta: float = 0.1


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) * math.pi / (cosine_cycle_iters - warmup_iters)
            )
        ) * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def main(config: Config, run_name: str = None):

    if run_name is None:
        run_name = ""
    else:
        run_name += "-"
    run_name += datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    devices = torch.cuda.device_count()
    assert devices >= 2, "This script requires at least 2 GPUs to run."
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(model_device)
    model.train()

    ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    ref_model.to(ref_device)
    ref_model.eval()

    wandb.init(project="cs336", name=run_name, config=config)

    train_dataset, val_dataset = get_dpo_dataset()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)

    num_batches = len(train_dataset) // config.batch_size
    num_steps = num_batches // config.grad_accum_steps

    output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    cur_loss = 0
    cur_step = 0
    for idx, (train_batch) in tqdm(enumerate(train_dataset), total=num_batches):
        loss = dpo_loss(
            model,
            ref_model,
            tokenizer,
            config.dpo_beta,
            train_batch["prompt"],
            train_batch["chosen"],
            train_batch["rejected"],
        )
        loss.backward()
        cur_loss += loss.item()
        if (idx + 1) % config.grad_accum_steps == 0:
            if config.use_lr_schedule:
                lr = learning_rate_schedule(
                    cur_step,
                    max_learning_rate=config.lr,
                    min_learning_rate=0.0,
                    warmup_iters=round(num_steps * config.warmup_ratio),
                    cosine_cycle_iters=num_steps
                    - round(num_steps * config.warmup_ratio),
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
            else:
                lr = optimizer.param_groups[0]["lr"]

            optimizer.step()
            optimizer.zero_grad()
            cur_step += 1
            wandb_log = {}
            if config.log_to_wandb:
                wandb_log["train_loss"] = cur_loss / config.grad_accum_steps
                wandb_log["lr"] = lr
            if cur_step % config.log_every == 0:
                print(
                    f"\nStep {cur_step}/{num_steps} ({cur_step/num_steps*100:.2f}%), Loss: {cur_loss / config.grad_accum_steps}"
                )
            cur_loss = 0

            if cur_step % (num_steps // config.num_checkpoints) == 0:
                print(f"\nSaving checkpoint {cur_step}")
                model.save_pretrained(
                    save_directory=os.path.join(output_dir, f"checkpoint_{cur_step}")
                )
                tokenizer.save_pretrained(
                    save_directory=os.path.join(output_dir, f"checkpoint_{cur_step}")
                )
            if cur_step % config.val_every == 0:
                print(f"\nRunning validation {cur_step}")
                val_loss = 0
                val_iters = 0
                model.eval()
                with torch.no_grad():
                    for val_batch in val_dataset:
                        loss = dpo_loss(
                            model,
                            ref_model,
                            tokenizer,
                            config.dpo_beta,
                            val_batch["prompt"],
                            val_batch["chosen"],
                            val_batch["rejected"],
                        )
                        val_loss += loss.item()
                        val_iters += 1
                        if val_iters >= config.num_val_batches:
                            break
                model.train()
                wandb_log["val_loss"] = val_loss / config.num_val_batches
                print(
                    f"\nStep {cur_step}/{num_steps} ({cur_step/num_steps*100:.2f}%), Validation loss: {val_loss / config.num_val_batches}"
                )

            if len(wandb_log) > 0 and config.log_to_wandb:
                wandb.log(wandb_log)

    wandb.finish()
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None)
    parser.add_argument(
        "--model_name_or_path", type=str, default="/data/Meta-Llama-3-8B"
    )
    parser.add_argument("--output_dir", type=str, default="dpo_results")
    parser.add_argument("--num_checkpoints", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=-1)
    args = parser.parse_args()
    config = Config(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        num_checkpoints=args.num_checkpoints,
        num_samples=args.num_samples,
    )
    main(config, run_name=args.run)
