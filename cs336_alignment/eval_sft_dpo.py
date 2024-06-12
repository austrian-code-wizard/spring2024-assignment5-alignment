import csv
import os
import re
import json
import argparse
import subprocess
from time import time
from datetime import datetime
from vllm import LLM, SamplingParams


dpo_sft_prompt_format = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
"""

mmlu_prompt = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

### Response:
"""


def load_mmlu_prompts(
    split: str = "test", path: str = "data/mmlu"
) -> list[tuple[str, str]]:
    data = []
    for filename in os.listdir(f"{path}/{split}"):
        subject = " ".join(filename.split("_")[:-1])
        with open(f"{path}/{split}/{filename}", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                prompt = mmlu_prompt.format(
                    subject=subject, question=row[0], options=row[1:-1]
                )
                data.append((prompt, row[-1]))
    return data


def parse_mmlu_response(response: str) -> str | None:
    answer = response.split("The correct answer is ")
    if len(answer) != 2:
        return None
    answer = answer[1][0]
    if answer not in "ABCD":
        return None
    return answer


def score_mmlu_response(correct_response: str, parsed_response: str | None):
    if parsed_response is None:
        return 0.0
    return 1.0 if parsed_response.strip() == correct_response.strip() else 0.0


gsm8k_prompt = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response:
"""


def load_gsm8k_prompts(
    split: str = "test", path: str = "data/gsm8k"
) -> list[tuple[str, float]]:
    data = []
    with open(f"{path}/{split}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            prompt = gsm8k_prompt.format(question=example["question"])
            true_answer = float(
                example["answer"].split("####")[-1].strip().replace(",", "")
            )
            data.append((prompt, true_answer))
    return data


def parse_gsm8k_response(response: str) -> float | None:
    if response.endswith("."):
        response = response[:-1]
    response = response.replace(",", "")
    response = re.sub(r"[^\w\s.]", " ", response)
    words = response.split()
    for word in reversed(words):
        try:
            return float(word)
        except ValueError:
            continue
    return None


def score_gsm8k_response(correct_response: float, parsed_response: float | None):
    if parsed_response is None:
        return 0.0
    return 1.0 if parsed_response == correct_response else 0.0


def load_alpaca_prompts(
    split: str = "eval", path: str = "data/alpaca_eval"
) -> list[dict[str, str]]:
    data = []
    with open(f"{path}/alpaca_{split}.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def parse_alpaca_response(response: str) -> str | None:
    return response


def score_alpaca_response_batch(
    results: dict[str, str | int | None], model_name: str, dataset_names: list[str]
):
    scores = []
    for d, name in zip(results, dataset_names):
        scores.append(
            {
                "instruction": d["prompt"].split("### Instruction:\n", maxsplit=1)[1].split("\n### Response:", maxsplit=1)[0],
                "output": d["generated_text"],
                "generator": model_name,
                "dataset": name,
            }
        )
    with open("results/alpaca_scores.json", "w+") as f:
        json.dump(scores, f, indent=2)

    # Now run `alpaca_eval --model_outputs results/alpaca_scores.json --annotators_config 'scripts/alpaca_eval_vllm_llama3_70b_fn' --base-dir '.'``


def load_simple_safety_prompts(
    split: str = "", path: str = "data/simple_safety_tests"
) -> list[dict[str, str]]:
    data = []
    with open(f"{path}/simple_safety_tests.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"instruction": row["prompts_final"], "output": None})
    return data


def parse_simple_safety_response(response: str) -> str | None:
    return response


def score_simple_safety_response(results: list[dict[str, str]]):
    with open("results/simple_safety_scores.json", "w+") as f:
        for d in results:
            res = {
                "prompts_final": d["prompt"].split("### Instruction:\n", maxsplit=1)[1].split("\n### Response:", maxsplit=1)[0],
                "output": d["generated_text"],
            }
            f.write(f"{json.dumps(res)}\n")

    """Now run ```bash
python scripts/evaluate_safety.py \
--input-path results/simple_safety_scores.json \
--model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct \
--num-gpus 2 \
--output-path simple_safety_results.json
    ```"""


DATASETS = {
    "mmlu": {
        "load": load_mmlu_prompts,
        "parse": parse_mmlu_response,
        "score": score_mmlu_response,
    },
    "gsm8k": {
        "load": load_gsm8k_prompts,
        "parse": parse_gsm8k_response,
        "score": score_gsm8k_response,
    },
    "alpaca": {
        "load": load_alpaca_prompts,
        "parse": parse_alpaca_response,
        "score": score_alpaca_response_batch,
    },
    "simple_safety": {
        "load": load_simple_safety_prompts,
        "parse": parse_simple_safety_response,
        "score": score_simple_safety_response,
    },
}


MODELS = {
    "llama3-8b": "/data/Meta-Llama-3-8B",
    "llama3-70b": "/home/shared/Meta-Llama-3-70B-Instruct",
    "llama3-8b-sft": "/home/c-moritzst/spring2024-assignment5-alignment/sft_results/sft_train-2024-06-08 16:08:04",
    "llama3-8b-dpo": "/home/c-moritzst/spring2024-assignment5-alignment/dpo_results/dpo_train-2024-06-11 22:16:58"
}

MAX_TOKENS = {"mmlu": 1024, "gsm8k": 1024, "alpaca": 1024, "simple_safety": 1024}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test"], default="test"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )
    parser.add_argument("--model", type=str, choices=MODELS.keys(), default="llama3-8b")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--results_path", type=str, default="results")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    data = DATASETS[args.dataset]["load"]()
    if args.num_samples > 0:
        data = data[: args.num_samples]

    if args.dataset in ["alpaca", "simple_safety"]:
        prompts = [dpo_sft_prompt_format.format(prompt=d["instruction"]) for d in data]
        responses = [d["output"] for d in data]
    else:
        prompts = [d[0] for d in data]
        responses = [d[1] for d in data]
    llm = LLM(model=MODELS[args.model])
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=MAX_TOKENS[args.dataset],
        stop=["### Instruction:"],
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    total_score = 0.0
    for output, true_response in zip(outputs, responses):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        parsed_response = DATASETS[args.dataset]["parse"](generated_text)

        if args.dataset in ["alpaca", "simple_safety"]:
            score = 0.0
        else:
            score = DATASETS[args.dataset]["score"](true_response, parsed_response)
        total_score += score
        if args.verbose:
            print(
                f"###Prompt: {prompt}\n###Generated text: {generated_text}\n###Parsed response: {parsed_response}\n###Correct response: {true_response}\n###Score: {score}\n\n"
            )
        results.append(
            {
                "prompt": prompt,
                "generated_text": generated_text,
                "parsed_response": parsed_response,
                "true_response": true_response,
                "score": score,
            }
        )
    file_path = (
        f"results/{args.dataset}_{args.model}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    )
    with open(file_path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"Total score: {total_score / len(outputs)}")
    print(f"Results saved to {file_path}")

    if args.dataset == "alpaca":
        dataset_names = [d["dataset"] for d in data]
        DATASETS[args.dataset]["score"](results, args.model, dataset_names)
    elif args.dataset == "simple_safety":
        DATASETS[args.dataset]["score"](results)


if __name__ == "__main__":
    main()
