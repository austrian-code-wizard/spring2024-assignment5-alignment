import csv
import os
import re
import json
import argparse
import subprocess
from time import time
from datetime import datetime
from vllm import LLM, SamplingParams


sys_prompt = """\
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
```{instruction}```

# Answer:
```"""

mmlu_prompt = """\
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.
Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

# Query: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
# Answer:"""


def load_mmlu_prompts(split: str = "test", path: str = "data/mmlu") -> list[tuple[str, str]]:
    data = []
    for filename in os.listdir(f"{path}/{split}"):
        subject = " ".join(filename.split("_")[:-1])
        with open(f"{path}/{split}/{filename}", "r") as f:
            reader = csv.reader(f)
            for row in reader:
                prompt = mmlu_prompt.format(subject=subject, question=row[0], options=row[1:-1])
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
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
{question}

# Answer:
"""


def load_gsm8k_prompts(split: str = "test", path: str = "data/gsm8k") -> list[tuple[str, float]]:
    data = []
    with open(f"{path}/{split}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            prompt = gsm8k_prompt.format(question=example["question"])
            true_answer = float(example["answer"].split("####")[-1].strip().replace(",", ""))
            data.append((prompt, true_answer))
    return data


numbers_as_words = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100
}


def parse_gsm8k_response(response: str) -> float | None:
    if response.endswith("."):
        response = response[:-1]
    response = response.replace(",", "")
    response = re.sub(r'[^\w\s.]', ' ', response)
    words = response.split()
    for word in reversed(words):
        try:
            if word in numbers_as_words:
                return numbers_as_words[word]
            return float(word)
        except ValueError:
            continue
    return None


def score_gsm8k_response(correct_response: float, parsed_response: float | None):
    if parsed_response is None:
        return 0.0
    return 1.0 if parsed_response == correct_response else 0.0


alpace_prompt = """\
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant.
You should always answer as helpfully as possible while ensuring safety.
Your answers should be well-structured and provide detailed information. They should also have an engaging tone.
Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.
Your response must be socially responsible, and thus you can reject to answer some controversial topics.

# Query:
{question}

# Answer:
"""


def load_alpaca_prompts(split: str = "eval", path: str = "data/alpaca_eval") -> list[tuple[str, str]]:
    data = []
    with open(f"{path}/alpaca_{split}.jsonl", "r") as f:
        for line in f:
            example = json.loads(line)
            prompt = alpace_prompt.format(question=example["instruction"])
            data.append((prompt, example["output"]))
    return data


def parse_alpaca_response(response: str) -> str | None:
    return response


def score_alpaca_response_batch(data: dict[str, str | int | None], model_name: str, dataset_names: list[str]):
    scores = []
    for d, name in zip(data, dataset_names):
        scores.append({
            "instruction": d["prompt"],
            "output": d["generated_text"],
            "generator": model_name,
            "dataset": name
        })
    with open("results/alpaca_scores.json", "w+") as f:
        json.dump(scores, f, indent=2)

    command = "conda init && conda activate cs336_alignment && alpaca_eval --model_outputs results/alpaca_scores.json --annotators_config 'scripts/alpaca_eval_vllm_llama3_70b_fn' --base-dir '.'"
    subprocess.run(command, shell=True, check=True)


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
        "score": score_alpaca_response_batch
    }
}


MODELS = {
    "llama3-8b": "/data/Meta-Llama-3-8B/",
    "llama3-70b": "/home/shared/Meta-Llama-3-70b-Instruct/",
}

MAX_TOKENS = {
    "mmlu": 1024,
    "gsm8k": 1024,
    "alpaca": 1024
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("--model", type=str, choices=MODELS.keys(), default="llama3-8b")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--results_path", type=str, default="results")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)


    data = DATASETS[args.dataset]["load"]()
    if args.num_samples > 0:
        data = data[:args.num_samples]
    prompts = [d[0] for d in data]
    responses = [d[1] for d in data]
    llm = LLM(model=MODELS[args.model])
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS[args.dataset], stop=["# Query:"]
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    total_score = 0.0
    for output, true_response in zip(outputs, responses):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        parsed_response = DATASETS[args.dataset]["parse"](generated_text)

        if args.dataset == "alpaca":
            score = 0.0
        else:
            score = DATASETS[args.dataset]["score"](true_response, parsed_response)
        total_score += score
        if args.verbose:
            print(f"###Prompt: {prompt}\n###Generated text: {generated_text}\n###Parsed response: {parsed_response}\n###Correct response: {true_response}\n###Score: {score}\n\n")
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "parsed_response": parsed_response,
            "true_response": true_response,
            "score": score
        })
    file_path = f"results/{args.dataset}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json"
    with open(file_path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"Total score: {total_score / len(outputs)}")
    print(f"Results saved to {file_path}")

    if args.dataset == "alpaca":
        dataset_names = [d["dataset"] for d in results]
        DATASETS[args.dataset]["score"](results, args.model, dataset_names)


if __name__ == "__main__":
    main()