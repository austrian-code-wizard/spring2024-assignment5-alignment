import csv
import os
import json
import argparse
from time import time
from tqdm import tqdm
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

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer: """


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
    answer = response.split("The correct answer is ")[1][0]
    if answer not in "ABCD":
        return None
    return answer


def score_mmlu_response(sample: tuple[str, str], parsed_response: str | None):
    if parsed_response is None:
        return 0.0
    return 1.0 if parsed_response == sample[1] else 0.0


DATASETS = {
    "mmlu": {
        "load": load_mmlu_prompts,
        "parse": parse_mmlu_response,
        "score": score_mmlu_response,
    }
}


MODELS = {
    "llama3-8b": "/data/Meta-Llama-3-8b",
    "llama3-70b": "/home/shared/Meta-Llama-3-70b-Instruct",
}

MAX_TOKENS = {
    "mmlu": 32
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


    prompts = DATASETS[args.dataset]["load"]()
    if args.num_samples > 0:
        prompts = prompts[:args.num_samples]
    llm = LLM(model=MODELS[args.model])
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS[args.dataset], stop=["\n"]
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    total_score = 0.0
    start = time()
    for output in tqdm(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        parsed_response = DATASETS[args.dataset]["parse"](generated_text)
        score = DATASETS[args.dataset]["score"](prompt, parsed_response)
        total_score += score
        if args.verbose:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Parsed response: {parsed_response!r}, Score: {score}")
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "parsed_response": parsed_response,
            "score": score
        })
    print(f"Estimated throughput: {len(outputs) / (time() - start):.2f} samples per second")
    file_path = f"results/{args.dataset}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt"
    with open(file_path, "w") as file:
        json.dump(results, file, indent=2)
    print(f"Total score: {total_score / len(outputs)}")
    print(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()