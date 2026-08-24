"""
Run one Ollama-served LLM (via LiteLLM) over a prompt-ready dev_subtask1_*.json
file produced by build_prompts.py, writing raw completions back into each
annotation's "prediction" field.

Usage:
    python run_llm.py --model ollama/gemma4:26b-a4b-it-q4_K_M \
        --data data/dev_subtask1_baseline.json \
        --output results/predictions/baseline/gemma4:26b-a4b-it-q4_K_M/baseline_1.json
"""

import argparse
import copy
import json
from pathlib import Path

import litellm


def load_data(data_file):
    with open(data_file, encoding="utf-8") as f:
        return json.load(f)


def prompt_model(prompt, model, system=None, api_base="http://localhost:11434"):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        response = litellm.completion(model=model, messages=messages, api_base=api_base)
        return response.choices[0].message.content
    except Exception:
        return ""


def run_pipeline(data, model, api_base):
    results = copy.deepcopy(data)
    for lang in results:
        docs = results[lang]
        for i, (filename, doc) in enumerate(docs.items()):
            print(f"[{lang}] {i + 1}/{len(docs)} {filename}")
            for ann in doc["annotations"]:
                ann["prediction"] = prompt_model(ann["prompt"], model=model, api_base=api_base)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="LiteLLM model string, e.g. ollama/mistral:7b"
    )
    parser.add_argument(
        "--data", required=True, help="Prompt-ready json from build_prompts.py"
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--api-base", default="http://localhost:11434", help="Ollama server URL")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading data...")
    data = load_data(args.data)

    print("Running pipeline...")
    results = run_pipeline(data, model=args.model, api_base=args.api_base)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")
