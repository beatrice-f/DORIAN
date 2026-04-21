import litellm, json, copy, argparse
from pathlib import Path

MODEL = "ollama/qwen3:8b"
DATA_FILE = Path("/home/beatrice/phd/ISWC26/DORIAN/data/dev_subtask1_verbalized_triples.json")


def load_data(data_file):
    with open(data_file, encoding="utf-8") as f:
        return json.load(f)


def prompt_model(prompt, model=MODEL, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = litellm.completion(
        model=model,
        messages=messages,
        api_base="http://localhost:11434",
    )
    return response.choices[0].message.content


def run_pipeline(data, model=MODEL):
    results = copy.deepcopy(data)
    for lang in results:
        docs = results[lang]
        for i, (filename, doc) in enumerate(docs.items()):
            print(f"[{lang}] {i+1}/{len(docs)} {filename}")
            for ann in doc["annotations"]:
                ann["prediction"] = prompt_model(ann["prompt"], model=model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="LiteLLM model string (default: %(default)s)")
    parser.add_argument("--data", default=str(DATA_FILE), help="Path to dev_subtask1_verbalized_triples.json")
    parser.add_argument("--output", default="/home/beatrice/phd/ISWC26/ICL_with_verbalized_triples/preds/output.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading data...")
    data = load_data(args.data)

    print("Running pipeline...")
    results = run_pipeline(data, model=args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")
