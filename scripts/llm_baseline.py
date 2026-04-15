import litellm, json, copy, re, argparse
from pathlib import Path
from collections import defaultdict

MODEL = "ollama/qwen3:8b"
DATA_DIR = Path("data/dev-labels")


def load_data(data_dir):
    output = {}
    for lang_dir in sorted(data_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name

        annotations = defaultdict(list)
        ann_file = lang_dir / "subtask-1-annotations.txt"
        for line in ann_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            filename, entity, start, end, *labels = parts
            annotations[filename].append({
                "entity": entity,
                "start": int(start),
                "end": int(end),
                "main_label": labels[0],
                "labels": labels[1:]
            })

        doc_dir = lang_dir / "subtask-1-documents"
        lang_data = {}
        for doc_file in sorted(doc_dir.glob("*.txt")):
            name = doc_file.name
            lang_data[name] = {
                "text": doc_file.read_text(encoding="utf-8"),
                "annotations": annotations.get(name, [])
            }

        output[lang] = lang_data
    return output

def isolate_entity(text, start, end):
    return text[:start] + "[[" + text[start:end+1] + "]]" + text[end+1:]

def prompt_model(prompt, model=MODEL, system=None):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = litellm.completion(
        model=model,
        messages=messages,
        api_base="http://localhost:11434",
        #max_tokens=100,
    )
    return response.choices[0].message.content


def build_prompt(text, entity, start, end):
    new_text = isolate_entity(text, start, end)
    return f"""You are a powerful multi-label classificator. Given a text and an entity within the text, you have to classify the entity into one or two of the following narrative roles:
Victim, Guardian, Foreign_Adversary, Virtuous, Instigator, Peacemaker, Incompetent, Tyrant, Conspirator, Rebel, Deceiver, Terrorist, Underdog, Corrupt, Exploited, Saboteur, Bigot, Traitor, Forgotten, Scapegoat, Martyr, Spy.

This is the text: "{new_text}",
this is the entity in the text: [[{entity}]].

Reply ONLY with the correct narrative role(s). Do not add any comment or text.""".strip() # /no_think


def run_pipeline(data, model=MODEL):
    results = copy.deepcopy(data)
    for lang in results:
        docs = results[lang]
        for i, (filename, doc) in enumerate(docs.items()):
            print(f"[{lang}] {i+1}/{len(docs)} {filename}")
            for ann in doc["annotations"]:
                prompt = build_prompt(doc["text"], ann["entity"], ann["start"], ann["end"])
                #ann["prediction"] = parse_prediction(prompt_model(prompt, model=model))
                ann["prediction"] = prompt_model(prompt, model=model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="LiteLLM model string (default: %(default)s)")
    parser.add_argument("--output", default="dev_subtask1_predictions_qwen9b.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading data...")
    data = load_data(DATA_DIR)

    print("Running pipeline...")
    results = run_pipeline(data, model=args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")
