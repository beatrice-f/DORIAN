import litellm, json, copy, re, argparse
from pathlib import Path
from collections import defaultdict
import time

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
    )
    return response.choices[0].message.content


def build_prompt(text, entity, start, end):
    new_text = isolate_entity(text, start, end)
    return f"""You are a powerful multi-label classificator. Given a text and an entity within the text, you have to classify the entity into one or two of the following narrative roles:
Victim: People cast as victims due to circumstances beyond
their control, specifically in two categories: (1) victims of physical harm, including natural disasters, acts of war, terrorism, physical assault, ... etc., and (2) victims of economic harm, such as sanctions and boycotts;
Guardian: Heroes or guardians who protect values or communities, ensuring safety and upholding justice;
Foreign_Adversary: Entities from other nations or regions creating geopolitical tension and acting against the interests of another country;
Virtuous: Individuals portrayed as virtuous, righteous, or noble, who are seen as fair, just, and upholding high moral standards; 
Instigator: Individuals or groups initiating conflict, often seen as the primary cause of tension and discord;
Peacemaker: Individuals who advocate for harmony, working tirelessly to resolve conflicts and bring about peace;
Incompetent: Entities causing harm through ignorance, lack of skill, or incompetence;
Tyrant: Tyrants and corrupt officials who abuse their power, ruling unjustly and oppressing those under their control; 
Conspirator: Those involved in plots and secret plans, often working behind the scenes to undermine or deceive others; 
Rebel: Rebels, revolutionaries, or freedom fighters who challenge the status quo and fight for significant change or liberation from oppression;
Deceiver: Deceivers, manipulators, or propagandists who twist the truth, spread misinformation, and manipulate public perception for their own benefit;
Terrorist: Terrorists, mercenaries, insurgents, fanatics, or extremists engaging in violence and terror to further ideological ends, often targeting civilians; 
Underdog: Entities who are considered unlikely to succeed due to their disadvantaged position but strive against greater forces and obstacles;
Corrupt: Individuals or entities that engage in unethical or illegal activities for personal gain, prioritizing profit or power over ethics;
Exploited: Individuals or groups used for others’ gain, often without their consent and with significant detriment to their wellbeing;
Saboteur: Saboteurs who deliberately damage or obstruct systems, processes, or organizations to cause disruption or failure; 
Bigot: Individuals accused of hostility or discrimination against specific groups;
Traitor: Individuals who betray a cause or country, often seen as disloyal and treacherous;
Forgotten: Marginalized or overlooked groups who are often ignored by society and do not receive the attention or support they need; 
Scapegoat: Entities blamed unjustly for problems or failures, often to divert attention from the real causes or culprits;
Martyr: Martyrs or saviors who sacrifice their well-being, or even their lives, for a greater good or cause;
Spy: Spies or double agents accused of espionage, gathering and transmitting sensitive information to a rival or enemy.

This is the text: "{new_text}",
this is the entity in the text: [[{entity}]].

Reply ONLY with the correct narrative role(s). Do not add any comment, role description or text.""".strip()


def run_pipeline(data, model=MODEL):
    results = copy.deepcopy(data)
    for lang in results:
        docs = results[lang]
        for i, (filename, doc) in enumerate(docs.items()):
            print(f"[{lang}] {i+1}/{len(docs)}")
            for ann in doc["annotations"]:
                prompt = build_prompt(doc["text"], ann["entity"], ann["start"], ann["end"])
                #ann["prediction"] = parse_prediction(prompt_model(prompt, model=model))
                ann["prediction"] = prompt_model(prompt, model=model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="LiteLLM model string (default: %(default)s)")
    parser.add_argument("--output", default="/home/beatrice/phd/ISWC26/baseline_qwen8b/isolateentity_predictions/dev_ICL_predictions_qwen8b.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading data...")
    data = load_data(DATA_DIR)

    print("Running pipeline...")
    results = run_pipeline(data, model=args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")
