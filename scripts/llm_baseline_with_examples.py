import litellm, json, copy, argparse
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

- Victim: Example: Victims of natural disasters, such as hurricanes or earthquakes; individuals affected by violent crimes. Victims of economic blockades, sanctions, or boycotts.
- Guardian: Example: Police officers protecting citizens during a crisis, firefighters saving lives during a disaster, community leaders standing against crime or leaders standing up for action to address climate change.
- Foreign_Adversary: Example: Rival nations involved in espionage or military confrontations, such as the Cold War adversaries, or countries accused of election interference.
- Virtuous: Example: Judges known for their fairness, or politicians with a reputation for honesty and ethical behavior.
- Instigator: Example: Politicians using inflammatory rhetoric to incite violence, or groups instigating protests to destabilize governments.
- Peacemaker: Example: Nelson Mandela’s efforts to reconcile South Africa post-apartheid, or diplomats working to broker peace deals between conflicting nations.
- Incompetent: Example: Leaders making reckless policy decisions without proper understanding, officials mishandling crisis responses, or managers whose poor judgment leads to organizational failures.
- Tyrant: Example: Dictators like Kim Jong-un in North Korea, or corrupt officials embezzling public funds and suppressing dissent.
- Conspirator: Example: Figures involved in political scandals or espionage, such as Watergate conspirators or modern cyber espionage cases.
- Rebel: Example: Leaders of independence movements like Mahatma Gandhi in India, or modern-day activists fighting for democratic reforms in authoritarian regimes.
- Deceiver: Example: Politicians spreading false information for political gain, or media engaging in propaganda.
- Terrorist: Example: Groups like ISIS or Al-Qaeda carrying out attacks, or lone-wolf terrorists committing acts of violence.
- Underdog: Example: Grassroots political candidates overcoming well-funded incumbents, or small nations standing up to larger, more powerful countries.
- Corrupt: Example: Companies involved in environmental pollution, executives engaged in massive financial fraud, or politicians accepting bribes and engaging in graft.
- Exploited: Example: Workers in sweatshops; victims of human trafficking; communities suffering from corporate exploitation of natural resources.
- Saboteur: Example: Insiders tampering with critical infrastructure, or activists sabotaging industrial operations.
- Bigot: Bigot: Individuals accused of hostility or discrimination against specific groups. This includes entities committing acts falling under racism, sexism, homophobia, antisemitism, islamophobia, or any kind of hate speech.
- Traitor: Example: Whistleblowers revealing sensitive information for personal gain, or soldiers defecting to enemy forces.
- Forgotten: Example: Indigenous populations facing ongoing discrimination; homeless individuals struggling without adequate support; refugees fleeing conflict or persecution.
- Scapegoat: Example: Minority groups blamed for economic problems; political opponents, accused of provoking national strife, without evidence.
- Martyr: Example: Civil rights leaders like Martin Luther King Jr., who was assassinated while fighting for equality, or journalists who risk their lives to report on corruption and injustice.
- Spy: Example: Historical figures like Aldrich Ames, who spied for the Soviet Union, or contemporary cases of corporate espionage.

This is the text: "{new_text}",
this is the entity in the text: [[{entity}]].

Reply ONLY with the correct narrative role(s). Do not add any comment, role description or text.""".strip()


def run_pipeline(data, model=MODEL):
    results = copy.deepcopy(data)
    for lang in results:
        docs = results[lang]
        for i, (filename, doc) in enumerate(docs.items()):
            print(f"[{lang}] {i+1}/{len(docs)} {filename}")
            for ann in doc["annotations"]:
                prompt = build_prompt(doc["text"], ann["entity"], ann["start"], ann["end"])
                ann["prediction"] = prompt_model(prompt, model=model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, help="LiteLLM model string (default: %(default)s)")
    parser.add_argument("--output", default="/home/beatrice/phd/ISWC26/baseline_qwen8b/isolateentity_predictions/dev_predictions_qwen8b_noisy.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print("Loading data...")
    data = load_data(DATA_DIR)

    print("Running pipeline...")
    results = run_pipeline(data, model=args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output}")
