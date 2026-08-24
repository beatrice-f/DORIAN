"""
Normalize raw LLM completions (free text) into a list of valid role labels.

Usage:
    python normalize_predictions.py --preds-dir results/predictions \
        --out-dir results/predictions_normalized
"""

import argparse
import copy
import json
import re
from pathlib import Path

VALID_LABELS = {
    "Victim", "Guardian", "Foreign Adversary", "Virtuous", "Instigator", "Peacemaker",
    "Incompetent", "Tyrant", "Conspirator", "Rebel", "Deceiver", "Terrorist",
    "Underdog", "Corrupt", "Exploited", "Saboteur", "Bigot", "Traitor",
    "Forgotten", "Scapegoat", "Martyr", "Spy",
}


def parse_prediction(raw):
    text = raw.strip()
    text = re.sub(r"(?i)foreign[_\-]adversary", "Foreign Adversary", text)
    multi_word = {label for label in VALID_LABELS if " " in label}
    found = []
    for label in multi_word:
        if label in text:
            found.append(label)
            text = text.replace(label, "")
    tokens = [t.strip() for t in re.split(r"[\s,\n]+", text) if t.strip()]
    found += [t for t in tokens if t in VALID_LABELS]
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preds-dir", required=True, help="Directory of raw prediction files (searched recursively)"
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir)

    for input_path in sorted(preds_dir.rglob("*.json")):
        rel = input_path.relative_to(preds_dir)
        output_path = out_dir / rel
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as f:
            results_raw = json.load(f)

        results_normalized = copy.deepcopy(results_raw)

        for lang in results_normalized:
            for filename, doc in results_normalized[lang].items():
                for ann in doc["annotations"]:
                    if "prediction" in ann:
                        ann["prediction"] = parse_prediction(ann["prediction"])

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_normalized, f, ensure_ascii=False, indent=2)

        print(f"  {rel}")
    print("Done.")
