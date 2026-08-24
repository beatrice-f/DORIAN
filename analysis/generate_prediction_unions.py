"""
For every model and every setting (baseline + the 3 graph-enhanced
settings), pool all available runs (1-3) together per entity and emit one
JSONL record per (llm, setting, lang, doc_id) listing every entity in that
document together with the union of its predicted role labels across runs
-- flattened (each run's 1-2 labels unpacked individually) and with repeats
kept, e.g. ["Tyrant", "Victim", "Tyrant", "Virtuous"].

This is the source of results/predictions_union.jsonl, consumed by
analysis/entropy_hellinger.ipynb (Tables 4 & 5).

Usage:
    python generate_prediction_unions.py [--predictions-dir results/predictions_normalized] [--out results/predictions_union.jsonl]
"""

import argparse
import json
import os

RUNS = (1, 2, 3)
SETTINGS = ("baseline", "all_triples", "entity_triples", "verbalized_triples")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotation_key(ann):
    return (ann["entity"], ann["start"], ann["end"])


def discover_models(baseline_dir):
    return sorted(
        d for d in os.listdir(baseline_dir) if os.path.isdir(os.path.join(baseline_dir, d))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        default="results/predictions_normalized",
        help="Root dir with one subdir per setting (see run_experiments.sh)",
    )
    parser.add_argument("--out", default="results/predictions_union.jsonl")
    args = parser.parse_args()

    baseline_dir = os.path.join(args.predictions_dir, "baseline")
    models = discover_models(baseline_dir)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for model in models:
            for setting in SETTINGS:
                setting_dir = os.path.join(args.predictions_dir, setting)
                run_data = []
                for n in RUNS:
                    path = os.path.join(setting_dir, model, f"{setting}_{n}.json")
                    if os.path.exists(path):
                        run_data.append(load_json(path))

                if not run_data:
                    continue

                # doc/lang structure (and entity keys) is identical across
                # runs, so use the first available run to enumerate them.
                for lang, docs in run_data[0].items():
                    for doc_id in docs:
                        # entity_key -> flattened list of predicted labels
                        # across all runs, repeats kept
                        entity_preds = {}
                        entity_gold = {}
                        entity_order = []

                        for run in run_data:
                            doc = run.get(lang, {}).get(doc_id)
                            if doc is None:
                                continue
                            for ann in doc.get("annotations", []):
                                key = annotation_key(ann)
                                if key not in entity_preds:
                                    entity_preds[key] = []
                                    entity_gold[key] = ann.get("labels", [])
                                    entity_order.append(key)
                                entity_preds[key].extend(
                                    label.strip() for label in ann.get("prediction", [])
                                )

                        entities = [
                            {
                                "entity": key[0],
                                "start": key[1],
                                "end": key[2],
                                "gold_labels": entity_gold[key],
                                "predictions": entity_preds[key],
                            }
                            for key in entity_order
                        ]

                        record = {
                            "llm": model,
                            "setting": setting,
                            "lang": lang,
                            "doc_id": doc_id,
                            "entities": entities,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
