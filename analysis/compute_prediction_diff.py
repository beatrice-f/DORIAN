"""
Compute the percentage of entity-role predictions that change between the
zero-shot baseline and each of the three graph-enhanced settings (all
triples, entity triples, verbalized triples), per model.

For a given model and run index n (1, 2, 3), the baseline run n and a given
graph-enhanced setting's run n are predictions over the *same* documents and
the *same* entities (same lang/doc_id/entity/start/end, in the same order),
so they are compared pairwise. The three runs are then pooled ("collapsed")
together into a single count per (model, setting) rather than reported
separately, as are all languages/documents/entities.

An entity's prediction is considered "different" if the set of predicted
role labels differs between the two settings (order-independent).

Usage:
    python compute_prediction_diff.py [--predictions-dir results/predictions_normalized] [--csv OUT.csv]
"""

import argparse
import json
import os
from collections import defaultdict

RUNS = (1, 2, 3)
GRAPH_SETTINGS = ("all_triples", "entity_triples", "verbalized_triples")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def annotation_key(ann):
    return (ann["entity"], ann["start"], ann["end"])


def prediction_set(ann):
    return frozenset(label.strip() for label in ann.get("prediction", []))


def discover_models(baseline_dir):
    return sorted(
        d for d in os.listdir(baseline_dir) if os.path.isdir(os.path.join(baseline_dir, d))
    )


def compare_run(baseline_path, graph_path):
    """Return (num_compared, num_different) for one pair of run files."""
    baseline = load_json(baseline_path)
    graph = load_json(graph_path)

    compared = 0
    different = 0

    for lang, docs in baseline.items():
        graph_docs = graph.get(lang)
        if graph_docs is None:
            continue
        for doc_id, doc in docs.items():
            graph_doc = graph_docs.get(doc_id)
            if graph_doc is None:
                continue

            graph_preds = {
                annotation_key(a): prediction_set(a) for a in graph_doc.get("annotations", [])
            }

            for ann in doc.get("annotations", []):
                key = annotation_key(ann)
                if key not in graph_preds:
                    continue
                compared += 1
                if prediction_set(ann) != graph_preds[key]:
                    different += 1

    return compared, different


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", default="results/predictions_normalized")
    parser.add_argument("--csv", help="optional path to write results as CSV")
    args = parser.parse_args()

    baseline_dir = os.path.join(args.predictions_dir, "baseline")
    models = discover_models(baseline_dir)

    # results[model][setting] = {"compared": int, "different": int}
    results = defaultdict(lambda: defaultdict(lambda: {"compared": 0, "different": 0}))
    # overall totals per setting, pooled across all models
    overall = defaultdict(lambda: {"compared": 0, "different": 0})

    missing_runs = []

    for model in models:
        for setting in GRAPH_SETTINGS:
            graph_dir = os.path.join(args.predictions_dir, setting)
            for n in RUNS:
                baseline_path = os.path.join(baseline_dir, model, f"baseline_{n}.json")
                graph_path = os.path.join(graph_dir, model, f"{setting}_{n}.json")

                if not os.path.exists(baseline_path) or not os.path.exists(graph_path):
                    missing_runs.append((model, setting, n))
                    continue

                compared, different = compare_run(baseline_path, graph_path)

                results[model][setting]["compared"] += compared
                results[model][setting]["different"] += different
                overall[setting]["compared"] += compared
                overall[setting]["different"] += different

    # ---- report ----
    settings_order = list(GRAPH_SETTINGS)

    header = f"{'model':<40}" + "".join(f"{s:>20}" for s in settings_order)
    print(header)
    print("-" * len(header))

    csv_rows = [["model"] + settings_order]

    for model in models:
        row_display = [model]
        row_csv = [model]
        for setting in settings_order:
            stats = results[model][setting]
            if stats["compared"] == 0:
                cell = "n/a"
            else:
                pct = 100.0 * stats["different"] / stats["compared"]
                cell = f"{pct:.2f}% ({stats['different']}/{stats['compared']})"
            row_display.append(cell)
            row_csv.append(
                ""
                if stats["compared"] == 0
                else f"{100.0 * stats['different'] / stats['compared']:.4f}"
            )
        print(f"{row_display[0]:<40}" + "".join(f"{c:>20}" for c in row_display[1:]))
        csv_rows.append(row_csv)

    print("-" * len(header))
    row_display = ["ALL MODELS (pooled)"]
    row_csv = ["ALL_MODELS_POOLED"]
    for setting in settings_order:
        stats = overall[setting]
        if stats["compared"] == 0:
            cell = "n/a"
        else:
            pct = 100.0 * stats["different"] / stats["compared"]
            cell = f"{pct:.2f}% ({stats['different']}/{stats['compared']})"
        row_display.append(cell)
        row_csv.append(
            "" if stats["compared"] == 0 else f"{100.0 * stats['different'] / stats['compared']:.4f}"
        )
    print(f"{row_display[0]:<40}" + "".join(f"{c:>20}" for c in row_display[1:]))
    csv_rows.append(row_csv)

    if missing_runs:
        print(
            f"\nNote: {len(missing_runs)} (model, setting, run) combinations were "
            f"skipped because a baseline or graph-enhanced prediction file was missing:"
        )
        for model, setting, n in missing_runs:
            print(f"  - {model} / {setting} / run {n}")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"\nWrote CSV to {args.csv}")


if __name__ == "__main__":
    main()
