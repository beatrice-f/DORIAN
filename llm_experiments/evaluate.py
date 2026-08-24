"""
Compute precision/recall/F1-micro/F1-macro/EMR (overall + per-language) from
normalized predictions. Produces results/metrics.csv (Tables 2 & 3).

Usage:
    python evaluate.py --normalized-dir results/predictions_normalized --out results/metrics.csv
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

OVERALL_METRICS = ["precision_micro", "recall_micro", "f1_micro", "f1_macro"]
LANG_METRICS = ["f1_micro", "f1_macro"]


def compute_metrics(preds):
    """Return overall scores + per-lang f1 scores."""
    all_true, all_pred = [], []
    lang_data = {}

    for lang, docs in preds.items():
        yt, yp = [], []
        for doc in docs.values():
            for ann in doc["annotations"]:
                yt.append(ann.get("labels", []))
                yp.append(ann.get("prediction", []))
        lang_data[lang] = (yt, yp)
        all_true.extend(yt)
        all_pred.extend(yp)

    mlb = MultiLabelBinarizer()
    mlb.fit(all_true + all_pred)

    def scores(yt, yp):
        Yt, Yp = mlb.transform(yt), mlb.transform(yp)
        return {
            "precision_micro": precision_score(Yt, Yp, average="micro", zero_division=0),
            "recall_micro": recall_score(Yt, Yp, average="micro", zero_division=0),
            "f1_micro": f1_score(Yt, Yp, average="micro", zero_division=0),
            "f1_macro": f1_score(Yt, Yp, average="macro", zero_division=0),
            "emr": np.all(Yt == Yp, axis=-1).mean(),
        }

    row = scores(all_true, all_pred)
    for lang, (yt, yp) in lang_data.items():
        s = scores(yt, yp)
        row[f"{lang}_f1_micro"] = s["f1_micro"]
        row[f"{lang}_f1_macro"] = s["f1_macro"]
        row[f"{lang}_emr"] = s["emr"]
    return row


def parse_stem(stem):
    """'baseline_2' -> experiment='baseline', run='2'"""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return stem, ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    normalized_dir = Path(args.normalized_dir)
    rows = []

    for json_file in sorted(normalized_dir.rglob("*.json")):
        model = json_file.parent.name
        experiment, run = parse_stem(json_file.stem)

        with open(json_file, encoding="utf-8") as f:
            preds = json.load(f)

        row = {"model": model, "experiment": experiment, "run": run}
        row.update(compute_metrics(preds))
        rows.append(row)

    # Detect language columns from first row
    lang_cols = [
        k
        for k in rows[0]
        if k not in {"model", "experiment", "run"} and k not in OVERALL_METRICS
    ]
    fieldnames = ["model", "experiment", "run"] + OVERALL_METRICS + sorted(lang_cols)

    # Mean rows grouped by (model, experiment)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["experiment"])].append(row)

    mean_rows = []
    for (model, experiment), group in groups.items():
        if len(group) < 2:
            continue
        mean_row = {"model": model, "experiment": experiment, "run": "mean"}
        for col in fieldnames[3:]:
            mean_row[col] = sum(r[col] for r in group) / len(group)
        mean_rows.append(mean_row)

    all_rows = rows + mean_rows
    all_rows.sort(key=lambda r: (r["model"], r["experiment"], r["run"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows ({len(rows)} runs + {len(mean_rows)} means) to {out_path}")
