import json, csv
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).parent
NORMALIZED_DIR = ROOT / "normalized"
RESULTS_DIR = ROOT / "eval_results"
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

    def scores(yt, yp, metrics):
        Yt, Yp = mlb.transform(yt), mlb.transform(yp)
        return {
            "precision_micro": precision_score(Yt, Yp, average="micro", zero_division=0),
            "recall_micro":    recall_score(Yt, Yp, average="micro", zero_division=0),
            "f1_micro":        f1_score(Yt, Yp, average="micro", zero_division=0),
            "f1_macro":        f1_score(Yt, Yp, average="macro", zero_division=0),
        }

    row = scores(all_true, all_pred, OVERALL_METRICS)
    for lang, (yt, yp) in lang_data.items():
        s = scores(yt, yp, LANG_METRICS)
        row[f"{lang}_f1_micro"] = s["f1_micro"]
        row[f"{lang}_f1_macro"] = s["f1_macro"]
    return row


def parse_stem(stem):
    """'baseline_icl_2' -> experiment='baseline_icl', run='2'"""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return stem, ""


if __name__ == "__main__":
    rows = []

    for json_file in sorted(NORMALIZED_DIR.rglob("*.json")):
        model = json_file.parent.name
        experiment, run = parse_stem(json_file.stem)

        with open(json_file, encoding="utf-8") as f:
            preds = json.load(f)

        row = {"model": model, "experiment": experiment, "run": run}
        row.update(compute_metrics(preds))
        rows.append(row)

    # Detect language columns from first row
    lang_cols = [k for k in rows[0] if k not in {"model", "experiment", "run"} and k not in OVERALL_METRICS]
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

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved {len(all_rows)} rows ({len(rows)} runs + {len(mean_rows)} means) to {out_path}")
