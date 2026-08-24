import argparse
import json
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-sentence frame-semantic predictions (span-finder/LOME, "
            "one JSONL file per document) into one unified per-document JSON file."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="frames",
        help="Directory of raw per-sentence span-finder/LOME predictions (JSONL per document)",
    )
    parser.add_argument(
        "--output-dir",
        default="frames",
        help="Where to write the unified per-document JSON files (may be the same as --input-dir)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pred_f in input_dir.glob("*.json"):
        preds = [json.loads(l.strip()) for l in open(pred_f).readlines()]

        pred = {
            "tokens": [t for p in preds for t in p["tokens"]],
            "sentence": " ".join([s["sentence"] for s in preds]),
            "language": preds[0]["language"],
            "metadata": preds[0]["metadata"],
            "frames": [],
        }

        start = 0
        for pred_i in preds:
            for frame in pred_i["frames"]:
                ai, aj = frame["idxs"]
                frame["idxs"] = (start + ai, start + aj)

                for role in frame["roles"]:
                    ri, rj = role["idxs"]
                    role["idxs"] = (start + ri, start + rj)

                pred["frames"].append(frame)

            start += len(pred_i["tokens"])

        json.dump(pred, open(output_dir / f"{pred_f.stem}.json", "w"))
