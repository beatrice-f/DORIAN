import argparse
import json
import os
from glob import glob

from tqdm import tqdm
from xcore import xCoRe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resolve coreferences in the benchmark's raw documents (xCoRe)."
    )
    parser.add_argument(
        "--input-glob",
        default="data/**/raw-documents/**/*.txt",
        help="Glob (recursive) over the raw benchmark documents",
    )
    parser.add_argument(
        "--output-dir",
        default="data_coref",
        help="Where to write one coreference-resolved JSON file per document",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=10000,
        help="Skip documents longer than this (model context limit)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    coref_model = xCoRe()

    for doc_path in tqdm(sorted(glob(args.input_glob, recursive=True))):
        doc_name = os.path.splitext(os.path.basename(doc_path))[0]
        out_path = os.path.join(args.output_dir, f"{doc_name}.json")

        if os.path.exists(out_path):
            continue

        text = open(doc_path).read()
        if len(text) < args.max_chars:
            pred = coref_model.predict(text)
            json.dump(pred, open(out_path, "w"))
