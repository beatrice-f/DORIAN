"""
Render entity-framing prompts for one knowledge-injection setting (Section
5.1) over a benchmark split, querying the local KG endpoint (see
pipeline/README.md) for the three triple-based settings.

Usage:
    python build_prompts.py --setting baseline \
        --input data/dev_subtask1.json --output data/dev_subtask1_baseline.json

    python build_prompts.py --setting verbalized_triples \
        --input data/dev_subtask1.json --output data/dev_subtask1_verbalized_triples.json \
        --endpoint http://localhost:3000/kg/sparql
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dorian.prompt import make_dorian_prompt, make_zeroshot_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each entry maps a Table 2 setting to the jinja prompt template and (for the
# three graph-enhanced settings) the SPARQL query used to pull triples from
# the KG. "all_triples" queries all salient triples in the document;
# "entity_triples"/"verbalized_triples" restrict to triples where the target
# entity fills a frame element (Listing 2), the latter rendered verbalized.
SETTINGS = {
    "baseline": dict(
        template=REPO_ROOT / "dorian/prompts/baseline.jinja",
        query=None,
    ),
    "all_triples": dict(
        template=REPO_ROOT / "dorian/prompts/triples.jinja",
        query=REPO_ROOT / "dorian/queries/all_salient.sparql",
    ),
    "entity_triples": dict(
        template=REPO_ROOT / "dorian/prompts/triples.jinja",
        query=REPO_ROOT / "dorian/queries/entity_centric.sparql",
    ),
    "verbalized_triples": dict(
        template=REPO_ROOT / "dorian/prompts/verbalized_triples.jinja",
        query=REPO_ROOT / "dorian/queries/entity_centric.sparql",
    ),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    parser.add_argument(
        "--input", required=True, help="Benchmark split json, e.g. data/dev_subtask1.json"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        default="http://localhost:3000/kg/sparql",
        help="Fuseki SPARQL endpoint serving the KG (see pipeline/README.md); "
        "unused for --setting baseline",
    )
    args = parser.parse_args()

    cfg = SETTINGS[args.setting]
    annotations = json.load(open(args.input, encoding="utf-8"))

    for lang, docs in annotations.items():
        for doc_id, doc_content in tqdm(list(docs.items()), desc=f"{args.setting}/{lang}"):
            for ann in doc_content["annotations"]:
                if cfg["query"] is None:
                    ann["prompt"] = make_zeroshot_prompt(
                        str(cfg["template"]),
                        doc_content["text"],
                        (ann["start"], ann["end"]),
                    )
                else:
                    ann["prompt"] = make_dorian_prompt(
                        str(cfg["template"]),
                        str(cfg["query"]),
                        doc_content["text"],
                        (ann["start"], ann["end"]),
                        doc_id.split(".")[0],
                        args.endpoint,
                    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False)
    print(f"Saved to {args.output}")
