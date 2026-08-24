# DORIAN — DORIS-augmented Analysis of Narratives

Code and results for **"Do Knowledge Graphs Influence LLMs in Entity Framing
Analysis?"** (Bulla, Fiumanò, Lazzari, Presutti), RAGE-KG 2026, co-located
with the 25th International Semantic Web Conference (ISWC).

We evaluate whether injecting frame-semantic knowledge structured as a
knowledge graph (KG) improves LLM performance on multilingual, fine-grained
**entity framing** — classifying how a news article portrays an entity
(protagonist/antagonist/innocent, and 22 finer-grained archetypes such as
*Victim*, *Guardian*, *Tyrant*) — building on the DORIS-ontology KG pipeline
from Fiumanò et al. and the
[multilingual entity framing benchmark](https://aclanthology.org/2025.findings-acl.17/)
of Mahmoud et al. (SemEval-2025 Task 10).

## Method

```
text ──► frame-semantic parsing ──► KG construction ──► triple filtering / verbalization ──► prompt ──► LLM ──► entity role
        (span-finder, LOME-style)     (DORIS ontology)      (salient / entity-centric)
```

We compare four prompting settings (Table 2 of the paper):

| Setting | Description |
|---|---|
| **Zero Shot** | text + entity only |
| **Triples** | + all salient triples from the document's KG |
| **Triples Filt.** | + only triples where the target entity fills a frame element |
| **Triples Filt. Verb.** | as above, verbalized into natural-language-like statements |

...against six LLMs served locally via [Ollama](https://ollama.com/), two
per family: `gemma4:26b`, `gemma4:e4b`, `mistral-small3.2:24b`,
`mistral:7b`, `qwen3:30b`, `qwen3:4b`. Each (model, setting, language) cell
is averaged over 3 runs. Beyond accuracy (precision/recall/F1/EMR), we
analyze *how* KG injection affects model behavior via prediction entropy
across runs and the Hellinger distance between baseline and KG-enhanced
prediction distributions.

## Repository layout

```
dorian/                core prompt-building library
  prompt.py               make_zeroshot_prompt / make_dorian_prompt
  prompts/*.jinja          the 4 prompt templates above
  queries/*.sparql         SPARQL queries against the KG (all-salient / entity-centric)
pipeline/               Section 4: text → KG (see pipeline/README.md)
llm_experiments/        Section 5: build prompts, run LLMs, normalize, evaluate
analysis/               Section 5.3: entropy + Hellinger distance (Tables 4 & 5)
results/                small result artifacts that directly back the paper's tables
  metrics.csv              precision/recall/F1-micro/F1-macro/EMR per (model, setting, run) — Tables 2 & 3
  predictions_union.jsonl  pooled per-entity predictions across runs — Tables 4 & 5
```

Raw/intermediate data (the benchmark itself, parsed frames, the constructed
KG, and per-run LLM prediction dumps) is **not** committed to this repo to
keep it small — see below for how to regenerate it.

## Reproducing

1. **Get the benchmark.** Obtain the Mahmoud et al. multilingual entity
   framing benchmark (SemEval-2025 Task 10) from its official source and
   place the raw documents/annotations under `data/` (expected layout:
   `data/<split>/<lang>/{raw-documents,subtask-1-documents,subtask-1-annotations.txt}`,
   matching what `pipeline/resolve_coref.py` and `llm_experiments/*` expect).
2. **Build the KG** — see [`pipeline/README.md`](pipeline/README.md).
3. **Build prompts** for each setting:
   ```bash
   pip install -e .
   for setting in baseline all_triples entity_triples verbalized_triples; do
       python llm_experiments/build_prompts.py --setting $setting \
           --input data/dev_subtask1.json \
           --output data/dev_subtask1_${setting}.json
   done
   ```
4. **Run the experiments** (requires [Ollama](https://ollama.com/) running
   locally with the 6 models above pulled):
   ```bash
   ./llm_experiments/run_experiments.sh
   ```
5. **Normalize + evaluate**:
   ```bash
   python llm_experiments/normalize_predictions.py --preds-dir results/predictions --out-dir results/predictions_normalized
   python llm_experiments/evaluate.py --normalized-dir results/predictions_normalized --out results/metrics.csv
   ```
6. **Entropy / Hellinger distance analysis** (Tables 4 & 5):
   ```bash
   python analysis/generate_prediction_unions.py
   jupyter nbconvert --execute analysis/entropy_hellinger.ipynb
   ```

Steps 3-6 reproduce `results/metrics.csv` and `results/predictions_union.jsonl`
as already committed here, so you can jump straight to step 6's notebook (or
`analysis/multilabel_stats.ipynb`, which reports the dataset's multi-label
statistics cited in Section 5.1) against the committed `results/` files
without rerunning the LLMs.

## Citation

```bibtex
@inproceedings{bulla2026dorian,
  title     = {Do Knowledge Graphs Influence LLMs in Entity Framing Analysis?},
  author    = {Bulla, Luana and Fiuman{\`o}, Beatrice and Lazzari, Nicolas and Presutti, Valentina},
  booktitle = {Proceedings of the Workshop on Retrieval-Augmented Generation Enabled by Knowledge Graphs (RAGE-KG 2026), co-located with ISWC 2026},
  year      = {2026}
}
```

## License

CC0 1.0 Universal — see [LICENSE](LICENSE).
