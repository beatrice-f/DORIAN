# KG construction pipeline (Section 4)

Turns raw benchmark documents into the DORIS-ontology knowledge graph used by
the KG-enhanced prompting settings in `llm_experiments/`.

```
raw text
  │  frame-semantic parsing (span-finder)
  ▼
frames/*.json          (one JSONL file per document: per-sentence frame predictions)
  │  pipeline/unify_frames.py
  ▼
frames/*.json          (unified: one JSON object per document)
  │  pipeline/create_kg.sh  (SPARQL Anything CONSTRUCT, pipeline/predictions_to_kg.sparql)
  ▼
kg/*.nq                (one named graph per document, DORIS ontology)
  │  tdb2.tdbloader + fuseki-server
  ▼
SPARQL endpoint (http://localhost:3000/kg/sparql), queried by dorian/prompt.py
```

Coreference resolution (`resolve_coref.py`) is a separate, optional
preprocessing pass over the raw documents; it is not currently wired into the
frame-parsing → KG steps above (its output in `data_coref/` is not consumed
downstream in this repo) but is kept here since it was part of the original
pipeline development.

## 1. Setup

- **Frame-semantic parsing**: this project used
  [n28div/span-finder](https://github.com/n28div/span-finder) (a span-based
  frame-semantic parser, in the spirit of LOME) to annotate each document
  with FrameNet frames and frame elements. Set it up per its own
  instructions and run it over the raw benchmark documents (one JSONL output
  file per document, each line a per-sentence prediction) into a `frames/`
  directory here.
- **SPARQL Anything** (KG construction): download the `sparql-anything-*.jar`
  release matching the version you want from
  [SPARQL-Anything releases](https://github.com/SPARQL-Anything/sparql.anything/releases)
  and place it as `pipeline/sparql-anything.jar` (or pass its path as the
  third argument to `create_kg.sh`). This repo used v1.1.0.
- **Apache Jena / Fuseki** (triple store to serve the KG for querying):
  download [Apache Jena](https://jena.apache.org/download/) and
  [Apache Jena Fuseki](https://jena.apache.org/download/) (this repo used
  6.0.0) and extract them locally, e.g. as `apache-jena-6.0.0/` and
  `apache-jena-fuseki-6.0.0/` next to this repo.

## 2. Build the KG

```bash
# 1. Frame-semantic parsing (external — see span-finder above), producing frames/*.json (per-sentence)

# 2. Merge per-sentence predictions into per-document frame files
python pipeline/unify_frames.py --input-dir frames --output-dir frames

# 3. Optional: coreference resolution over the raw benchmark documents
python pipeline/resolve_coref.py --input-glob "data/**/raw-documents/**/*.txt" --output-dir data_coref

# 4. Construct the KG (one .nq named-graph file per document)
mkdir -p kg
pipeline/create_kg.sh frames/ kg/ pipeline/sparql-anything.jar

# 5. (optional) flatten the KG to a tsv for quick inspection
python pipeline/kg_to_csv.py
```

## 3. Serve the KG

```bash
apache-jena-6.0.0/bin/tdb2.tdbloader --loc tdb_kg kg/*.nq
apache-jena-fuseki-6.0.0/fuseki-server --loc tdb_kg/ --port 3000 /kg
```

The KG is now queryable at `http://localhost:3000/kg/sparql`, the default
`--endpoint` used by `llm_experiments/build_prompts.py`.

## Known gap

The paper's salient-frame filtering (~617 frames selected via a SPARQL query
against Framester by semantic type of their frame elements — see the paper's
Listing 1) is not currently implemented as a standalone step in this
pipeline; the two SPARQL queries actually used by `dorian/queries/` restrict
by entity mention (Listing 2) and join Framester for canonical frame/role
labels, but do not apply the semantic-type salience filter. If you need to
reproduce that filtering exactly, you'll need to add it as an extra step
(e.g. filtering `frames/*.json` against the Listing-1 query's frame list
before running `create_kg.sh`).
