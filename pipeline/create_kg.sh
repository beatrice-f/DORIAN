#!/bin/bash
# Build the DORIS-ontology KG (Section 4) from unified per-document frame
# predictions, using SPARQL Anything's CONSTRUCT query.
#
# Usage: ./create_kg.sh <frames_dir> <kg_dir> [path_to_sparql_anything_jar]
#
# $1 is where the unified frame JSON files are (see unify_frames.py)
# $2 is where the resulting .nq named-graph files should go
# $3 (optional) path to the SPARQL Anything jar; see README.md for the
#    download link/version. Defaults to sparql-anything.jar next to this script.

BASEDIR=$(dirname "$0")
JAR=${3:-$BASEDIR/sparql-anything.jar}

if [ ! -f "$JAR" ]; then
    echo "SPARQL Anything jar not found at $JAR — see pipeline/README.md for the download link." >&2
    exit 1
fi

mkdir -p "$2"

TOTAL=$(ls "$1"*.json 2>/dev/null | wc -l)
echo "Found $TOTAL annotated documents"

I=0
for f in "$1"*.json; do
    FILENAME=$(basename -- "$f")
    FILENAME="${FILENAME%.*}"

    if [ ! -f "$2$FILENAME.nq" ]; then
        java -jar "$JAR" -q "$BASEDIR/predictions_to_kg.sparql" -v input="$f" -f nq > "$2$FILENAME.nq"
        printf "\r%d / %d" "$I" "$TOTAL"
    fi
    I=$((I+1))
done
