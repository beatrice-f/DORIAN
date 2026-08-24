#!/bin/bash
# Run all 6 paper models x 4 injection settings x 3 repeated runs (Table 2/3).
# Requires: Ollama running locally with the models below pulled, and
# data/dev_subtask1_<setting>.json already built via build_prompts.py.
#
# Usage: ./run_experiments.sh [setting ...]   (default: all four settings)
set -euo pipefail

cd "$(dirname "$0")/.."

MODELS=(
    "gemma4:26b-a4b-it-q4_K_M"
    "gemma4:e4b-it-q4_K_M"
    "mistral-small3.2:24b-instruct-2506-q4_K_M"
    "mistral:7b"
    "qwen3:30b-a3b-q4_K_M"
    "qwen3:4b-q4_K_M"
)

if [ "$#" -gt 0 ]; then
    SETTINGS=("$@")
else
    SETTINGS=(baseline all_triples entity_triples verbalized_triples)
fi

RUNS=(1 2 3)
DATA_DIR=data
PRED_DIR=results/predictions

for model in "${MODELS[@]}"; do
    ollama pull "$model"
    for setting in "${SETTINGS[@]}"; do
        mkdir -p "$PRED_DIR/$setting/$model"
        for run in "${RUNS[@]}"; do
            out="$PRED_DIR/$setting/$model/${setting}_${run}.json"
            if [ -f "$out" ]; then
                echo "skip (exists): $out"
                continue
            fi
            python llm_experiments/run_llm.py \
                --model "ollama/$model" \
                --data "$DATA_DIR/dev_subtask1_${setting}.json" \
                --output "$out"
        done
    done
done
