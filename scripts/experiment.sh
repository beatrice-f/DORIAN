#!/bin/bash

#for model in "gemma4:e4b-it-q4_K_M" "qwen3.5:9b-q4_K_M" "olmo-3:7b-instruct-q4_K_M" "gemma4:26b-a4b-it-q4_K_M" "qwen3.5:27b-q4_K_M" "olmo-3:32b-think-q4_K_M" "mistral:7b" "mistral-small3.2:24b-instruct-2506-q4_K_M" ;
for model in "qwen3:8b" "qwen3:30b-a3b-q4_K_M" ;
do
    mkdir -p predictions/$model/
    ollama pull $model

    for run in 1 2 3;
    do
        # baseline
        python baseline_experiments/llm_baseline.py --model ollama/$model --output predictions/$model/baseline_$run.json
        python baseline_experiments/llm_baseline_ICL.py --model ollama/$model --output predictions/$model/baseline_icl_$run.json
        # python baseline_experiments/llm_baseline_with_examples.py --model ollama/$model --output predictions/$model/baseline_examples_$run.json

        # RAG
        # python RAG/rag_oneshot.py --model ollama/$model --output predictions/$model/rag_oneshot_$run.json
        # python RAG/rag_fewshot.py --model ollama/$model --output predictions/$model/rag_fewshot_$run.json
    done
done
