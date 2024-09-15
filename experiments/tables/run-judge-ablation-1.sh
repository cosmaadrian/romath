#!/bin/bash
set -e
cd ..

FILES=(
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_bac_0_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_comps_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_bac_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_comps_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_bac_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_comps_0_0.0.csv
)

JUDGEINFO='--prompt_lang en --judge_model mistralai/Mixtral-8x7B-Instruct-v0.1'
for filename in "${FILES[@]}"; do
    echo "Evaluating $filename"
    python evaluate.py --output results2/results-mixtral/ --pred_file $filename $JUDGEINFO
done
