#!/bin/bash
set -e
cd ..

FILES=(
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_bac_0_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_bac_5_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_comps_0_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_comps_5_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_bac_0_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_bac_5_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_comps_0_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_comps_5_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_bac_0_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_bac_5_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_comps_0_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_comps_5_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_bac_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_bac_5_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_comps_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_comps_5_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_bac_0_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_bac_5_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_comps_0_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_comps_5_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct_synthetic_0_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct_synthetic_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_synthetic_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_synthetic_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_synthetic_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_synthetic_5_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_synthetic_0_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_synthetic_5_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_synthetic_0_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_synthetic_5_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_synthetic_0_0.0.csv
./predictions/mistralai-Mixtral-8x7B-Instruct-v0.1_synthetic_5_0.0.csv
)

JUDGEINFO='--prompt_lang en --judge_model Qwen/Qwen2-7B-Instruct'
for filename in "${FILES[@]}"; do
    echo "Evaluating $filename"
    python evaluate.py --output results2/results-qwen2-7b/ --pred_file $filename $JUDGEINFO
done