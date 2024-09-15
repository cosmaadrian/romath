#!/bin/bash
set -e
cd ..

FILES=(
./predictions/deepseek-ai-deepseek-math-7b-instruct_bac_0_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct_bac_5_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct_comps_0_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct_comps_5_0.0.csv
./predictions/deepseek-ai-deepseek-math-7b-instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_bac_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_bac_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_comps_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-70B-Instruct_comps_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_bac_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_bac_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_comps_0_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct_comps_5_0.0.csv
./predictions/meta-llama-Meta-Llama-3-8B-Instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_bac_0_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_bac_5_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct-bac-8-32-0.1_bac_0_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_comps_0_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct_comps_5_0.0.csv
./predictions/microsoft-Phi-3-mini-4k-instruct-comps-8-32-0.1_comps_0_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_bac_0_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_bac_5_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1-bac-8-32-0.1_bac_0_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_comps_0_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1_comps_5_0.0.csv
./predictions/mistralai-Mathstral-7b-v0.1-comps-8-32-0.1_comps_0_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_synthetic_0_0.0.csv
./predictions/OpenLLM-Ro-RoLlama3-8b-Instruct_synthetic_5_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_synthetic_0_0.0.csv
./predictions/OpenLLM-Ro-RoMistral-7b-Instruct_synthetic_5_0.0.csv
./predictions/Qwen-Qwen2-0.5B-Instruct_synthetic_0_0.0.csv
./predictions/Qwen-Qwen2-0.5B-Instruct_synthetic_5_0.0.csv
./predictions/Qwen-Qwen2-1.5B-Instruct_synthetic_0_0.0.csv
./predictions/Qwen-Qwen2-1.5B-Instruct_synthetic_5_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_synthetic_0_0.0.csv
./predictions/Qwen-Qwen2-7B-Instruct_synthetic_5_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_synthetic_0_0.0.csv
./predictions/Qwen-Qwen2-Math-7B-Instruct_synthetic_5_0.0.csv
)

JUDGEINFO='--prompt_lang en --judge_model Qwen/Qwen2-7B-Instruct'
for filename in "${FILES[@]}"; do
    echo "Evaluating $filename"
    python evaluate.py --output results2/results-qwen2-7b/ --pred_file $filename $JUDGEINFO
done