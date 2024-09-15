#!/bin/bash
set -e
cd ..

FILES=(
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-bac-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-bac-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-bac-test_model_nllb-200-distilled-600M_0_0.0.csv
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-comps-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-comps-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/deepseek-ai-deepseek-math-7b-instruct_romath-comps-test_model_nllb-200-distilled-600M_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-bac-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-bac-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-bac-test_model_nllb-200-distilled-600M_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-comps-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-comps-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/mistralai-Mathstral-7b-v0.1_romath-comps-test_model_nllb-200-distilled-600M_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-comps-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-comps-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-comps-test_model_nllb-200-distilled-600M_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-bac-test_model_nllb-200-1_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-bac-test_model_nllb-200-3_0_0.0.csv
./predictions_translated/Qwen-Qwen2-Math-7B-Instruct_romath-bac-test_model_nllb-200-distilled-600M_0_0.0.csv
)

JUDGEINFO='--prompt_lang en --judge_model Qwen/Qwen2-7B-Instruct'
for filename in "${FILES[@]}"; do
    echo "Evaluating $filename"
    python evaluate.py --translated --output results-translated/ --pred_file $filename $JUDGEINFO
done