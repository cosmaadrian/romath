#!/bin/bash
set -e
cd ..

# Evaluate a bunch of judges (ro prompt)
# meta-llama/Meta-Llama-3-70B-Instruct
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model meta-llama/Meta-Llama-3-70B-Instruct --output_dir predictions_judges/

# mistralai/Mixtral-8x7B-Instruct-v0.1
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model mistralai/Mixtral-8x7B-Instruct-v0.1 --output_dir predictions_judges/

# microsoft/Phi-3-mini-4k-instruct
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model microsoft/Phi-3-mini-4k-instruct --output_dir predictions_judges/

# Qwen/Qwen2-0.5B-Instruct
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-Math-7B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-7B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-1.5B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-0.5B-Instruct --output_dir predictions_judges/

# mistralai/Mathstral-7b-v0.1
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model mistralai/Mathstral-7b-v0.1 --output_dir predictions_judges/

# deepseek-ai/deepseek-math-7b-instruct
python evaluate_judge.py --prompt ro --dataset assets/romath-judge-train.csv --model deepseek-ai/deepseek-math-7b-instruct --output_dir predictions_judges/

#######################################################################################################
#######################################################################################################
#######################################################################################################

# Evaluate a bunch of judges (en prompt)
# meta-llama/Meta-Llama-3-70B-Instruct
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model meta-llama/Meta-Llama-3-70B-Instruct --output_dir predictions_judges/

# mistralai/Mixtral-8x7B-Instruct-v0.1
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model mistralai/Mixtral-8x7B-Instruct-v0.1 --output_dir predictions_judges/

# microsoft/Phi-3-mini-4k-instruct
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model microsoft/Phi-3-mini-4k-instruct --output_dir predictions_judges/

# Qwen/Qwen2-0.5B-Instruct
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-Math-7B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-7B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-1.5B-Instruct --output_dir predictions_judges/
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model Qwen/Qwen2-0.5B-Instruct --output_dir predictions_judges/

# mistralai/Mathstral-7b-v0.1
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model mistralai/Mathstral-7b-v0.1 --output_dir predictions_judges/

# deepseek-ai/deepseek-math-7b-instruct
python evaluate_judge.py --prompt en --dataset assets/romath-judge-train.csv --model deepseek-ai/deepseek-math-7b-instruct --output_dir predictions_judges/