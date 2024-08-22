#!/bin/bash
set -e

# Model: microsoft/Phi-3-mini-4k-instruct (3.8B params)
python fine_tune.py --batch_size 4 --dataset comps --model microsoft/Phi-3-mini-4k-instruct

# TODO Needs to be run on A100 / H100!
# Model: microsoft/Phi-3-small-8k-instruct (7B params)
# python fine_tune.py --batch_size 4 --dataset bac --model microsoft/Phi-3-small-8k-instruct

# TODO Needs to be run on A100 / H100!
# Model: microsoft/Phi-3-medium-4k-instruct (14B params)
# python fine_tune.py --batch_size 4 --dataset comps --model microsoft/Phi-3-medium-4k-instruct

#############################################################################################################
#############################################################################################################
#############################################################################################################

# Model: Qwen/Qwen2-0.5B-Instruct
python fine_tune.py --batch_size 4 --dataset bac --model Qwen/Qwen2-0.5B-Instruct

# Model: Qwen/Qwen2-1.5B-Instruct
python fine_tune.py --batch_size 4 --dataset comps --model Qwen/Qwen2-1.5B-Instruct

# Model: Qwen/Qwen2-7B-Instruct
python fine_tune.py --batch_size 4 --dataset bac --model Qwen/Qwen2-7B-Instruct

# Model: Qwen/Qwen2-Math-7B-Instruct
python fine_tune.py --batch_size 4 --dataset comps --model Qwen/Qwen2-Math-7B-Instruct

#############################################################################################################
#############################################################################################################
#############################################################################################################

# Model: mistralai/Mathstral-7b-v0.1
python fine_tune.py --batch_size 4 --dataset bac --model mistralai/Mathstral-7b-v0.1

#############################################################################################################
#############################################################################################################
#############################################################################################################

# Model: deepseek-ai/deepseek-math-7b-instruct
python fine_tune.py --batch_size 4 --dataset comps --model deepseek-ai/deepseek-math-7b-instruct

#############################################################################################################
#############################################################################################################
#############################################################################################################

# Model: meta-llama/Meta-Llama-3-8B-Instruct
python fine_tune.py --batch_size 4 --dataset bac --model meta-llama/Meta-Llama-3-8B-Instruct

#############################################################################################################
#############################################################################################################
#############################################################################################################
# Romanian Models
#############################################################################################################
#############################################################################################################
#############################################################################################################

# Model: OpenLLM-Ro/RoMistral-7b-Instruct
python fine_tune.py --batch_size 4 --dataset comps --model OpenLLM-Ro/RoMistral-7b-Instruct

# Model: OpenLLM-Ro/RoLlama3-8b-Instruct
python fine_tune.py --batch_size 4 --dataset bac --model OpenLLM-Ro/RoLlama3-8b-Instruct
