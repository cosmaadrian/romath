#!/bin/bash
set -e
cd ..

T=0.0
BS=32

# deepseek-ai-deepseek-math-7b-instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/deepseek-ai-deepseek-math-7b-instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model deepseek-ai/deepseek-math-7b-instruct --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model deepseek-ai/deepseek-math-7b-instruct --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# meta-llama-Meta-Llama-3-8B-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/meta-llama-Meta-Llama-3-8B-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model meta-llama/Meta-Llama-3-8B-Instruct --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model meta-llama/Meta-Llama-3-8B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# microsoft-Phi-3-mini-4k-instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/microsoft-Phi-3-mini-4k-instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model microsoft/Phi-3-mini-4k-instruct --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model microsoft/Phi-3-mini-4k-instruct --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# mistralai-Mathstral-7b-v0.1-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/mistralai-Mathstral-7b-v0.1-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model mistralai/Mathstral-7b-v0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model mistralai/Mathstral-7b-v0.1 --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# Qwen-Qwen2-0.5B-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/Qwen-Qwen2-0.5B-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model Qwen/Qwen2-0.5B-Instruct --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model Qwen/Qwen2-0.5B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# Qwen-Qwen2-1.5B-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/Qwen-Qwen2-1.5B-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune zero-shot
python predict.py --dataset comps --model Qwen/Qwen2-1.5B-Instruct --output predictions/ --temperature $T  --batch_size $BS

# no-fine-tune 5-shot
python predict.py --dataset comps --model Qwen/Qwen2-1.5B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size $BS

#######################################################
#######################################################
#######################################################

# Qwen-Qwen2-7B-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/Qwen-Qwen2-7B-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune zero-shot
python predict.py --dataset comps --model Qwen/Qwen2-7B-Instruct --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model Qwen/Qwen2-7B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size 8

#######################################################
#######################################################
#######################################################

# Qwen-Qwen2-Math-7B-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/Qwen-Qwen2-Math-7B-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune zero-shot
python predict.py --dataset comps --model Qwen/Qwen2-Math-7B-Instruct --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model Qwen/Qwen2-Math-7B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size 8

#######################################################
#######################################################
#######################################################

# OpenLLM-Ro-RoLlama3-8b-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/OpenLLM-Ro-RoLlama3-8b-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune zero-shot
python predict.py --dataset comps --model OpenLLM-Ro/RoLlama3-8b-Instruct --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model OpenLLM-Ro/RoLlama3-8b-Instruct --output predictions/ --temperature $T --shots 5  --batch_size 8

#######################################################
#######################################################
#######################################################

# OpenLLM-Ro-RoMistral-7b-Instruct-comps-8-32-0.1
# fine-tuned, zero-shot
python predict.py --dataset comps --model checkpoints/OpenLLM-Ro-RoMistral-7b-Instruct-comps-8-32-0.1 --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune zero-shot
python predict.py --dataset comps --model OpenLLM-Ro/RoMistral-7b-Instruct --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model OpenLLM-Ro/RoMistral-7b-Instruct --output predictions/ --temperature $T --shots 5  --batch_size 8

#######################################################
#######################################################
#######################################################

# meta-llama/Meta-Llama-3-70B-Instruct
# no-fine-tune zero-shot
python predict.py --dataset comps --model meta-llama/Meta-Llama-3-70B-Instruct --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model meta-llama/Meta-Llama-3-70B-Instruct --output predictions/ --temperature $T --shots 5  --batch_size 8

#######################################################
#######################################################
#######################################################

# mistralai/Mixtral-8x7B-Instruct-v0.1
# no-fine-tune zero-shot
python predict.py --dataset comps --model mistralai/Mixtral-8x7B-Instruct-v0.1 --output predictions/ --temperature $T  --batch_size 8

# no-fine-tune 5-shot
python predict.py --dataset comps --model mistralai/Mixtral-8x7B-Instruct-v0.1 --output predictions/ --temperature $T --shots 5  --batch_size 8