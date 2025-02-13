"""
    Script that uses a model to make predictions on a given dataset.
    A model can make multiple predictions for a single problem to compute metrics such as pass@k, accuracy@k.
    Outputs a .csv file with the predictions.
"""
import pandas as pd
import glob
from collections import defaultdict
import numpy as np
np.random.seed(42)

import argparse
import datasets
import torch
import os
import tqdm
import pprint
import math

from copy import deepcopy
from evaluate.utils import complete_prompts
from peft import AutoPeftModelForCausalLM

import transformers
from transformers import AutoModelForCausalLM
from evaluate.prompts.romanian_reasoning_prompt import PROMPT
from evaluate.math_equivalence import is_equivalent

def extract_answer(response):
    if '<răspuns>' not in response or '</răspuns>' not in response:
        return response.split('<raționament>')[-1].split('</raționament>')[0].strip()

    answer = response.split('<răspuns>')[-1]
    answer = answer.split('</răspuns>')[0]

    return answer.strip()

parser = argparse.ArgumentParser(description='Predict on dataset')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'Model name')
parser.add_argument('--dataset', type = str, default = 'bac', help = 'Dataset name. (synthetic / bac / comps)')
parser.add_argument('--output', type = str, default = 'predictions/', help = 'Output folder.')
parser.add_argument('--batch_size', type = int, default = 1, help = 'Batch size.')
parser.add_argument('--temperature', type = float, default = 0.0, help = 'Temperature of model.')
args = parser.parse_args()

print("Running predictions for", args.__dict__)

HF_TOKEN = os.environ.get('HF_TOKEN', None)

is_fine_tuned = os.path.exists(args.model)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    trust_remote_code = True
)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Load dataset
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test')
# limit test dataset to only verifiable answers
test_dataset = test_dataset.filter(lambda example: example['answer'] != 'Proof')

outputs = defaultdict(list)
message_batch = []
for i, example in enumerate(tqdm.tqdm(test_dataset, total = len(test_dataset))):
    question = example['problem']
    solution = example['solution']
    messages = complete_prompts(PROMPT, problem_statement = question)

    message_batch.append({
        'messages': messages,
        'example': example
    })

    if len(message_batch) == args.batch_size:
        all_messages = [b['messages'] for b in message_batch]
        tokens = tokenizer.apply_chat_template(
            all_messages,
            max_length = 2048,
            padding = 'longest',
            return_tensors = 'pt',
            return_dict = True,
            truncation = True,
            add_generation_prompt = True
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            responses_ids = model.generate(
                max_new_tokens = 512,
                do_sample = args.temperature > 0.0,
                temperature = args.temperature if args.temperature > 0.0 else None,
                top_p = 0.9 if args.temperature > 0.0 else None,
                min_p = 0.1 if args.temperature > 0.0 else None,
                top_k = None,
                pad_token_id = tokenizer.eos_token_id,
                **tokens
            )

        # remove the prompt part from the response_ids, keep only the response
        responses_ids = responses_ids[:, tokens['input_ids'].shape[1]:]

        responses = tokenizer.batch_decode(
            responses_ids,
            skip_special_tokens = True,
            clean_up_tokenization_spaces = True,
        )

        for j in range(args.batch_size):
            content = responses[j]
            example = message_batch[j]['example']

            # print('Predicted:', content, "Extracted:", extract_answer(content))
            # print(example['answer'])
            is_correct = is_equivalent(extract_answer(content), example['answer'])

            outputs['idx'].append(example['idx'])
            outputs['model'].append(args.model)
            outputs['dataset'].append(args.dataset)
            outputs['domain'].append(example['domain'])
            outputs['temperature'].append(args.temperature)
            outputs['shots'].append(0)
            outputs['fine-tuned'].append(True)

            outputs['problem'].append(example['problem'])
            outputs['solution'].append(example['solution'])
            if 'answer' in example:
                outputs['answer'].append(example['answer'])
            else:
                outputs['answer'].append(None)

            outputs['response'].append(content)
            outputs['is_correct'].append(is_correct)
        message_batch = []

df = pd.DataFrame(outputs)

model_name = args.model.replace('/', '-')
if is_fine_tuned:
    checkpoint_idx = args.model.split('checkpoint-')[-1]
    is_sft = args.model.startswith('checkpoints-sft')
    if 'Llama-3.2-1B' in args.model: model_name = 'Llama-3.2-1B' + ('-sft' if is_sft else '-grpo') + f'-{checkpoint_idx}'
    if 'Llama-3.2-3B' in args.model: model_name = 'Llama-3.2-2B' + ('-sft' if is_sft else '-grpo') + f'-{checkpoint_idx}'
    if 'Qwen2-1.5B' in args.model: model_name = 'Qwen2-1.5B' + ('-sft' if is_sft else '-grpo') + f'-{checkpoint_idx}'
    if 'Qwen2-0B' in args.model: model_name = 'Qwen2-0B' + ('-sft' if is_sft else '-grpo') + f'-{checkpoint_idx}'

os.makedirs(args.output, exist_ok = True)
df.to_csv(f"{args.output}/{model_name}_{args.dataset}.csv", index = False)