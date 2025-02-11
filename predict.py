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
from evaluate.prompts.romanian_prediction_prompt import PROMPT


parser = argparse.ArgumentParser(description='Predict on dataset')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'Model name')
parser.add_argument('--dataset', type = str, default = 'bac', help = 'Dataset name. (synthetic / bac / comps)')
parser.add_argument('--output', type = str, default = 'predictions/', help = 'Output folder.')
parser.add_argument('--batch_size', type = int, default = 1, help = 'Batch size.')

parser.add_argument('--shots', type = int, default = 0, help = 'Number of examples in the prompt.')
parser.add_argument('--temperature', type = float, default = 0.0, help = 'Temperature of model.')
args = parser.parse_args()

print("Running predictions for", args.__dict__)

HF_TOKEN = os.environ.get('HF_TOKEN', None)

def compute_max_length_power_of_two(dataset, tokenizer):
    max_length = 0
    for sample in tqdm.tqdm(dataset, total = len(dataset), desc = f"Computing max length for cosmadrian/romath-{args.dataset}"):
        content = f"\n### Soluția este:\n{sample['solution']}"
        tokens = tokenizer.encode(content, add_special_tokens = False)
        max_length = max(max_length, len(tokens))
    return 2**(math.ceil(math.log(max_length, 2)))

def populate_few_shot(template, train_dataset, shots = 0):
    """
        Populates a few shot template with examples from the dataset. The same example for all models / setups.
    """
    template = deepcopy(template)

    if shots == 0:
        return template

    system_prompt = [template[0]]
    final_prompt = [template[-1]]

    sampled_idxs = np.random.choice(len(train_dataset), shots, replace = False)
    raw_shots = train_dataset.select(sampled_idxs)

    shot_list = []
    for i, example in enumerate(raw_shots):
        shot_list.append({
            "role": "user",
            "content":
            f"""Care este rezolvarea următoarei probleme?\n{example['problem']}"""
        })

        content = f"\n### Soluția este:\n{example['solution']}"
        if 'answer' in example and example['answer'] != 'Proof':
            content = f"\n### Soluția este:\n{example['solution']}. Răspunsul final este: \\boxed{{{example['answer']}}}"

        shot_list.append({
            "role": "assistant",
            "content": content
        })

    final_template = system_prompt + shot_list + final_prompt
    return final_template

is_fine_tuned = os.path.exists(args.model)

if is_fine_tuned:
    checkpoint_name = glob.glob(args.model + '/*')[0]

    model = AutoPeftModelForCausalLM.from_pretrained(
        checkpoint_name,
        token = HF_TOKEN,
        device_map = "auto",
        load_in_8bit = True,
        trust_remote_code = True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name, token = HF_TOKEN)
else:
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token = HF_TOKEN,
        device_map = "auto",
        load_in_8bit = True,
        trust_remote_code = True
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

# Load dataset
train_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'train')
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test')

outputs = defaultdict(list)

max_length = min(compute_max_length_power_of_two(test_dataset, tokenizer), 2048)
print("Computed max length:", max_length)

message_batch = []
for i, example in enumerate(tqdm.tqdm(test_dataset, total = len(test_dataset))):
    question = example['problem']
    solution = example['solution']
    messages = complete_prompts(PROMPT, problem_statement = question)
    messages = populate_few_shot(messages, train_dataset, args.shots)

    message_batch.append({
        'messages': messages,
        'example': example
    })

    if len(message_batch) == args.batch_size:
        all_messages = [b['messages'] for b in message_batch]
        tokens = tokenizer.apply_chat_template(
            all_messages,
            max_length = 2048,
            padding = 'max_length',
            return_tensors = 'pt',
            return_dict = True,
            truncation = True,
            add_generation_prompt = True
        )
        tokens = {k: v.to(model.device) for k, v in tokens.items()}

        with torch.no_grad():
            responses_ids = model.generate(
                temperature = args.temperature,
                do_sample = args.temperature > 0.0,
                max_new_tokens = max_length,
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

            outputs['idx'].append(example['idx'])
            outputs['model'].append(args.model)
            outputs['dataset'].append(args.dataset)
            outputs['domain'].append(example['domain'])
            outputs['temperature'].append(args.temperature)
            outputs['shots'].append(args.shots)
            outputs['fine-tuned'].append(is_fine_tuned)

            outputs['problem'].append(example['problem'])
            outputs['solution'].append(example['solution'])
            if 'answer' in example:
                outputs['answer'].append(example['answer'])
            outputs['response'].append(content)

        message_batch = []

df = pd.DataFrame(outputs)

model_name = args.model.replace('/', '-')
if is_fine_tuned:
    model_name = os.path.basename(args.model)

os.makedirs(args.output, exist_ok = True)
df.to_csv(f"{args.output}/{model_name}_{args.dataset}_{args.shots}_{args.temperature}.csv", index = False)