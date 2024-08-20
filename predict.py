"""
    Script that uses a model to make predictions on a given dataset.
    A model can make multiple predictions for a single problem to compute metrics such as pass@k, accuracy@k.
    Outputs a .csv file with the predictions.
"""
import pandas as pd
from collections import defaultdict
import numpy as np
np.random.seed(42)

import argparse
import datasets
import torch
import os
import tqdm
import pprint

from copy import deepcopy
from evaluate.utils import complete_prompts

import transformers
from transformers import pipeline
from transformers import AutoModelForCausalLM
from evaluate.prompts.prediction_prompt import PROMPT


parser = argparse.ArgumentParser(description='Predict on dataset')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'Model name')
parser.add_argument('--dataset', type = str, default = 'bac', help = 'Dataset name. (synthetic / bac / comps)')
parser.add_argument('--output', type = str, default = 'predictions/', help = 'Output folder.')

parser.add_argument('--shots', type = int, default = 0, help = 'Number of examples in the prompt.')
parser.add_argument('--k', type = int, default = 1, help = 'Number of predictions to make (for acc@k).')
parser.add_argument('--temperature', type = float, default = 0.0, help = 'Temperature of model.')
args = parser.parse_args()

print("Running predictions for", args.__dict__)

HF_TOKEN = os.environ.get('HF_TOKEN', None)

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

        content = f"{example['solution']}"
        if example['answer'] != 'Proof':
            content = f"{example['solution']}. Răspunsul final este: \\boxed{{{example['answer']}}}"

        shot_list.append({
            "role": "assistant",
            "content": content
        })

    final_template = system_prompt + shot_list + final_prompt
    return final_template


# TODO check if is PEFT model and use the correct model class
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    torch_dtype = torch.float16, # TODO? use 8-bit if available?
    trust_remote_code = True
)
tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)
text_generator = pipeline("text-generation", model = model, tokenizer = tokenizer)

# Load dataset
train_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'train')
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test')

# test_dataset = test_dataset.take(10)

outputs = defaultdict(list)

for i, example in enumerate(tqdm.tqdm(test_dataset, total = len(test_dataset), position = 0)):
    question = example['problem']
    solution = example['solution']

    messages = complete_prompts(PROMPT, problem_statement = question)
    messages = populate_few_shot(messages, train_dataset, args.shots)

    responses = []
    for j in tqdm.tqdm(range(args.k), position = 1, leave = i == len(test_dataset) - 1):
        response = text_generator(
            messages,
            do_sample = True, max_new_tokens = 2048, temperature = args.temperature
        )

        content = response[0]['generated_text'][-1]['content']
        outputs['idx'].append(example['idx'])
        outputs['model'].append(args.model)
        outputs['dataset'].append(args.dataset)
        outputs['domain'].append(example['domain'])
        outputs['temperature'].append(args.temperature)
        outputs['shots'].append(args.shots)

        outputs['problem'].append(question)
        outputs['solution'].append(solution)
        outputs['answer'].append(example['answer'])
        outputs['response'].append(content)

df = pd.DataFrame(outputs)

os.makedirs(args.output, exist_ok = True)
df.to_csv(f"{args.output}/{args.model.replace('/', '-')}_{args.dataset}_{args.shots}_{args.temperature}.csv", index = False)