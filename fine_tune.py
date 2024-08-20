"""
    Script that fine-tunes a given model on a given dataset using PEFT.
    Saves the model in the output directory.
"""

import os
import torch
import pprint
import argparse

import transformers
import datasets

from evaluate.utils import complete_prompts
from evaluate.prompts.prediction_prompt import PROMPT

from peft import get_peft_model, LoraConfig, TaskType
from transformers import Trainer, TrainingArguments

HF_TOKEN = os.environ.get('HF_TOKEN', None)

parser = argparse.ArgumentParser(description='Fine-tune model using PEFT')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'Model name')
parser.add_argument('--dataset', type = str, default = 'bac', help = 'Dataset name. (synthetic / bac / comps)')
parser.add_argument('--output', type = str, default = 'checkpoints/', help = 'Output folder.')

parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--r', type = int, default = 8)
parser.add_argument('--lora_alpha', type = int, default = 32)
parser.add_argument('--lora_dropout', type = float, default = 0.1)

args = parser.parse_args()
print("Running fine-tuning with", args.__dict__)

os.environ["WANDB_PROJECT"] = "romath"

run_slug = f'{args.model.replace("/", "-")}-{args.dataset}-{args.r}-{args.lora_alpha}-{args.lora_dropout}'

def make_instruction(sample, tokenizer):
    messages = complete_prompts(PROMPT, problem_statement = sample['problem'])
    content = f"{sample['solution']}"
    if sample['answer'] != 'Proof':
        content = f"{sample['solution']}. Răspunsul final este: \\boxed{{{sample['answer']}}}"

    label = [{
        "role": "assistant",
        "content": content
    }]

    input_tokens = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        max_tokens = 2048,
        return_tensors = 'pt',
        return_dict = True,
        return_token_type_ids = True,
        padding = 'max_length',
        truncation = True,
    )

    label_tokens = tokenizer.apply_chat_template(
        label,
        add_generation_prompt = False,
        max_tokens = 1024,
        return_dict = True,
        return_tensors = 'pt',
        padding = 'max_length',
        truncation = True,
    )

    return {
        'input_ids': input_tokens['input_ids'][0],
        'attention_mask': input_tokens['attention_mask'][0],
        'labels': label_tokens['input_ids'][0],
    }

peft_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM,
    inference_mode = False,
    r = args.r,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    torch_dtype = torch.float16, # TODO? use 8-bit if available?
    trust_remote_code = True
)

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Load dataset
train_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'train')
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test')

#################
train_dataset = train_dataset.take(128)
test_dataset = test_dataset.take(128)
#################

# Prepare data
train_dataset = train_dataset.map(lambda x: make_instruction(x, tokenizer))
test_dataset = test_dataset.map(lambda x: make_instruction(x, tokenizer))

# Fine-tune model
training_args = TrainingArguments(
    output_dir = os.path.join(args.output, run_slug),

    warmup_steps = 128,
    learning_rate = 2e-5,

    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,

    num_train_epochs = 2,
    weight_decay = 0.01,

    eval_strategy = "steps",
    eval_steps = 128,

    save_strategy = "steps",
    save_steps = 128,

    load_best_model_at_end = False,
    push_to_hub = False,

    run_name = run_slug,
    report_to = 'wandb',
    logging_steps = 32,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)

trainer.train()