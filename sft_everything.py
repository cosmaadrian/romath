"""
    Script that fine-tunes a given model on a given dataset using PEFT.
    Saves the model in the output directory.
"""

import os
import math
import tqdm
import pprint
import argparse

import datasets
import transformers
from transformers import DataCollatorForLanguageModeling, Trainer

from evaluate.utils import complete_prompts
from evaluate.prompts.romanian_reasoning_prompt import PROMPT

from functools import partial

from peft import LoraConfig, TaskType

from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

HF_TOKEN = os.environ.get('HF_TOKEN', None)

parser = argparse.ArgumentParser(description='Fine-tune model using PEFT')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'HF Model name')
parser.add_argument('--output', type = str, default = 'checkpoints-sft/', help = 'Output folder.')

parser.add_argument('--batch_size', type = int, default = 16)

args = parser.parse_args()
print("Running fine-tuning with", args.__dict__)

os.environ["WANDB_PROJECT"] = "romath"
os.environ["WANDB_RUN_GROUP"] = 'sft'

run_slug = f'{args.model.replace("/", "-")}-sft'

def make_instruction(problem_statement, solution, answer, tokenizer):
    messages = complete_prompts(PROMPT, problem_statement = problem_statement)

    content = f"<raționament>{solution}</raționament>"
    if answer != 'Proof':
        content = f"<raționament>{solution}</raționament><răspuns>{answer}</răspuns>"

    label = {
        "role": "assistant",
        "content": content
    }

    messages = messages + [label]
    instruction_text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    return instruction_text

def format_instructions(batch, tokenizer):
    return tokenizer([
        make_instruction(batch['problem'][i], batch['solution'][i], batch['answer'][i], tokenizer)
        for i in range(len(batch['problem']))
    ], padding = 'longest', max_length = 1024, truncation = True)

model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    trust_remote_code = True,
)
model.enable_input_require_grads()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

dataset_names = ['bac', 'comps', 'synthetic']

train_datasets = []
test_datasets = []
for dataset in dataset_names:
    ds_train = datasets.load_dataset('cosmadrian/romath', dataset, split = 'train', token = HF_TOKEN)
    ds_test = datasets.load_dataset('cosmadrian/romath', dataset, split = 'test', token = HF_TOKEN)

    if 'synthetic' in dataset:
        ds_train = ds_train.shuffle(seed = 42).select(range(5000))
        ds_train = ds_train.map(lambda x: {'problem': x['problem'], 'solution': x['solution'], 'answer': x['solution']})
        ds_test = ds_test.shuffle(seed = 42).select(range(500))
        ds_test = ds_test.map(lambda x: {'problem': x['problem'], 'solution': x['solution'], 'answer': x['solution']})

    train_datasets.append(ds_train)
    test_datasets.append(ds_test)

train_dataset = datasets.concatenate_datasets(train_datasets).shuffle(seed = 42)
train_dataset = train_dataset.map(lambda x: format_instructions(x, tokenizer), batched = True)

test_dataset = datasets.concatenate_datasets(test_datasets).shuffle(seed = 42)
test_dataset = test_dataset.map(lambda x: format_instructions(x, tokenizer), batched = True)
# Load dataset

# Fine-tune model
training_args = SFTConfig(
    run_name = run_slug,
    report_to = 'wandb',

    output_dir = os.path.join(args.output, run_slug),
    overwrite_output_dir = True,
    optim = "adamw_torch_fused",
    max_seq_length = 1024,

    warmup_steps = 32,
    learning_rate = 2e-5,
    gradient_accumulation_steps = 16,
    gradient_checkpointing = False,

    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,

    num_train_epochs = 1,
    weight_decay = 0.01,

    bf16 = True,
    tf32 = True,

    save_total_limit = 1,

    eval_strategy = "epoch",
    save_strategy = "epoch",

    load_best_model_at_end = False,
    push_to_hub = False,
    logging_steps = 8,
    packing = False,
)

collator = DataCollatorForLanguageModeling(
    mlm = False,
    tokenizer = tokenizer,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    data_collator = collator,
)

trainer.train()
