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

from evaluate.utils import complete_prompts
from evaluate.prompts.prediction_prompt import PROMPT

from functools import partial

from peft import LoraConfig, TaskType

from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

HF_TOKEN = os.environ.get('HF_TOKEN', None)

parser = argparse.ArgumentParser(description='Fine-tune model using PEFT')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-1.5B-Instruct', help = 'HF Model name')
parser.add_argument('--dataset', type = str, default = 'bac', help = 'Dataset name. (synthetic | bac | comps)')
parser.add_argument('--output', type = str, default = 'checkpoints/', help = 'Output folder.')

parser.add_argument('--batch_size', type = int, default = 16)
parser.add_argument('--r', type = int, default = 8)
parser.add_argument('--lora_alpha', type = int, default = 32)
parser.add_argument('--lora_dropout', type = float, default = 0.1)

args = parser.parse_args()
print("Running fine-tuning with", args.__dict__)

os.environ["WANDB_PROJECT"] = "romath"
os.environ["WANDB_RUN_GROUP"] = args.model.split("/")[0] + "-" + args.dataset

run_slug = f'{args.model.replace("/", "-")}-{args.dataset}-{args.r}-{args.lora_alpha}-{args.lora_dropout}'

def make_instruction(problem_statement, solution, answer, tokenizer):
    messages = complete_prompts(PROMPT, problem_statement = problem_statement)

    content = f"\n### Soluția este:\n{solution}"
    if answer != 'Proof':
        content = f"\n### Soluția este:\n{solution}. Răspunsul final este: \\boxed{{{answer}}}"

    label = {
        "role": "assistant",
        "content": content
    }

    messages = messages + [label]
    instruction_text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
    return instruction_text

def format_instructions(batch, tokenizer):
    return [
        make_instruction(batch['problem'][i], batch['solution'][i], batch['answer'][i], tokenizer)
        for i in range(len(batch['problem']))
    ]

peft_config = LoraConfig(
    bias = "none",
    r = args.r,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,

    task_type = TaskType.CAUSAL_LM,
    target_modules = 'all-linear'
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    load_in_8bit = True,
    trust_remote_code = True,
)
model.enable_input_require_grads()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN, padding_size = 'left' if 'mistralai' in args.model else 'right')

# Load dataset
train_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'train', token = HF_TOKEN)
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test', token = HF_TOKEN)

def compute_max_length_power_of_two(dataset, tokenizer):
    max_length = 0
    for sample in tqdm.tqdm(dataset, total = len(dataset), desc = f"Computing max length for cosmadrian/romath-{args.dataset}"):
        instruction = make_instruction(
            sample['problem'],
            sample['solution'],
            sample['answer'],
            tokenizer
        )
        max_length = max(max_length, len(instruction))
    return 2**(math.ceil(math.log(max_length, 2)))

# Fine-tune model
training_args = SFTConfig(
    run_name = run_slug,
    report_to = 'wandb',

    output_dir = os.path.join(args.output, run_slug),
    overwrite_output_dir = True,
    optim = "adamw_torch_fused",
    max_seq_length = min(compute_max_length_power_of_two(train_dataset, tokenizer), 2048),

    warmup_steps = 32,
    learning_rate = 2e-5,
    gradient_accumulation_steps = 16,
    gradient_checkpointing = False,

    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,

    num_train_epochs = 2,
    weight_decay = 0.01,

    bf16 = True,
    tf32 = True,

    save_total_limit = 1,

    eval_strategy = "steps",
    eval_steps = 256,

    save_strategy = "steps",
    save_steps = 256,

    load_best_model_at_end = False,
    push_to_hub = False,
    logging_steps = 8,
    packing = False,
)

collator = DataCollatorForCompletionOnlyLM(
    response_template = '### Soluția este:\n',
    tokenizer = tokenizer
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    formatting_func = partial(format_instructions, tokenizer = tokenizer),

    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    peft_config = peft_config,
    data_collator = collator,
)

trainer.train()
