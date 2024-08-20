"""
    Script that fine-tunes a given model on a given dataset using PEFT.
    Saves the model in the output directory.
"""

import os
import pprint
import argparse

import datasets
import transformers

from evaluate.utils import complete_prompts
from evaluate.prompts.prediction_prompt import PROMPT

from peft import LoraConfig, TaskType

from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from trl import SFTTrainer

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
    texts = []
    for i in range(len(sample['problem'])):
        messages = complete_prompts(PROMPT, problem_statement = sample['problem'][i])
        content = f"Soluția este:\n{sample['solution'][i]}"
        if sample['answer'] != 'Proof':
            content = f"Soluția este:\n{sample['solution'][i]}. Răspunsul final este: \\boxed{{{sample['answer'][i]}}}"

        label = {
            "role": "assistant",
            "content": content
        }

        messages = messages + [label]
        instruction_text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)

        texts.append(instruction_text)

    return texts

peft_config = LoraConfig(
    bias = "none",
    r = args.r,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,

    task_type = TaskType.CAUSAL_LM,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    args.model,
    token = HF_TOKEN,
    device_map = "auto",
    load_in_8bit=True,
    trust_remote_code = True,
)
model.enable_input_require_grads()

tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, token = HF_TOKEN)

# Load dataset
train_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'train', token = HF_TOKEN)
test_dataset = datasets.load_dataset('cosmadrian/romath', args.dataset, split = 'test', token = HF_TOKEN)

# Fine-tune model
training_args = SFTConfig(
    output_dir = os.path.join(args.output, run_slug),
    overwrite_output_dir = True,
    optim = "adamw_torch_fused",

    warmup_steps = 32,
    learning_rate = 2e-5,
    gradient_accumulation_steps = 1,
    gradient_checkpointing = False,

    per_device_train_batch_size = args.batch_size,
    per_device_eval_batch_size = args.batch_size,

    num_train_epochs = 2,
    weight_decay = 0.01,

    bf16=True,
    tf32=True,

    save_total_limit = 1,

    eval_strategy = "steps",
    eval_steps = 256,

    save_strategy = "steps",
    save_steps = 256,

    load_best_model_at_end = False,
    push_to_hub = False,

    run_name = run_slug,
    report_to = 'wandb',
    logging_steps = 16,
    packing = False,
)

# response_token_ids = tokenizer.encode('\nSoluția este: ', add_special_tokens = False)[2:]
collator = DataCollatorForCompletionOnlyLM(
    response_template = '\nSoluția este:\n',
    tokenizer = tokenizer
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    max_seq_length = 2048,
    formatting_func = lambda x: make_instruction(x, tokenizer),
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    peft_config = peft_config,
    data_collator = collator,
)

trainer.train()
