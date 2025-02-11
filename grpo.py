import re
import os
import torch
import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from evaluate.math_equivalence import is_equivalent

from evaluate.prompts.romanian_reasoning_prompt import PROMPT
SYSTEM_PROMPT = PROMPT[0]['content']

import argparse

HF_TOKEN = os.environ.get('HF_TOKEN', None)

parser = argparse.ArgumentParser(description='GRPO')
parser.add_argument('--model', type = str, default = 'Qwen/Qwen2-0.5B-Instruct', help = 'HF Model name')
parser.add_argument('--output', type = str, default = 'checkpoints-grpo/', help = 'Output folder.')
parser.add_argument('--batch_size', type = int, default = 16)

args = parser.parse_args()
print("Running fine-tuning with", args.__dict__)

os.environ["WANDB_PROJECT"] = "romath"
os.environ["WANDB_RUN_GROUP"] = 'grpo'

run_slug = f'{args.model.replace("/", "-")}-grpo'

def extract_xml_answer(text: str) -> str:
    answer = text.split("<răspuns>")[-1]
    answer = answer.split("</răspuns>")[0]
    return answer.strip()

def get_romath_questions(split = "train") -> Dataset:
    dataset_names = ['bac', 'comps', 'synthetic']

    train_datasets = []
    for dataset_name in dataset_names:
        ds_train = datasets.load_dataset('cosmadrian/romath', dataset_name, split = split, token = HF_TOKEN)

        if dataset_name != 'synthetic':
            ds_train = ds_train.filter(lambda example: example['answer'] != 'Proof')
        else:
            ds_train = ds_train.rename_column("solution", "answer")
            if split == "train":
                ds_train = ds_train.shuffle(seed = 42).select(range(200))

        train_datasets.append(ds_train)

    train_dataset = datasets.concatenate_datasets(train_datasets)
    
    if split == 'train':
        train_dataset = train_dataset.shuffle(seed = 42)

    train_dataset = train_dataset.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['problem']}
        ],
        'answer': x['answer']
    })

    return train_dataset

# Reward functions
def reward_correctness(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [4.0 if is_equivalent(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

def reward_strict_format(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<raționament>.*?<\/raționament><răspuns>.*?<\/răspuns>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def reward_keyword_count(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    
    rewards = [
        0.05 * int(r.count('<raționament>') == 1) + 0.05 * int(r.count('</raționament>') == 1) + 0.05 * int(r.count('<răspuns>') == 1) + 0.05 * int(r.count('</răspuns>') == 1)
        for r in responses
    ]
    return rewards

def reward_solution_length(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]["content"] for completion in completions]
    rewards = [float(len(r)) / 512 for r in responses]
    return rewards

training_args = GRPOConfig(
    output_dir = args.output,
    run_name = run_slug,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = 'cosine',
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = args.batch_size,
    gradient_accumulation_steps = 16,
    num_generations = 8,
    max_prompt_length = 256,
    max_completion_length = 128,
    num_train_epochs = 10,
    save_steps = 100,
    max_grad_norm = 0.1,
    log_on_each_node = False,
    use_vllm = True,
    vllm_gpu_memory_utilization = .3,
    vllm_device = "cuda:0",
    report_to = "wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')
tokenizer.pad_token = tokenizer.eos_token

train_dataset = get_romath_questions(split = "train")
test_dataset = get_romath_questions(split = "test")
# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs=[
        reward_correctness,
        reward_strict_format,
        reward_keyword_count,
        # reward_solution_length
    ],
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
)
trainer.train()