import re
import os
import torch

import math
import datasets
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
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
parser.add_argument('--seed', type = int, default = 69)

args = parser.parse_args()
print("Running fine-tuning with", args.__dict__)

os.environ["WANDB_PROJECT"] = "romath"
os.environ["WANDB_RUN_GROUP"] = 'grpo'

run_slug = f'{args.model.replace("checkpoints-sft", "").replace("/", "-")}-grpo-{args.seed}'

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
                ds_train = ds_train.shuffle(seed = args.seed).select(range(200))

        train_datasets.append(ds_train)

    train_dataset = datasets.concatenate_datasets(train_datasets)
    
    if split == 'train':
        train_dataset = train_dataset.shuffle(seed = args.seed)

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
    return [1.0 if is_equivalent(r, a) else 0.0 for r, a in zip(extracted_responses, answer)]

def reward_strict_format(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<raționament>.*?<\/raționament><răspuns>.*?<\/răspuns>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL | re.MULTILINE) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    # https://arxiv.org/abs/2502.03373
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward

def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = sol

            answer_parsed = extract_xml_answer(content)

            is_correct = is_equivalent(answer_parsed, gold_parsed)

            # Apply cosine scaling based on length
            progress = len(content) / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards

    return cosine_scaled_reward

rewards = [
    reward_correctness,
    reward_strict_format,
    # get_repetition_penalty_reward(3, -1),
    # get_cosine_scaled_reward(
    #     min_value_wrong = 0.0, 
    #     max_value_wrong = -0.5, 
    #     min_value_correct = 0.5, 
    #     max_value_correct = 1.0
    # ),
]

training_args = GRPOConfig(
    output_dir = args.output + '/' + run_slug,
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
    gradient_accumulation_steps = 4,
    num_generations = 4,
    max_prompt_length = 256,
    max_completion_length = 256,
    num_train_epochs = 10,
    save_steps = 100,
    max_grad_norm = 0.1,
    log_on_each_node = False,
    use_vllm = True,
    vllm_gpu_memory_utilization = .5,
    vllm_device = "cuda:0",
    report_to = "wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = get_romath_questions(split = "train")
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = rewards,
    args = training_args,
    train_dataset = train_dataset,
    # peft_config = peft_config,
)
trainer.train()