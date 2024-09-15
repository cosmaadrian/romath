import argparse
import tqdm
import numpy as np
import os
import datasets

import pandas as pd
from evaluate.solution_judge import SolutionJudge


parser = argparse.ArgumentParser(description = 'Use the judge to get scores.')
parser.add_argument('--pred_file', type = str, help = 'Input path for the .csv files with model-provided solutions.')
parser.add_argument('--judge_model', default = 'Qwen/Qwen2-1.5B-Instruct', type = str, help = 'Hugging Face model name for the judge.')
parser.add_argument('--prompt_lang', default = 'ro', type = str, help = 'Judge system prompt language.')
parser.add_argument('--output', default = 'results/', type = str, help = 'Output folder path for the .csv files with scores.')
parser.add_argument('--translated', action = 'store_true', help = 'Is it translated?')
args = parser.parse_args()

HF_TOKEN = os.environ.get('HF_TOKEN', None)

print("Running evaluation for", args.__dict__)

prediction_df = pd.read_csv(args.pred_file)
prediction_df['solution'] = prediction_df['solution'].astype(str)
prediction_df['response'] = prediction_df['response'].astype(str)

if 'answer' in prediction_df.columns:
    prediction_df['answer'] = prediction_df['answer'].astype(str)
else:
    prediction_df['answer'] = prediction_df['solution']

model_name = prediction_df['model'].unique()[0]
dataset = prediction_df['dataset'].unique()[0]
shots = prediction_df['shots'].unique()[0]
temperature = prediction_df['temperature'].unique()[0]

output_filename = f'{args.output}/{model_name.replace("/", "-")}_{dataset}_{shots}_{temperature}.csv'

if os.path.exists(output_filename):
    print(f'!!Output file "{output_filename}" already exists, skipping.')
    exit()

judge = SolutionJudge(args.judge_model, prompt = args.prompt_lang)

dataset_name = dataset

# TODO this is a hack, remove it later!
if args.translated:
    path = '/export/home/acs/prof/ioan_adrian.cosma/romath-competitions/translated/'

    if dataset_name == 'bac':
        path += 'romath-bac-test_model_nllb-200-3.3B.csv'
    else:
        path += 'romath-comps-test_model_nllb-200-3.3B.csv'

    translated_df = pd.read_csv(path)

    translated_df = translated_df[['idx', 'translated_problem_unchanged_math', 'translated_solution_unchanged_math']]
    prediction_df = pd.merge(prediction_df, translated_df, on = 'idx')
    prediction_df['problem'] = prediction_df['translated_problem_unchanged_math']
    prediction_df['solution'] = prediction_df['translated_solution_unchanged_math']
else:
    test_dataset = datasets.load_dataset('cosmadrian/romath', dataset_name, split = 'test')

    problem_dict = {row['idx']: row['problem'] for row in test_dataset}
    solution_dict = {row['idx']: row['solution'] for row in test_dataset}

    good_problems = pd.DataFrame(problem_dict.items(), columns = ['idx', 'problem'])
    good_solutions = pd.DataFrame(solution_dict.items(), columns = ['idx', 'solution'])

    prediction_df = prediction_df.drop(columns = ['problem', 'solution'])
    prediction_df = pd.merge(prediction_df, good_problems, on = 'idx')
    prediction_df = pd.merge(prediction_df, good_solutions, on = 'idx')

prediction_df['judge_pred'] = None

for i, row in tqdm.tqdm(prediction_df.iterrows(), total = len(prediction_df)):
    prediction_df.loc[i, 'judge_pred'] = judge.evaluate(
        question = row['problem'],
        true = row['solution'] if row['answer'] == 'Proof' else row['answer'],
        prediction = row['response'],
        has_single_answer = row['answer'] != 'Proof'
    )

prediction_df = prediction_df[['idx', 'dataset', 'domain', 'model', 'shots', 'temperature', 'judge_pred']]

os.makedirs(args.output, exist_ok = True)
prediction_df.to_csv(output_filename, index = False)