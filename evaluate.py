from evaluate.solution_judge import SolutionJudge
import argparse
import tqdm
import os
import pandas as pd

parser = argparse.ArgumentParser(description = 'Use the judge to get scores.')
parser.add_argument('--pred_file', type = str, help = 'Input path for the .csv files with model-provided solutions.')
parser.add_argument('--judge_model', default = 'Qwen/Qwen2-1.5B-Instruct', type = str, help = 'Hugging Face model name for the judge.')
parser.add_argument('--prompt_lang', default = 'ro', type = str, help = 'Judge system prompt language.')
parser.add_argument('--output', default = 'results/', type = str, help = 'Output folder path for the .csv files with scores.')
args = parser.parse_args()

print("Running evaluation for", args.__dict__)

prediction_df = pd.read_csv(args.pred_file)
prediction_df['solution'] = prediction_df['answer'].astype(str)
prediction_df['answer'] = prediction_df['answer'].astype(str)

judge = SolutionJudge(args.judge_model, prompt = args.prompt_lang)

for i, row in tqdm.tqdm(prediction_df.iterrows(), total = len(prediction_df)):
    prediction_df.loc[i, 'judge_pred'] = judge.evaluate(
        question = row['problem'],
        true = row['solution'] if row['answer'] == 'Proof' else row['answer'],
        prediction = row['response'],
        has_single_answer = row['answer'] != 'Proof'
    )

prediction_df = prediction_df[['idx', 'dataset', 'domain', 'model', 'shots', 'temperature', 'judge_pred']]

model_name = prediction_df['model'].unique()[0]
dataset = prediction_df['dataset'].unique()[0]
shots = prediction_df['shots'].unique()[0]
temperature = prediction_df['temperature'].unique()[0]

os.makedirs(args.output, exist_ok = True)
prediction_df.to_csv(f'{args.output}/{model_name.replace("/", "-")}_{dataset}_{shots}_{temperature}.csv', index = False)