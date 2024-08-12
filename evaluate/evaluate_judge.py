import pandas as pd
import argparse
from tqdm import tqdm
from judge_proofs import SolutionJudge
import os

def make_predictions(romath_judge_df, judge):
    romath_judge_df = romath_judge_df.copy()

    predictions = []
    for idx, row in tqdm(romath_judge_df.iterrows(), total = len(romath_judge_df)):
        true_solution = row['answer'] if row['has_single_answer'] else row['solution']
        response = judge.evaluate(question = row['problem'], true = true_solution, prediction = row['prediction'], has_single_answer = row['has_single_answer'])
        predictions.append(response)

    romath_judge_df['judge_pred'] = predictions

    return romath_judge_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Annotate answer')
    parser.add_argument('--dataset', type=str, help='Dataset path')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--prompt', type=str, help='Prompt language (en/ro)')

    args = parser.parse_args()

    romath_judge_df = pd.read_csv(args.dataset)
    romath_judge_df = romath_judge_df[:10]
    romath_judge_df['has_single_answer'] = romath_judge_df['answer'].apply(lambda x: False if x == 'Proof' else True)

    judge = SolutionJudge(model_name = args.model, prompt = args.prompt)
    prediction_df = make_predictions(romath_judge_df, judge)

    prediction_df['model'] = args.model
    prediction_df['prompt'] = args.prompt
    prediction_df = prediction_df[['idx', 'domain', 'judge_pred', 'is_correct', 'reason', 'has_single_answer', 'model', 'prompt']]

    os.makedirs('../results', exist_ok = True)    
    prediction_df.to_csv(f'../results/predictions_{args.model.replace("/", "_")}_{args.prompt}.csv', index = False)

