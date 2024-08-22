"""
    This script (roughly) evaluates the performance of the Judge model to verify proofs in Romanian.
    Uses the SolutionJudge class from judge_proofs.py to evaluate the predictions.

    It only outputs predictions and does not compute any metrics. To compute metrics, use evaluate/compute_performance_metrics.py.
"""

import os
import argparse
import pandas as pd

from tqdm import tqdm
from evaluate.solution_judge import SolutionJudge


def make_predictions(romath_judge_df, judge):
    romath_judge_df = romath_judge_df.copy()

    predictions = []
    for _, row in tqdm(romath_judge_df.iterrows(), total = len(romath_judge_df)):
        true_solution = row['solution']
        response = judge.evaluate(question = row['problem'], true = true_solution, prediction = row['prediction'], has_single_answer = False)
        predictions.append(response)

    romath_judge_df['judge_pred'] = predictions

    return romath_judge_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate Judge.')

    # The dataset is a .csv file purposely constructed, not one from huggingface!
    parser.add_argument('--dataset', type=str, help='Dataset path')

    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--prompt', type=str, help='Prompt language (en/ro)')
    parser.add_argument('--output_dir', type=str, help='Output path')

    args = parser.parse_args()

    print("Running Judge Predictions for", args.__dict__)

    romath_judge_df = pd.read_csv(args.dataset)
    romath_judge_df = romath_judge_df

    judge = SolutionJudge(model_name = args.model, prompt = args.prompt)
    prediction_df = make_predictions(romath_judge_df, judge)

    prediction_df['model'] = args.model
    prediction_df['prompt'] = args.prompt
    prediction_df = prediction_df[['idx', 'model', 'prompt', 'domain', 'judge_pred', 'is_correct', 'reason']]

    os.makedirs(args.output_dir, exist_ok = True)
    prediction_df.to_csv(f'{args.output_dir}/judge-predictions-{args.model.replace("/", "_")}_{args.prompt}.csv', index = False)

