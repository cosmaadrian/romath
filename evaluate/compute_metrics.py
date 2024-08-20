import os
import argparse
import pandas as pd
import numpy as np

from glob import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

parser = argparse.ArgumentParser(description='Compute performance metrics')
parser.add_argument('--input_dir', type=str, help='Input path with all prediction .csv files.')
parser.add_argument('--evaluate_judge', action = 'store_true', help = 'Whether to evaluate the judge predictions or a normal model.')
parser.add_argument('--output_dir', type=str, help='Output path')
args = parser.parse_args()

print("Computing metrics for", args.__dict__)

def compute_classification_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    tn, fp, fn, tp =  confusion_matrix(true, pred, labels = [0,1]).ravel()

    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (fn + tp)
    tpr = tp / (fn + tp)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr, 'tpr': tpr}

def compute_accuracy_at_k(true, preds_df, k):
    predictions = []
    for problem_idx, group in preds_df.groupby('idx'):
        if len(group) < k:
            print(f'Problem {problem_idx} has less than {k} predictions.')

        # check if at least one of the k predictions is correct (based on judge_pred)
        if 1 in group['judge_pred'].values[:k]:
            predictions.append(1)
        else:
            predictions.append(0)

    predictions = np.array(predictions)
    correct_predictions = np.ones_like(predictions)
    acc = accuracy_score(correct_predictions, predictions)

    return {f'accuracy@{k}': acc}

# concatenate all the results that we have
results = pd.concat([pd.read_csv(file) for file in glob(f'{args.input_dir}/*.csv')]).reset_index(drop = True)

# considering -1 predictions as incorrect. Any better idea?
results['judge_pred'] = results['judge_pred'].apply(lambda x: 0 if x == -1 else x)

if args.evaluate_judge:
    grouped_results = results.groupby(['model', 'prompt'])

    results = []
    for (model_name, prompt_lang), group in grouped_results:
        overall_metrics = compute_classification_metrics(group['is_correct'].values, group['judge_pred'].values)
        results.append({'metric_type': 'overall', 'model': model_name, 'prompt': prompt_lang, **overall_metrics})

        for reason, reason_group in group.groupby('reason'):
            reason_metrics = compute_classification_metrics(reason_group['is_correct'].values, reason_group['judge_pred'].values)
            results.append({'metric_type': f'reason_{reason}', 'model': model_name, 'prompt': prompt_lang, **reason_metrics})

    results_df = pd.DataFrame(results)

    os.makedirs(args.output_dir, exist_ok = True)
    results_df.to_csv(f'{args.output_dir}/judge_metrics.csv', index = False)

else:
    grouped_results = results.groupby(['model'])

    results = []
    for model_name, group in grouped_results:
        overall_metrics = compute_classification_metrics(np.ones_like(group['judge_pred'].values), group['judge_pred'].values)

        # compute accuracy @ 1 and accuracy @ 10
        accuracy_at_1 = compute_accuracy_at_k(true = np.ones_like(group['judge_pred'].values), preds_df = group, k = 1)
        accuracy_at_10 = compute_accuracy_at_k(true = np.ones_like(group['judge_pred'].values), preds_df = group, k = 10)

        results.append({'metric_type': 'overall', 'model': model_name, **{**overall_metrics, **accuracy_at_1, **accuracy_at_10}})

        for domain, domain_group in group.groupby('domain'):
            normal_metrics = compute_classification_metrics(np.ones_like(domain_group['judge_pred'].values), domain_group['judge_pred'].values)

            # compute accuracy @ 1 and accuracy @ 10
            accuracy_at_1 = compute_accuracy_at_k(true = np.ones_like(domain_group['judge_pred'].values), preds_df = domain_group, k = 1)
            accuracy_at_10 = compute_accuracy_at_k(true = np.ones_like(domain_group['judge_pred'].values), preds_df = domain_group, k = 10)
            results.append({'metric_type': f'domain_{domain}', 'model': model_name, **{**normal_metrics, **accuracy_at_1, **accuracy_at_10}})

        results_df = pd.DataFrame(results)
        os.makedirs(args.output_dir, exist_ok = True)
        results_df.to_csv(f'{args.output_dir}/model_metrics.csv', index = False)