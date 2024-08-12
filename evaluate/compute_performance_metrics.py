import pandas as pd

from glob import glob
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_metrics(true, pred):
    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    tn, fp, fn, tp =  confusion_matrix(true, pred, labels = [0,1]).ravel()
    
    # TODO:  invalid value encountered in scalar divide tpr = tp / (fn + tp)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (fn + tp)
    tpr = tp / (fn + tp)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'tnr': tnr, 'fpr': fpr, 'fnr': fnr, 'tpr': tpr}

# concatenate all the results that we have
results = pd.concat([pd.read_csv(file) for file in glob('../results/predictions_*.csv')]).reset_index(drop = True)

# considering -1 predictions as incorrect. TODO: any better idea?
results['judge_pred'] = results['judge_pred'].apply(lambda x: 0 if x == -1 else x)

grouped_results = results.groupby(['model', 'prompt'])

results = []
for (model_name, prompt_lang), group in grouped_results:
    overall_metrics = compute_metrics(group['is_correct'].values, group['judge_pred'].values)
    results.append({'metric_type': 'overall', 'model': model_name, 'prompt': prompt_lang, **overall_metrics})

    for reason, reason_group in group.groupby('reason'):
        reason_metrics = compute_metrics(reason_group['is_correct'].values, reason_group['judge_pred'].values)
        results.append({'metric_type': f'reason_{reason}', 'model': model_name, 'prompt': prompt_lang, **reason_metrics})

results_df = pd.DataFrame(results)
results_df.to_csv('../results/judge_metrics.csv', index = False)
    