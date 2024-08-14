import pandas as pd
import tqdm
import random
import nltk
import re
import argparse

parser = argparse.ArgumentParser(description='Make the dataset to evaluate the correctness of the Judge.')
parser.add_argument('--n_samples', type = int, default = 150, help = 'Number of samples to take from each dataset.')
parser.add_argument('--output', type = str, help = 'Output folder.')
args = parser.parse_args()

# TODO use the args!

def levenstein(text1: str, text2: str) -> float:
    return 1 - nltk.edit_distance(text1, text2) / max(len(text1), len(text2))

# TODO read from huggingface instead!
bac_df = pd.read_csv('assets/romath-bac-train_ans.csv')
bac_df = bac_df[bac_df['answer'] == 'Proof'].reset_index(drop = True)
comps_df = pd.read_csv('assets/romath-comps-train_ans.csv')
comps_df = comps_df[comps_df['answer'] == 'Proof'].reset_index(drop = True)

all_solutions = pd.concat([bac_df, comps_df]).reset_index(drop = True)

dataset = pd.concat([
    bac_df.sample(150, random_state = 42)[['domain', 'problem', 'solution']],
    comps_df.sample(150, random_state = 42)[['domain', 'problem', 'solution']],
]).reset_index(drop = True)

# filter out rows present in "dataset"
all_solutions = all_solutions[all_solutions.apply(lambda x: x['solution'] not in dataset['solution'].values, axis = 1)]

dataset['idx'] = range(len(dataset))
dataset['prediction'] = None
dataset['is_correct'] = None
dataset['reason'] = None

def other_solution(solution, all_solutions):
    sols = all_solutions['solution'].sample(500)
    most_similar_solution = sols[sols.apply(lambda x: levenstein(x, solution)).idxmax()]
    return most_similar_solution, "Other"

def manipulate_solution(solution, all_solutions, try_number = 0):
    if try_number > 10:
        return other_solution(solution, all_solutions)

    original_solution = solution

    if random.choice([True, False]):
        solution = re.sub(r'(\d+)', '-\\1', solution)

    if random.choice([True, False]):
        solution = re.sub(r'<', '>', solution)
        solution = re.sub('\\\\lt', '\\\\gt', solution)

    if random.choice([True, False]):
        solution = re.sub(r'>', '<', solution)
        solution = re.sub('\\\\gt', '\\\\lt', solution)

    if random.choice([True, False]):
        solution = re.sub(r'>=', '<', solution)
        solution = re.sub('\\\\geq', '\\\\lt', solution)

    if random.choice([True, False]):
        solution = re.sub(r'<', '>=', solution)
        solution = re.sub('\\\\lt', '\\\\geq', solution)

    if random.choice([True, False]):
        solution = re.sub(r'!=', '=', solution)

    if random.choice([True, False]):
        solution = re.sub(r'\+', '-', solution)

    if random.choice([True, False]):
        solution = re.sub(r'-', '+', solution)

    if random.choice([True, False]):
        numbers = re.findall(r'\d+', solution)
        if numbers:
            number = random.choice(numbers)
            solution = re.sub(number, str(int(number) + random.choice([-1, 1])), solution)

    if original_solution == solution:
        print('No change! Trying again.')
        return manipulate_solution(solution, all_solutions, try_number = try_number + 1)

    return solution, "Manipulate"

for i, row in tqdm.tqdm(dataset.iterrows(), total = len(dataset)):
    original_solution = row['solution']

    if random.choice([True, False]):
        is_correct = 1

        if random.choice([True, False]):
            dataset.at[i, 'prediction'] = original_solution
        else:
            # find all math stuff delimited by either \( \)  or \[ \]
            math_stuff = re.findall(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', original_solution)
            if not math_stuff:
                dataset.at[i, 'prediction'] = original_solution
            else:
                only_math_stuff = '\\(' + '; '.join([match[0] for match in math_stuff if match[0] != '']) + '\\)'
                dataset.at[i, 'prediction'] = only_math_stuff

        dataset.at[i, 'reason'] = "Correct"
    else:
        is_correct = 0
        fn = random.choice([other_solution, manipulate_solution])
        wrong_solution, reason = fn(original_solution, all_solutions)

        dataset.at[i, 'prediction'] = wrong_solution
        dataset.at[i, 'reason'] = reason

    dataset.at[i, 'is_correct'] = is_correct

dataset = dataset[['idx', 'domain', 'problem', 'solution', 'prediction', 'is_correct', 'reason']]

# TODO save to args.output
dataset.to_csv('assets/romath-judge-train.csv', index = False)