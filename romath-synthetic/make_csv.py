import os
import glob
import pandas as pd
from collections import defaultdict

PATH = 'final-dataset/romath_deepmind/'

train_folders = glob.glob(PATH + 'train-*/')

df = defaultdict(list)
idx = 0
for folder in train_folders:
    files = glob.glob(folder + '*.txt')
    split_name = folder.split('/')[-2]
    difficulty = split_name.split('-')[-1]

    for file in files:
        domain = file.split('/')[-1].split('.')[0].split('__')[0]
        module = file.split('/')[-1].split('.')[0].split('__')[1]

        with open(file, 'rt') as f:
            data = f.read().split('\n')

        definitions = data[::2]
        solutions = data[1::2]

        for definition, solution in zip(definitions, solutions):
            if solution == 'True':
                solution = 'Adevărat'

            if solution == 'False':
                solution = 'Fals'

            df['idx'].append(idx)
            df['split'].append(split_name)
            df['difficulty'].append(difficulty)
            df['domain'].append(domain.capitalize())
            df['module'].append(module.replace('_', ' ').replace('composed', '(composed)'))
            df['problem'].append(definition)
            df['solution'].append(solution)
            idx += 1

df = pd.DataFrame(df)
df.to_csv('final-dataset/romath-deepmind-train.csv', index = False)

###############################################################################
###############################################################################
###############################################################################

test_folders = [PATH + 'interpolate/', PATH + 'extrapolate/']

df = defaultdict(list)
idx = 0

for folder in test_folders:
    files = glob.glob(folder + '*.txt')
    split_name = folder.split('/')[-2]

    for file in files:
        domain = file.split('/')[-1].split('.')[0].split('__')[0]
        module = file.split('/')[-1].split('.')[0].split('__')[1]

        with open(file, 'rt') as f:
            data = f.read().split('\n')

        definitions = data[::2]
        solutions = data[1::2]

        for definition, solution in zip(definitions, solutions):
            if solution == 'True':
                solution = 'Adevărat'

            if solution == 'False':
                solution = 'Fals'

            df['idx'].append(idx)
            df['split'].append(split_name)
            df['domain'].append(domain.capitalize())
            df['module'].append(module.replace('_', ' ').replace('composed', '(composed)'))
            df['problem'].append(definition)
            df['solution'].append(solution)
            idx += 1

df = pd.DataFrame(df)
df.to_csv('final-dataset/romath-deepmind-test.csv', index = False)