# Mathematics Dataset (Romanian Translation)

## TL;DR
This repository is adapted from `https://github.com/google-deepmind/mathematics_dataset`. We translated the key phrases from each module from English into Romanian (see [modules/](modules/)). The rest of the repository works as the original.

## Introduction

This dataset code generates mathematical question and answer pairs, from a range
of question types at roughly school-level difficulty. This is designed to test
the mathematical learning and algebraic reasoning skills of learning models.

Original paper: [Analysing Mathematical
Reasoning Abilities of Neural Models](https://openreview.net/pdf?id=H1gR5iR5FX)
(Saxton, Grefenstette, Hill, Kohli).

## Generating examples

Generated examples can be printed to stdout via the `generate` script. For
example:

```shell
python -m mathematics_dataset.generate --filter=linear_1d
```

will generate example (question, answer) pairs for solving linear equations in
one variable.

We've also included `generate_to_file.py` as an example of how to write the
generated examples to text files. You can use this directly, or adapt it for
your generation and training needs.

## Making the .csv

For convenience, to compile all problems in a .csv file, run:

```shell
python -m make_csv.py
```

