<h1 align="center"><span style="font-weight:normal">RoMath: A Mathematical Reasoning Benchmark in Romanian</h1>
<h2 align="center"> Accepted in MathNLP 2025 <br> The 2025 Conference on Empirical Methods in Natural Language Processing <br> EMNLP 2025</h2>
<div align="center">

[Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en), [Ana-Maria Bucur](https://scholar.google.com/citations?user=TQuQ5IAAAAAJ&hl=en), [Emilian Radoi](https://scholar.google.com/citations?user=yjtWIf8AAAAJ&hl=en)
</div>

<div align="center">

[📘 Abstract](#intro) |
[⚒️ Usage](#usage) |
[♻️ Reproducing the Results](#repro) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>


## <a name="tldr"> </a> TL;DR
<div>
  <div align="center">

  [📜 Arxiv Link](https://arxiv.org/abs/2409.11074) | [🤗 Huggingface Dataset](https://huggingface.co/datasets/cosmadrian/romath) | [🪧 Math NLP Poster](https://docs.google.com/presentation/d/1q-g3Tf2t2TLIlb3TpVpYsWqs7Pp0M2tJRT-p3Ue71XY/edit?usp=sharing)
  </div>
</div>

## <a name="intro"></a> 📘 Abstract

_Mathematics has long been conveyed through natural language, primarily for human understanding. With the rise of mechanized mathematics and proof assistants, there's a growing need to translate informal mathematical text into formal languages. However, most existing benchmarks focus solely on English, overlooking other languages. This paper introduces RoMath, a Romanian mathematical reasoning benchmark suite comprising three datasets: RoMath-Synthetic, RoMath-Baccalaureate, and RoMath-Competitions. These datasets cover a range of mathematical domains and difficulty levels, aiming to improve non-English language models and promote multilingual AI development. By focusing on Romanian, a low-resource language with unique linguistic features, RoMath addresses the limitations of Anglo-centric models and emphasizes the need for dedicated resources beyond simple automatic translation. We benchmark several language models, highlighting the importance of creating resources for underrepresented languages._

## <a name="usage"></a> ⚒️ Usage

Loading the data from 🤗 Huggingface Datasets:

```python
import datasets

subset = 'bac' # could be comps or synthetic

train_dataset = datasets.load_dataset('cosmadrian/romath', subset, split = 'train')
test_dataset = datasets.load_dataset('cosmadrian/romath', subset, split = 'test')

# Do your thing ...
```

## <a name="repro"></a> ♻️ Reproducing the Results

### Generating your own split for _Synthetic_
While a pre-generated split for RoMath-Synthetic is provided for convenience on [🤗 HuggingFace](https://huggingface.co/datasets/cosmadrian/romath), you can generate your own problems using the [original DeepMind](https://github.com/google-deepmind/mathematics_dataset) code with key phrases translated.

See [romath-synthetic/](romath-synthetic/) directory for instructions.

### Running Experiments

Experiments for the paper are organized in the in the `experiments/` directory, with separate scripts for each experiment in the paper. We used SLURM on a private cluster to train, make predictions and evaluate models. Use `./do_sbatch.sh <script.sh> <n_gpus>` to run a particular bash script. Modify the `./do_sbatch.sh` file to suit your needs.

To run a particular model on a dataset use the following commands:
```
# Optional LoRA-Fine-tuning
python fine_tune.py --model <hf_model_name> --dataset [bac|comps|synthetic] --output checkpoints/
```

```
# Use a (trained) model to make predictions on a test set.
python predict.py --model <hf_model_name> --dataset [bac|comps|synthetic] --temperature 0.5 --k 3 --shots 5 --output predictions/
```

```
# Evaluate the predictions of a model using a judge model.
python evaluate.py --pred_file predictions/Qwen-Qwen2-1.5B-Instruct_bac_2_0.5.csv --judge_model <hf_model_name> --output results/
```

```
# Compute the relevant metrics for all evaluated prediction files in a folder.
python evaluate/compute_metrics.py --input_dir results/ --output_dir metrics/
```

For translation, use the `translate.py` python script, alongside the `predict_translated.py` script.

For constructing the Judge Dataset (i.e., Table 3), run the `evaluate/make_judge_dataset.py` with the appropriate arguments and run `evaluate_judge.py` script.

### GRPO Training

For training with rewards, first train an SFT model on _Baccalaureate_ and _Competitions_ using a "reasoning" format like `<raționament>...</raționament><răspuns>...</răspuns>`, like so:

```
python sft_everything.py --batch_size 4 --model meta-llama/Llama-3.2-1B-Instruct --seed 42
```

Afterwards, train using GRPO (only correctness reward and strict_format reward), like so (adjust the parameters in the script to suit your hardware capabilities):

```
python grpo.py --batch_size 1 --model checkpoints-sft/meta-llama-Llama-3.2-1B-Instruct-sft/checkpoint-246/ --seed 42
```

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

[RoMath: A Mathematical Reasoning Benchmark in 🇷🇴 Romanian 🇷🇴](https://arxiv.org/abs/2409.11074)

```
@misc{cosma2024romath,
      title={RoMath: A Mathematical Reasoning Benchmark in Romanian},
      author={Adrian Cosma and Ana-Maria Bucur and Emilian Radoi},
      year={2024},
      eprint={2409.11074},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.11074},
}
```

## <a name="license"></a> 📝 License

This work is protected by [Attribution-NonCommercial 4.0 International](LICENSE)
