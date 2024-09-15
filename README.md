<h1 align="center"><span style="font-weight:normal">RoMath: A Mathematical Reasoning Benchmarking Suite from Problem Descriptions in 🇷🇴 Romanian 🇷🇴 </h1>

<div align="center">
  
[Adrian Cosma](https://scholar.google.com/citations?user=cdYk_RUAAAAJ&hl=en), [Ana-Maria Bucur](https://scholar.google.com/citations?user=TQuQ5IAAAAAJ&hl=en), [Emilian Radoi](https://scholar.google.com/citations?user=yjtWIf8AAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Abstract](#intro) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>


## <a name="tldr"> </a> TL;DR 
<div>
  <div align="center">
    
  [📜 Arxiv Link]() | [🤗 Huggingface Dataset](https://huggingface.co/datasets/cosmadrian/romath)
  </div>
</div>

## <a name="intro"></a> 📘 Abstract

_Mathematics has long been conveyed through natural language, primarily for human understanding. With the rise of mechanized mathematics and proof assistants, there's a growing need to translate informal mathematical text into formal languages. However, most existing benchmarks focus solely on English, overlooking other languages. This paper introduces RoMath, a Romanian mathematical reasoning benchmark suite comprising three datasets: RoMath-Synthetic, RoMath-Baccalaureate, and RoMath-Competitions. These datasets cover a range of mathematical domains and difficulty levels, aiming to improve non-English language models and promote multilingual AI development. By focusing on Romanian, a low-resource language with unique linguistic features, RoMath addresses the limitations of Anglo-centric models and emphasizes the need for dedicated resources beyond simple automatic translation. We benchmark several language models, highlighting the importance of creating resources for underrepresented languages._

```
# Optional
python fine_tune.py --model <hf_model_name> --dataset [bac|comps|synthetic] --output checkpoints/
```
```
python predict.py --model <hf_model_name> --dataset [bac|comps|synthetic] --temperature 0.5 --k 3 --shots 5 --output predictions/
```

```
python evaluate.py --pred_file results/Qwen-Qwen2-1.5B-Instruct_bac_2_0.5.csv --judge_model <hf_model_name> --output results/
```

```
python evaluate/compute_metrics.py --input_dir results/ --output_dir metrics/
```

## <a name="citation"></a> 📖 Citation
If you found our work useful, please cite our paper:

[RoMath: A Mathematical Reasoning Benchmarking Suite from Problem Descriptions in 🇷🇴 Romanian 🇷🇴]()

```
@InProceedings{cosma24romath,
  author="Cosma, Adrian and Bucur, Ana-Maria and Radoi, Emilian",
  title="RoMath: A Mathematical Reasoning Benchmarking Suite from Problem Descriptions in Romanian",
  year="2024",
}
```

## <a name="license"></a> 📝 License

This work is protected by [Attribution-NonCommercial 4.0 International](LICENSE)
