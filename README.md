# romath
Official repository for "RoMath - A Mathematical Reasoning Benchmarking Suite from Descriptions in 🇷🇴 Romanian 🇷🇴"


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

TODO
- [ x ] Run evaluation with the judge for all predictions
    - run_judge.sh with everything uncommented
    - run_judge2.sh with everything uncommented

- [ x ] Run predictions with translated versions of a couple of models (math ones?)


!!!!!!!!! TODO !!!!!!!!!!

predict.py puts the same question on the whole batch, prediction is made with the right question but the judge does not

!!!!!!!!!!!! FIX THE JUDGE AND RUN AGAIN EVERYTHING!!!!!!!!!!!!!