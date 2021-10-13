#!/bin/bash

for d in output/ablation/bert/*
do
    # sbatch slurm_cpu.sh python quantitative_experiments.py --run_dir "$d/files" --output_dir "$d/files"
    area=`cat "$d/files/wandb-metadata.json" | jq -r .args[-8]`
    echo "$area"
    python generate_minecraft_samples.py --out_ "$d/files" --input_dir ../minecraft_worlds --input_name "Drehmal v2.1 PRIMORDIAL" --input_area_name "$area" --scales 0.75 0.5 0.25 --num_layer 3 --nfc 64 --repr_type bert --num_samples 5 --render_obj
done

