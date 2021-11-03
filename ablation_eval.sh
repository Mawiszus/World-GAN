#!/bin/bash

for d in $1/*
do
    echo "$d"
    sbatch slurm_cpu.sh python quantitative_experiments.py --run_dir "$d/files" --output_dir "$d/files"
    sbatch slurm_cpu.sh python experiment_entropy.py --folder "$d/files/random_samples/torch_blockdata" --not_cuda
    area=`cat "$d/files/wandb-metadata.json" | jq -r .args[-8]`
    echo "$area"
    python generate_minecraft_samples.py --out_ "$d/files" --input_dir ../minecraft_worlds --input_name "Drehmal v2.1 PRIMORDIAL" --input_area_name "$area" --scales 0.75 0.5 0.25 --num_layer 3 --nfc 64 --repr_type bert_naive --num_samples 5 --render_obj
    for gd in $d/files/arbitrary_random_samples_v1.00000_h1.00000_d1.00000/objects/3/*.obj
    do
        blender -b --python ./minecraft/blender_scripts/CyclesMineways.py "$gd" 14 0
        blender -b --python ./minecraft/blender_scripts/CyclesMineways.py "$gd" 14 1
        blender -b --python ./minecraft/blender_scripts/CyclesMineways.py "$gd" 14 2
        blender -b --python ./minecraft/blender_scripts/CyclesMineways.py "$gd" 14 3
    done
done

