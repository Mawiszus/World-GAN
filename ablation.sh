#!/bin/zsh

sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name ruins --sub_coords 0.0 1.0 0.0 1.0 0.0 1.0
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name simple_beach --sub_coords 0.0 0.5 0.0 1.0 0.0 1.0
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name desert --sub_coords 0.25 0.75 0.0 1.0 0.25 0.75
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name plains --sub_coords 0.25 0.75 0.0 1.0 0.25 0.75
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name swamp --sub_coords 0.0 1.0 0.0 1.0 0.0 1.0
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name vanilla_village --sub_coords 0.33333 0.66667 0.0 1.0 0.33333 0.66667
sleep 60
sbatch slurm.sh python main.py --input_dir ../minecraft_worlds/ --input_name Drehmal\ v2.1\ PRIMORDIAL --scales 0.75 0.5 0.25 --num_layer 3 --alpha 100 --niter 3000 --nfc 64 --pad_with_noise --repr_type bert --input_area_name vanilla_mineshaft --sub_coords 0.0 1.0 0.0 1.0 0.0 1.0