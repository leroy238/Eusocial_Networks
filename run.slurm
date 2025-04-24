#!/bin/bash

#SBATCH --partition=Bunescu
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time 60:00:00
#SBATCH --job-name bees
#SBATCH --output logs/bees_4.log
#SBATCH --export=ALL

"" > logs/bees_4.log
module load anaconda3
conda init bash
conda activate ~/.conda/envs/RL
python test.py
