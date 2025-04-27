#!/bin/bash

#SBATCH --partition=Bunescu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=60:00:00

# ---- Here you can define your job parameters
EPISODES=10000
MAX_BUFFER=64
LR=0.001
GAMMA=0.99
MINIBATCH=32
TARGET_UPDATE=10000
NUM_BEES=32
HIDDEN_DIM=128
N=5
DECAY=0.95
NO_COMM=0

# ---- Make a dynamic job name and log name
JOB_NAME=bees_e${EPISODES}_lr${LR}_n${N}
LOG_FILE=logs/${JOB_NAME}.log

#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --export=ALL

# ---- Load modules and run
"" > ${LOG_FILE}  # clear the log file if it exists
module load anaconda3
conda init bash
conda activate ~/.conda/envs/RL

# ---- Pass the parameters to python
python run.py --episodes $EPISODES --max_buffer $MAX_BUFFER --lr $LR --gamma $GAMMA --minibatch $MINIBATCH --target_update $TARGET_UPDATE --num_bees $NUM_BEES --hidden_dim $HIDDEN_DIM --N $N --decay $DECAY --use_com $NO_COMM
