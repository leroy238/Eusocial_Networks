#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=60:00:00
#SBATCH --job-name=beez
#SBATCH --output=logs/%x_%j.out
#SBATCH --export=ALL

# ---- Here you can define your job parameters
EPISODES=10000
MAX_BUFFER=512
LR=0.0001
GAMMA=0.99
EPSILON=.5
MINIBATCH=32
TARGET_UPDATE=100
NUM_BEES=32
HIDDEN_DIM=128
N=5
DECAY=0.95
NO_COMM=0

# ---- Make a dynamic job name and log name
JOB_NAME="bees_e${EPISODES}_mbuf${MAX_BUFFER}_lr${LR}_g${GAMMA}_mb${MINIBATCH}_tu${TARGET_UPDATE}_nb${NUM_BEES}_hd${HIDDEN_DIM}_n${N}_d${DECAY}_no_com${NO_COMM}"
LOG_FILE=logs/${JOB_NAME}.log




# ---- Load modules and run
module load anaconda3
conda init bash
conda activate ~/.conda/envs/RL

echo "Running with parameters:"
echo "Episodes: $EPISODES"
echo "Max Buffer: $MAX_BUFFER"
echo "Learning Rate: $LR"
echo "Gamma: $GAMMA"
echo "Epsilon: $EPSILON"
echo "Minibatch: $MINIBATCH"
echo "Target Update: $TARGET_UPDATE"
echo "Num Bees: $NUM_BEES"
echo "Hidden Dim: $HIDDEN_DIM"
echo "N: $N"
echo "Decay: $DECAY"
echo "No communication: $NO_COMM"

# ---- Pass the parameters to python
python run.py --episodes $EPISODES --max_buffer $MAX_BUFFER --lr $LR --gamma $GAMMA --epsilon $EPSILON --minibatch $MINIBATCH --target_update $TARGET_UPDATE --num_bees $NUM_BEES --hidden_dim $HIDDEN_DIM --N $N --decay $DECAY --no_com $NO_COMM
