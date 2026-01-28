#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH -o out_gol_%j
#SBATCH -e err_gol_%j
#SBATCH --qos=qos_gpu-t3

module load arch/a100
module load python/3.11.5
module load cuda/12.8.0
conda activate rlca

which python3
python3 --version
echo $CONDA_DEFAULT_ENV
nvidia-smi

cd "$WORK"/phd/projects/complexity

BOARD_SIZE=${BOARD_SIZE:-500}
INIT_DENSITY=${INIT_DENSITY:-0.5}

python3 -m gol_criticality.cli \
    --board-size $BOARD_SIZE \
    --init-density $INIT_DENSITY \
    --num-warmup 10000 \
    --num-perturbations 100000 \
    --num-experiments 10 \
    --output-dir ./results \
    --plot
