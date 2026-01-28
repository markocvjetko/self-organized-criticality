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
conda activate criticality

which python3
python3 --version
echo $CONDA_DEFAULT_ENV
nvidia-smi

cd "$WORK"/phd/projects/complexity

BOARD_SIZE=${BOARD_SIZE:-500}
INIT_DENSITY=${INIT_DENSITY:-0.5}
OUTPUT_DIR=${OUTPUT_DIR:-./lustre/fsn1/projects/rech/imi/uix29qp/criticality}

python3 -m gol_criticality.cli \
    --board-size $BOARD_SIZE \
    --init-density $INIT_DENSITY \
    --num-warmup 100 \
    --num-perturbations 1000 \
    --num-experiments 3 \
    --output-dir $OUTPUT_DIR \
    --plot
