#!/bin/bash
#SBATCH --account=imi@v100
#SBATCH -C v100-32g
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH -o logs/out_gol_%j
#SBATCH -e logs/err_gol_%j
#SBATCH --qos=qos_gpu-t3

module purge
module load cuda/12.8.0
module load python/3.11.5
conda activate criticality

which python3
python3 --version
echo $CONDA_DEFAULT_ENV
nvidia-smi

cd "$WORK"/proj/self-organized-criticality

BOARD_SIZE=${BOARD_SIZE:-500}
NUM_PERTURBATIONS=${NUM_PERTURBATIONS:-100000}
INIT_DENSITY=${INIT_DENSITY:-0.5}
OUTPUT_DIR=${OUTPUT_DIR:-/lustre/fsn1/projects/rech/imi/uix29qp/criticality/board_${BOARD_SIZE}_${NUM_PERTURBATIONS}}

python3 -m gol_criticality.cli \
    --board-size $BOARD_SIZE \
    --init-density $INIT_DENSITY \
    --num-warmup 10000 \
    --num-perturbations $NUM_PERTURBATIONS \
    --num-experiments 5 \
    --output-dir $OUTPUT_DIR \
    --experiment-id $BOARD_SIZE\_$NUM_PERTURBATIONS \
    --plot
