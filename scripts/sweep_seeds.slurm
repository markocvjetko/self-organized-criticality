#!/bin/bash
#SBATCH --job-name=gol_seed
#SBATCH --account=imi@v100
#SBATCH --time=11:59:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
##SBATCH --qos=qos_gpu-t3
#SBATCH -o logs/gol_seed_%A_%a.out
#SBATCH -e logs/gol_seed_%A_%a.err
#SBATCH --array=0-99

# One array task per seed. Submit with e.g.:
#   sbatch --array=0-99 sweep_seeds.slurm
# Each task runs gol_cycle_bara.py with seed = SEED_BASE + SLURM_ARRAY_TASK_ID
# and writes to its own JSON file so parallel tasks never clobber each other.

module purge
module load python/3.11.5
conda activate criticality

which python3
python3 --version
echo "$CONDA_DEFAULT_ENV"
nvidia-smi

# Run from the directory the job was submitted from (the scripts dir).
cd "$SLURM_SUBMIT_DIR" || exit 1

# --- sweep configuration (override via --export=ALL,VAR=... on sbatch) ---
SEED_BASE=${SEED_BASE:-0}
BATCH=${BATCH:-128}
DENSITY=${DENSITY:-0.5}
MINSIZE=${MINSIZE:-10}
MAXSIZE=${MAXSIZE:-1024}
OUTDIR=${OUTDIR:-data}

SEED=$((SEED_BASE + SLURM_ARRAY_TASK_ID))
mkdir -p "$OUTDIR"
JSON_PATH="$OUTDIR/data_seed${SEED}.json"

echo "array_task=$SLURM_ARRAY_TASK_ID seed=$SEED -> $JSON_PATH"

python3 gol_cycle_bara.py \
    --batch="$BATCH" \
    --seed="$SEED" \
    --density="$DENSITY" \
    --toroidal \
    --minsize="$MINSIZE" \
    --maxsize="$MAXSIZE" \
    --json-path="$JSON_PATH"
