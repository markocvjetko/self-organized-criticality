#!/bin/bash
# Submit multiple experiments with different parameters

for board_size in 100 150 200 300 400 500 1000; do
    for num_perturbations in 100000 1000000; do
    sbatch --export=ALL,BOARD_SIZE=$board_size,NUM_PERTURBATIONS=$num_perturbations \
        ./launch_experiments.sh
    echo "Submitted: board=$board_size, num_perturbations=$num_perturbations"
    done
done