#!/bin/bash
# Submit multiple experiments with different parameters

for board_size in 100 200; do
    sbatch --export=ALL,BOARD_SIZE=$board_size \
        scripts/launch_experiments.sh
    echo "Submitted: board=$board_size"
done
