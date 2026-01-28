"""Game of Life Criticality Experiments."""
from gol_criticality.game import GameOfLife
from gol_criticality.experiment import evolve_until_cycle, perturb_single_cell, run_perturbation_experiment
from gol_criticality.plotting import plot_distributions

__version__ = "0.1.0"
__all__ = [
    "GameOfLife",
    "evolve_until_cycle",
    "perturb_single_cell",
    "run_perturbation_experiment",
    "plot_distributions",
]
