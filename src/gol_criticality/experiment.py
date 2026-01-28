"""Experiment logic for perturbation studies."""
from pathlib import Path
from typing import Callable

import torch

from gol_criticality.game import GameOfLife
from gol_criticality.utils import compute_state_hash, save_results, generate_experiment_id


def evolve_until_cycle(
    game: GameOfLife,
    initial_state: torch.Tensor,
    max_steps: int | None = None,
) -> tuple[int, int, int, int, torch.Tensor]:
    """
    Evolve until a cycle is detected.

    Returns:
        steps_to_cycle: Number of steps until cycle detected
        total_cell_changes: Sum of cell flips across all steps (counts every flip)
        unique_cells_affected: Number of cells that changed at least once
        cycle_period: Length of the detected cycle
        final_state: State when cycle was detected
    """
    state = initial_state.clone()
    state_history: dict[bytes, int] = {}
    total_cell_changes = 0

    # Track which cells have ever changed
    affected_mask = torch.zeros_like(state, dtype=torch.bool)

    state_hash = compute_state_hash(state)
    state_history[state_hash] = 0

    step = 0
    while max_steps is None or step < max_steps:
        prev_state = state
        state = game.step(state)
        step += 1

        # Compute changes this step
        changed_this_step = (state != prev_state)
        cell_changes = torch.sum(changed_this_step.to(torch.float32)).item()
        total_cell_changes += int(cell_changes)

        # Track unique cells affected
        affected_mask |= changed_this_step

        state_hash = compute_state_hash(state)
        if state_hash in state_history:
            cycle_period = step - state_history[state_hash]
            unique_cells = int(torch.sum(affected_mask).item())
            return step, total_cell_changes, unique_cells, cycle_period, state

        state_history[state_hash] = step

    # Max steps reached without cycle
    unique_cells = int(torch.sum(affected_mask).item())
    return step, total_cell_changes, unique_cells, 0, state


def perturb_single_cell(state: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Perturb a single dead cell near living cells.

    Returns:
        perturbed_state: State with one cell flipped to alive
        (row, col): Coordinates of the perturbed cell
    """
    state_perturbed = state.clone()
    dtype = state.dtype
    living_mask = state[0, 0] > 0
    living = living_mask.to(dtype).unsqueeze(0).unsqueeze(0)

    neighborhood = torch.nn.functional.max_pool2d(
        living, kernel_size=5, stride=1, padding=2
    )[0, 0]

    neighborhood_mask = neighborhood > 0
    candidate_mask = neighborhood_mask & (~living_mask)
    coords = torch.nonzero(candidate_mask, as_tuple=False)

    if coords.size(0) == 0:
        raise ValueError("No eligible dead cells found near living cells.")

    idx = torch.randint(0, coords.size(0), (1,)).item()
    row, col = coords[idx].tolist()
    state_perturbed[0, 0, row, col] = 1.0

    return state_perturbed, (row, col)


def run_perturbation_experiment(
    game: GameOfLife,
    initial_state: torch.Tensor,
    num_warmup_perturbations: int = 0,
    num_perturbations: int = 10000,
    output_dir: Path | str | None = None,
    save_interval: int = 1000,
    experiment_id: str | None = None,
    config: dict | None = None,
    verbose: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict], torch.Tensor]:
    """
    Run perturbation experiment with optional intermediate saving.

    Args:
        game: GameOfLife instance
        initial_state: Starting state
        num_warmup_perturbations: Perturbations before recording
        num_perturbations: Perturbations to record
        output_dir: Directory for saving results (None to disable)
        save_interval: How often to save intermediate results
        experiment_id: Unique identifier for this experiment
        config: Configuration dict to include in saved results
        verbose: Print progress to stdout
        progress_callback: Optional callback(current, total, phase) for progress

    Returns:
        results: List of result dicts for each perturbation
        final_state: Final stable state
    """
    if experiment_id is None:
        experiment_id = generate_experiment_id()
    if config is None:
        config = {}
    if output_dir is not None:
        output_dir = Path(output_dir)

    def log(msg: str):
        if verbose:
            print(msg)

    # Initial convergence
    log("Phase 1: Running initial state to convergence...")
    steps_init, changes_init, unique_init, period_init, stable_state = evolve_until_cycle(game, initial_state)
    log(f"  Converged in {steps_init} steps, period {period_init}")

    # Warmup phase
    if num_warmup_perturbations > 0:
        log(f"Phase 2: Running {num_warmup_perturbations} warmup perturbations...")
        for i in range(num_warmup_perturbations):
            if progress_callback:
                progress_callback(i + 1, num_warmup_perturbations, "warmup")
            if verbose and (i + 1) % max(1, num_warmup_perturbations // 10) == 0:
                log(f"  Warmup {i + 1}/{num_warmup_perturbations}")
            perturbed, _ = perturb_single_cell(stable_state)
            _, _, _, _, stable_state = evolve_until_cycle(game, perturbed)
    else:
        log("Phase 2: Skipped (no warmup perturbations)")

    # Recording phase
    log(f"Phase 3: Running {num_perturbations} recorded perturbations...")
    results = []

    for i in range(num_perturbations):
        if progress_callback:
            progress_callback(i + 1, num_perturbations, "record")

        perturbed, (row, col) = perturb_single_cell(stable_state)
        steps, changes, unique_cells, period, stable_state = evolve_until_cycle(game, perturbed)

        results.append({
            "steps": steps,
            "changes": changes,
            "unique_cells": unique_cells,
            "period": period,
            "perturbation_coord": [row, col],
        })

        if verbose and (i + 1) % max(1, num_perturbations // 10) == 0:
            log(f"  Perturbation {i + 1}/{num_perturbations}")

        # Save intermediate results
        if output_dir and save_interval > 0 and (i + 1) % save_interval == 0:
            save_results(results, output_dir, f"{experiment_id}_checkpoint", config)

    # Final save
    if output_dir:
        save_results(results, output_dir, experiment_id, config, stable_state)
        log(f"Results saved to {output_dir}/{experiment_id}_results.json")

    return results, stable_state
