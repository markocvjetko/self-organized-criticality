"""Experiment logic for perturbation studies."""
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch

from gol_criticality.game import GameOfLife
from gol_criticality.utils import compute_state_hash, save_results, generate_experiment_id


def evolve_until_cycle(
    game: GameOfLife,
    initial_state: torch.Tensor,
    baseline_state: Optional[torch.Tensor] = None,
    max_steps: Optional[int] = None,
) -> Tuple[int, int, int, int, torch.Tensor]:
    """
    Evolve until a cycle is detected.

    Args:
        game: GameOfLife instance
        initial_state: The (possibly perturbed) state to evolve
        baseline_state: If provided, evolve this in parallel and only count
                       cells that differ from baseline (isolates perturbation effect
                       from background oscillations like blinkers)
        max_steps: Maximum steps before giving up

    Returns:
        steps_to_cycle: Number of steps until cycle detected
        total_cell_changes: Sum of cells differing from baseline across all steps
        unique_cells_affected: Number of cells that differed from baseline at least once
        cycle_period: Length of the detected cycle
        final_state: State when cycle was detected
    """
    state = initial_state.clone()
    baseline = baseline_state.clone() if baseline_state is not None else None
    state_history: Dict[bytes, int] = {}
    total_cell_changes = 0

    # Track which cells have ever been affected by the perturbation
    affected_mask = torch.zeros_like(state, dtype=torch.bool)

    state_hash = compute_state_hash(state)
    state_history[state_hash] = 0

    step = 0
    while max_steps is None or step < max_steps:
        state = game.step(state)
        if baseline is not None:
            baseline = game.step(baseline)
        step += 1

        # Compute cells affected by perturbation (differ from baseline)
        if baseline is not None:
            # Only count cells that differ from what they would be without perturbation
            differs_from_baseline = (state != baseline)
            cell_changes = torch.sum(differs_from_baseline.to(torch.float32)).item()
            affected_mask |= differs_from_baseline
        else:
            # Fallback: no baseline, count all changes (legacy behavior)
            # This path is used for initial convergence where there's no perturbation
            cell_changes = 0  # No perturbation effect to measure

        total_cell_changes += int(cell_changes)

        state_hash = compute_state_hash(state)
        if state_hash in state_history:
            cycle_period = step - state_history[state_hash]
            unique_cells = int(torch.sum(affected_mask).item())
            return step, total_cell_changes, unique_cells, cycle_period, state

        state_history[state_hash] = step

    # Max steps reached without cycle
    unique_cells = int(torch.sum(affected_mask).item())
    return step, total_cell_changes, unique_cells, 0, state


def perturb_single_cell(state: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
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
    output_dir: Optional[Union[Path, str]] = None,
    save_interval: int = 1000,
    experiment_id: Optional[str] = None,
    config: Optional[dict] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[dict], torch.Tensor]:
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
            _, _, _, _, stable_state = evolve_until_cycle(game, perturbed, baseline_state=stable_state)
    else:
        log("Phase 2: Skipped (no warmup perturbations)")

    # Recording phase
    log(f"Phase 3: Running {num_perturbations} recorded perturbations...")
    results = []
    start_time = time.perf_counter()

    for i in range(num_perturbations):
        if progress_callback:
            progress_callback(i + 1, num_perturbations, "record")

        perturbed, (row, col) = perturb_single_cell(stable_state)
        steps, changes, unique_cells, period, stable_state = evolve_until_cycle(game, perturbed, baseline_state=stable_state)

        results.append({
            "steps": steps,
            "changes": changes,
            "unique_cells": unique_cells,
            "period": period,
            "perturbation_coord": [row, col],
        })

        if verbose and (i + 1) % max(1, num_perturbations // 10) == 0:
            elapsed = time.perf_counter() - start_time
            sps = (i + 1) / elapsed if elapsed > 0 else 0.0
            log(f"  Perturbation {i + 1}/{num_perturbations} | SPS: {sps:.2f}")

        # Save intermediate results
        if output_dir and save_interval > 0 and (i + 1) % save_interval == 0:
            save_results(results, output_dir, f"{experiment_id}_checkpoint", config)

    # Log final SPS
    total_elapsed = time.perf_counter() - start_time
    final_sps = num_perturbations / total_elapsed if total_elapsed > 0 else 0.0
    log(f"Phase 3 complete: {num_perturbations} perturbations in {total_elapsed:.2f}s (avg SPS: {final_sps:.2f})")

    # Final save
    if output_dir:
        save_results(results, output_dir, experiment_id, config, stable_state)
        log(f"Results saved to {output_dir}/{experiment_id}_results.json")

    return results, stable_state
