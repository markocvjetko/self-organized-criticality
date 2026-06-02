"""Command-line interface for running experiments."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch

from gol_criticality.game import GameOfLife
from gol_criticality.experiment import run_perturbation_experiment
from gol_criticality.utils import generate_experiment_id


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Game of Life criticality experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Board configuration
    parser.add_argument(
        "--board-size", "-b",
        type=int,
        default=300,
        help="Board size (square grid)",
    )
    parser.add_argument(
        "--init-density", "-d",
        type=float,
        default=0.5,
        help="Initial cell density (0.0 to 1.0)",
    )
    parser.add_argument(
        "--toroidal", "-t",
        action="store_true",
        help="Use toroidal (wrap-around) boundary conditions",
    )

    # Experiment parameters
    parser.add_argument(
        "--num-warmup", "-w",
        type=int,
        default=0,
        help="Number of warmup perturbations (not recorded)",
    )
    parser.add_argument(
        "--num-perturbations", "-n",
        type=int,
        default=1000,
        help="Number of perturbations to record",
    )
    parser.add_argument(
        "--num-experiments", "-e",
        type=int,
        default=1,
        help="Number of independent experiments to run",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("./results"),
        help="Directory for saving results",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="Save checkpoint every N perturbations (0 to disable)",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Custom experiment ID (auto-generated if not provided)",
    )

    # Runtime options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    # Plotting options
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate distribution plots after experiments complete",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=50,
        help="Number of histogram bins for plots",
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    opts = parse_args(args)

    # Set random seed if provided
    if opts.seed is not None:
        torch.manual_seed(opts.seed)

    # Generate run ID and create run folder
    run_id = opts.experiment_id or generate_experiment_id()
    run_dir = opts.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build config dict for saving
    config = {
        "board_size": opts.board_size,
        "init_density": opts.init_density,
        "toroidal": opts.toroidal,
        "num_warmup": opts.num_warmup,
        "num_perturbations": opts.num_perturbations,
        "num_experiments": opts.num_experiments,
        "device": opts.device,
        "seed": opts.seed,
    }

    # Save run-level config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if not opts.quiet:
        print(f"Configuration: {json.dumps(config, indent=2)}")
        print(f"Output directory: {run_dir}")
        print(f"Running {opts.num_experiments} experiment(s)...")

    # Collect results and final states from all experiments
    all_results = []
    all_final_states = []

    # Run experiments
    for exp_idx in range(opts.num_experiments):
        exp_id = f"{run_id}_{exp_idx + 1:03d}"

        if not opts.quiet:
            print(f"\n{'='*60}")
            print(f"Experiment {exp_idx + 1}/{opts.num_experiments}: {exp_id}")
            print(f"{'='*60}")

        # Initialize game and state
        game = GameOfLife(
            board_size=(opts.board_size, opts.board_size),
            device=opts.device,
            init_density=opts.init_density,
            toroidal=opts.toroidal,
        )
        initial_state = game.init_state()

        # Run experiment
        results, final_state = run_perturbation_experiment(
            game=game,
            initial_state=initial_state,
            num_warmup_perturbations=opts.num_warmup,
            num_perturbations=opts.num_perturbations,
            output_dir=run_dir,
            save_interval=opts.save_interval,
            experiment_id=exp_id,
            config=config,
            verbose=not opts.quiet,
        )

        # Collect results and final states for plotting
        all_results.append(results)
        all_final_states.append(final_state)

        if not opts.quiet:
            # Print summary statistics
            steps_list = [r["steps"] for r in results]
            changes_list = [r["changes"] for r in results]
            unique_cells_list = [r["unique_cells"] for r in results]
            print(f"\nSummary for {exp_id}:")
            print(f"  Steps - min: {min(steps_list)}, max: {max(steps_list)}, "
                  f"mean: {sum(steps_list)/len(steps_list):.1f}")
            print(f"  Total changes - min: {min(changes_list)}, max: {max(changes_list)}, "
                  f"mean: {sum(changes_list)/len(changes_list):.1f}")
            print(f"  Unique cells - min: {min(unique_cells_list)}, max: {max(unique_cells_list)}, "
                  f"mean: {sum(unique_cells_list)/len(unique_cells_list):.1f}")

    if not opts.quiet:
        print(f"\nAll experiments complete. Results saved to {run_dir}/")

    # Generate plots if requested
    if opts.plot:
        if not opts.quiet:
            print("\nGenerating plots...")
        from gol_criticality.plotting import plot_distributions
        saved_plots = plot_distributions(
            results_list=all_results,
            output_dir=run_dir,
            final_states=all_final_states,
            n_bins=opts.plot_bins,
            prefix="experiment",
        )
        if not opts.quiet:
            print(f"Saved {len(saved_plots)} plots:")
            for p in saved_plots:
                print(f"  - {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
