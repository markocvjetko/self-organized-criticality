#!/usr/bin/env python3
"""Generate plots from experiment results, including cross-experiment comparisons."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gol_criticality.plotting import plot_distributions


def extract_run_id(filename: str) -> str:
    """Extract run ID (e.g., '001') from filename like '100_100000_001_results.json'."""
    match = re.search(r'_(\d{3})_(?:checkpoint_)?results\.json$', filename)
    return match.group(1) if match else ""


def load_experiment_results(exp_dir: Path) -> list:
    """Load all results from an experiment directory, prioritizing complete over checkpoint."""
    results_list = []
    loaded_run_ids = set()

    # Load complete results first
    for f in sorted(exp_dir.glob("*_results.json")):
        if "checkpoint" not in f.name:
            run_id = extract_run_id(f.name)
            if run_id:
                loaded_run_ids.add(run_id)
            with open(f) as fp:
                data = json.load(fp)
                results_list.append(data["results"])

    # Load checkpoint results only for runs without complete results
    for f in sorted(exp_dir.glob("*_checkpoint_results.json")):
        run_id = extract_run_id(f.name)
        if run_id and run_id not in loaded_run_ids:
            with open(f) as fp:
                data = json.load(fp)
                results_list.append(data["results"])

    return results_list


def generate_individual_plots(base_dir: Path, force: bool = False):
    """Generate plots for each experiment directory that's missing them."""
    generated = []
    skipped = []

    for exp_top_dir in sorted(base_dir.iterdir()):
        if not exp_top_dir.is_dir() or not exp_top_dir.name.startswith("board_"):
            continue

        # Find the nested experiment directory
        exp_dir = None
        for subdir in exp_top_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                exp_dir = subdir
                break

        if exp_dir is None:
            continue

        # Check if plots already exist
        existing_plots = list(exp_dir.glob("experiment_*.png"))
        if existing_plots and not force:
            skipped.append(exp_top_dir.name)
            continue

        # Load results
        results_list = load_experiment_results(exp_dir)
        if not results_list:
            print(f"  No results found in {exp_dir}")
            continue

        # Generate plots
        plot_distributions(results_list, exp_dir, n_bins=50, prefix="experiment")
        generated.append(exp_top_dir.name)
        print(f"  Generated plots for {exp_top_dir.name} ({len(results_list)} runs)")

    return generated, skipped


def generate_comparison_plot(base_dir: Path, output_path: Path, n_bins: int = 50):
    """Generate a comparison plot showing distributions for all board sizes."""
    # Collect data from all experiments
    experiments = {}

    for exp_top_dir in sorted(base_dir.iterdir()):
        if not exp_top_dir.is_dir() or not exp_top_dir.name.startswith("board_"):
            continue

        # Parse board size and num_perturbations from directory name
        match = re.match(r'board_(\d+)_(\d+)', exp_top_dir.name)
        if not match:
            continue

        board_size = int(match.group(1))
        num_perturbations = int(match.group(2))

        # Find the nested experiment directory
        exp_dir = None
        for subdir in exp_top_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                exp_dir = subdir
                break

        if exp_dir is None:
            continue

        # Load results
        results_list = load_experiment_results(exp_dir)
        if not results_list:
            continue

        # Merge all runs into single arrays
        all_steps = []
        all_changes = []
        all_unique_cells = []
        for results in results_list:
            all_steps.extend([r['steps'] for r in results])
            all_changes.extend([r['changes'] for r in results])
            all_unique_cells.extend([r['unique_cells'] for r in results])

        key = (board_size, num_perturbations)
        experiments[key] = {
            'steps': all_steps,
            'changes': all_changes,
            'unique_cells': all_unique_cells,
            'n_samples': len(all_steps),
        }

    if not experiments:
        print("No experiments found for comparison plot")
        return

    # Separate by num_perturbations (100K vs 1M)
    exp_100k = {k: v for k, v in experiments.items() if k[1] == 100000}
    exp_1m = {k: v for k, v in experiments.items() if k[1] == 1000000}

    # Create comparison plots for each perturbation count
    for exp_group, suffix in [(exp_100k, "100k"), (exp_1m, "1m")]:
        if not exp_group:
            continue

        # Sort by board size
        sorted_keys = sorted(exp_group.keys(), key=lambda x: x[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_keys)))

        for idx, key in enumerate(sorted_keys):
            board_size, num_pert = key
            data = exp_group[key]
            color = colors[idx]
            label = f"Board {board_size} (n={data['n_samples']})"

            # Durations
            steps = data['steps']
            if len(steps) > 0 and max(steps) > min(steps):
                counts, bins = np.histogram(
                    steps,
                    bins=np.logspace(np.log10(max(1, min(steps))), np.log10(max(steps) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[0].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

            # Total changes
            changes = data['changes']
            if len(changes) > 0 and max(changes) > min(changes):
                counts, bins = np.histogram(
                    changes,
                    bins=np.logspace(np.log10(max(1, min(changes))), np.log10(max(changes) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[1].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

            # Unique cells
            unique_cells = data['unique_cells']
            if len(unique_cells) > 0 and max(unique_cells) > min(unique_cells):
                counts, bins = np.histogram(
                    unique_cells,
                    bins=np.logspace(np.log10(max(1, min(unique_cells))), np.log10(max(unique_cells) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[2].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

        # Format axes
        for ax, title, xlabel in zip(
            axes,
            ['Avalanche Durations', 'Total Cell Changes', 'Unique Cells Changed'],
            ['Duration (Steps)', 'Total Cell Flips', 'Unique Cells Affected']
        ):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=9, loc='best')

        pert_label = "100K" if suffix == "100k" else "1M"
        fig.suptitle(f'Distribution Comparison Across Board Sizes ({pert_label} perturbations)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_file = output_path / f"comparison_{suffix}_perturbations.png"
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved comparison plot: {out_file}")


def generate_balanced_comparison_plot(base_dir: Path, output_path: Path, n_bins: int = 50):
    """Generate comparison plot with equal sample sizes across all board sizes."""
    # Collect data from all experiments
    experiments = {}

    for exp_top_dir in sorted(base_dir.iterdir()):
        if not exp_top_dir.is_dir() or not exp_top_dir.name.startswith("board_"):
            continue

        match = re.match(r'board_(\d+)_(\d+)', exp_top_dir.name)
        if not match:
            continue

        board_size = int(match.group(1))
        num_perturbations = int(match.group(2))

        exp_dir = None
        for subdir in exp_top_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                exp_dir = subdir
                break

        if exp_dir is None:
            continue

        results_list = load_experiment_results(exp_dir)
        if not results_list:
            continue

        # Merge all runs into single arrays
        all_steps = []
        all_changes = []
        all_unique_cells = []
        for results in results_list:
            all_steps.extend([r['steps'] for r in results])
            all_changes.extend([r['changes'] for r in results])
            all_unique_cells.extend([r['unique_cells'] for r in results])

        key = (board_size, num_perturbations)
        experiments[key] = {
            'steps': all_steps,
            'changes': all_changes,
            'unique_cells': all_unique_cells,
            'n_samples': len(all_steps),
        }

    if not experiments:
        print("No experiments found for balanced comparison plot")
        return

    # Separate by num_perturbations
    exp_100k = {k: v for k, v in experiments.items() if k[1] == 100000}
    exp_1m = {k: v for k, v in experiments.items() if k[1] == 1000000}

    for exp_group, suffix in [(exp_100k, "100k"), (exp_1m, "1m")]:
        if not exp_group:
            continue

        # Find minimum sample count across all board sizes
        min_n = min(data['n_samples'] for data in exp_group.values())
        print(f"  Balancing {suffix} experiments to n={min_n:,} samples each")

        sorted_keys = sorted(exp_group.keys(), key=lambda x: x[0])

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_keys)))

        for idx, key in enumerate(sorted_keys):
            board_size, num_pert = key
            data = exp_group[key]
            color = colors[idx]
            label = f"Board {board_size}"

            # Take first min_n samples
            steps = data['steps'][:min_n]
            changes = data['changes'][:min_n]
            unique_cells = data['unique_cells'][:min_n]

            # Durations
            if len(steps) > 0 and max(steps) > min(steps):
                counts, bins = np.histogram(
                    steps,
                    bins=np.logspace(np.log10(max(1, min(steps))), np.log10(max(steps) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[0].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

            # Total changes
            if len(changes) > 0 and max(changes) > min(changes):
                counts, bins = np.histogram(
                    changes,
                    bins=np.logspace(np.log10(max(1, min(changes))), np.log10(max(changes) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[1].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

            # Unique cells
            if len(unique_cells) > 0 and max(unique_cells) > min(unique_cells):
                counts, bins = np.histogram(
                    unique_cells,
                    bins=np.logspace(np.log10(max(1, min(unique_cells))), np.log10(max(unique_cells) + 1), n_bins),
                    density=True
                )
                centers = np.sqrt(bins[:-1] * bins[1:])
                axes[2].plot(centers, counts, marker='o', linestyle='-', color=color,
                            label=label, markersize=3, alpha=0.8)

        # Format axes
        for ax, title, xlabel in zip(
            axes,
            ['Avalanche Durations', 'Total Cell Changes', 'Unique Cells Changed'],
            ['Duration (Steps)', 'Total Cell Flips', 'Unique Cells Affected']
        ):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=9, loc='best')

        pert_label = "100K" if suffix == "100k" else "1M"
        fig.suptitle(f'Balanced Comparison (n={min_n:,} each, {pert_label} perturbations)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        out_file = output_path / f"comparison_balanced_{suffix}_perturbations.png"
        fig.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved balanced comparison plot: {out_file}")


def main():
    base_dir = Path(__file__).parent.parent / "results" / "criticality"

    if not base_dir.exists():
        print(f"Results directory not found: {base_dir}")
        return

    print("=" * 60)
    print("Generating individual experiment plots...")
    print("=" * 60)
    generated, skipped = generate_individual_plots(base_dir)

    print(f"\nGenerated: {len(generated)} experiments")
    for name in generated:
        print(f"  - {name}")

    print(f"\nSkipped (already have plots): {len(skipped)} experiments")
    for name in skipped:
        print(f"  - {name}")

    print("\n" + "=" * 60)
    print("Generating cross-experiment comparison plots...")
    print("=" * 60)
    generate_comparison_plot(base_dir, base_dir)

    print("\n" + "=" * 60)
    print("Generating balanced comparison plots (equal n per board size)...")
    print("=" * 60)
    generate_balanced_comparison_plot(base_dir, base_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
