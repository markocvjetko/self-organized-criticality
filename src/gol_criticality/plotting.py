"""Plotting functions for experiment results."""
import math
from pathlib import Path

import numpy as np
import torch


def plot_distributions(
    results_list: list[list[dict]],
    output_dir: Path,
    final_states: list[torch.Tensor] | None = None,
    n_bins: int = 50,
    prefix: str = "experiment",
) -> list[Path]:
    """
    Generate distribution plots from experiment results.

    Args:
        results_list: List of experiment results (each is a list of dicts with 'steps', 'changes', 'unique_cells')
        output_dir: Directory to save plots
        final_states: List of final state tensors for each experiment (optional)
        n_bins: Number of histogram bins
        prefix: Filename prefix for saved plots

    Returns:
        List of paths to saved plot files
    """
    # Import matplotlib here to keep it optional for headless runs
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for SSH
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # Extract data from all experiments
    all_steps = []
    all_changes = []
    all_unique_cells = []
    per_exp_steps = []
    per_exp_changes = []
    per_exp_unique_cells = []

    for results in results_list:
        steps = [r['steps'] for r in results]
        changes = [r['changes'] for r in results]
        unique_cells = [r['unique_cells'] for r in results]
        per_exp_steps.append(steps)
        per_exp_changes.append(changes)
        per_exp_unique_cells.append(unique_cells)
        all_steps.extend(steps)
        all_changes.extend(changes)
        all_unique_cells.extend(unique_cells)

    # Plot 1: Merged distributions (durations, total changes, unique cells)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Durations
    counts_s, bins_s = np.histogram(
        all_steps,
        bins=np.logspace(np.log10(max(1, min(all_steps))), np.log10(max(all_steps) + 1), n_bins),
        density=True
    )
    centers_s = np.sqrt(bins_s[:-1] * bins_s[1:])
    axes[0].plot(centers_s, counts_s, marker='o', linestyle='-', color='tab:blue', markersize=4)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Duration (Steps to Convergence)', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Avalanche Durations', fontsize=13, fontweight='bold')
    axes[0].grid(True, which="both", alpha=0.3)

    # Total changes
    counts_c, bins_c = np.histogram(
        all_changes,
        bins=np.logspace(np.log10(max(1, min(all_changes))), np.log10(max(all_changes) + 1), n_bins),
        density=True
    )
    centers_c = np.sqrt(bins_c[:-1] * bins_c[1:])
    axes[1].plot(centers_c, counts_c, marker='o', linestyle='-', color='tab:orange', markersize=4)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Total Cell Flips', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Total Cell Changes', fontsize=13, fontweight='bold')
    axes[1].grid(True, which="both", alpha=0.3)

    # Unique cells
    counts_u, bins_u = np.histogram(
        all_unique_cells,
        bins=np.logspace(np.log10(max(1, min(all_unique_cells))), np.log10(max(all_unique_cells) + 1), n_bins),
        density=True
    )
    centers_u = np.sqrt(bins_u[:-1] * bins_u[1:])
    axes[2].plot(centers_u, counts_u, marker='o', linestyle='-', color='tab:green', markersize=4)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Unique Cells Affected', fontsize=12)
    axes[2].set_ylabel('Probability Density', fontsize=12)
    axes[2].set_title('Unique Cells Changed', fontsize=13, fontweight='bold')
    axes[2].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    path = output_dir / f"{prefix}_merged.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # Plot 2: Individual experiments (durations, total changes, unique cells)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_idx, (steps, changes, unique_cells) in enumerate(zip(per_exp_steps, per_exp_changes, per_exp_unique_cells)):
        # Durations
        counts_s, bins_s = np.histogram(
            steps,
            bins=np.logspace(np.log10(max(1, min(steps))), np.log10(max(steps) + 1), n_bins),
            density=True
        )
        centers_s = np.sqrt(bins_s[:-1] * bins_s[1:])
        axes[0].plot(centers_s, counts_s, marker='o', linestyle='-',
                     label=f'Exp {exp_idx + 1}', markersize=3, alpha=0.7)

        # Total changes
        counts_c, bins_c = np.histogram(
            changes,
            bins=np.logspace(np.log10(max(1, min(changes))), np.log10(max(changes) + 1), n_bins),
            density=True
        )
        centers_c = np.sqrt(bins_c[:-1] * bins_c[1:])
        axes[1].plot(centers_c, counts_c, marker='o', linestyle='-',
                     label=f'Exp {exp_idx + 1}', markersize=3, alpha=0.7)

        # Unique cells
        counts_u, bins_u = np.histogram(
            unique_cells,
            bins=np.logspace(np.log10(max(1, min(unique_cells))), np.log10(max(unique_cells) + 1), n_bins),
            density=True
        )
        centers_u = np.sqrt(bins_u[:-1] * bins_u[1:])
        axes[2].plot(centers_u, counts_u, marker='o', linestyle='-',
                     label=f'Exp {exp_idx + 1}', markersize=3, alpha=0.7)

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Duration (Steps)', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title('Durations by Experiment', fontsize=13, fontweight='bold')
    axes[0].grid(True, which="both", alpha=0.3)
    if len(per_exp_steps) <= 10:
        axes[0].legend(fontsize=9)

    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Total Cell Flips', fontsize=12)
    axes[1].set_ylabel('Probability Density', fontsize=12)
    axes[1].set_title('Total Changes by Experiment', fontsize=13, fontweight='bold')
    axes[1].grid(True, which="both", alpha=0.3)
    if len(per_exp_changes) <= 10:
        axes[1].legend(fontsize=9)

    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Unique Cells Affected', fontsize=12)
    axes[2].set_ylabel('Probability Density', fontsize=12)
    axes[2].set_title('Unique Cells by Experiment', fontsize=13, fontweight='bold')
    axes[2].grid(True, which="both", alpha=0.3)
    if len(per_exp_unique_cells) <= 10:
        axes[2].legend(fontsize=9)

    plt.tight_layout()
    path = output_dir / f"{prefix}_individual.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # Plot 3: Summary statistics (averages per experiment)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    exp_indices = np.arange(1, len(per_exp_steps) + 1)

    # Duration stats
    mean_durations = [np.mean(s) for s in per_exp_steps]
    std_durations = [np.std(s) for s in per_exp_steps]
    axes[0].errorbar(exp_indices, mean_durations, yerr=std_durations,
                     fmt='o-', capsize=4, color='tab:blue', markersize=8)
    axes[0].set_xlabel('Experiment', fontsize=12)
    axes[0].set_ylabel('Mean Duration (Steps)', fontsize=12)
    axes[0].set_title('Average Duration per Experiment', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(exp_indices)

    # Total changes stats
    mean_changes = [np.mean(c) for c in per_exp_changes]
    std_changes = [np.std(c) for c in per_exp_changes]
    axes[1].errorbar(exp_indices, mean_changes, yerr=std_changes,
                     fmt='o-', capsize=4, color='tab:orange', markersize=8)
    axes[1].set_xlabel('Experiment', fontsize=12)
    axes[1].set_ylabel('Mean Total Flips', fontsize=12)
    axes[1].set_title('Average Total Changes per Experiment', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(exp_indices)

    # Unique cells stats
    mean_unique = [np.mean(u) for u in per_exp_unique_cells]
    std_unique = [np.std(u) for u in per_exp_unique_cells]
    axes[2].errorbar(exp_indices, mean_unique, yerr=std_unique,
                     fmt='o-', capsize=4, color='tab:green', markersize=8)
    axes[2].set_xlabel('Experiment', fontsize=12)
    axes[2].set_ylabel('Mean Unique Cells', fontsize=12)
    axes[2].set_title('Average Unique Cells per Experiment', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(exp_indices)

    plt.tight_layout()
    path = output_dir / f"{prefix}_averages.png"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_paths.append(path)

    # Plot 4: Final states grid (if provided)
    if final_states and len(final_states) > 0:
        from gol_criticality.game import GameOfLife

        num_experiments = len(final_states)
        n_cols = math.ceil(math.sqrt(num_experiments))
        n_rows = math.ceil(num_experiments / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        axes = np.array(axes).reshape(-1)  # Flatten in case axes is 2D

        for i, final_state in enumerate(final_states):
            state_img = GameOfLife.draw(final_state)
            axes[i].imshow(state_img, cmap='gray', interpolation='nearest')
            axes[i].set_title(f'Experiment {i + 1}', fontsize=10)
            axes[i].axis('off')

        # Hide unused subplots
        for j in range(len(final_states), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        path = output_dir / f"{prefix}_final_states.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
