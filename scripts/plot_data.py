#!/usr/bin/env python
"""Plot transient and loop length vs grid size from data.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", nargs="?", default="data.json", help="path to data.json")
    p.add_argument("-o", "--out", default="data.png", help="output image path")
    p.add_argument("--logy", action="store_true", help="log-scale the y axes")
    args = p.parse_args()

    data = json.loads(Path(args.data).read_text())
    sizes = np.array(sorted(int(k) for k in data))

    t_mean = np.array([data[str(s)]["transient_mean"] for s in sizes])
    t_std = np.array([data[str(s)]["transient_std"] for s in sizes])
    l_mean = np.array([data[str(s)]["loop_mean"] for s in sizes])
    l_std = np.array([data[str(s)]["loop_std"] for s in sizes])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, mean, std, label, color in [
        (axes[0, 0], t_mean, t_std, "transient (mu)", "tab:blue"),
        (axes[0, 1], l_mean, l_std, "loop (lambda)", "tab:red"),
    ]:
        ax.plot(sizes, mean, "-o", color=color, markersize=3, label="mean")
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.2,
                        label="+/- std")
        ax.set_xlabel("grid size")
        ax.set_ylabel(label)
        ax.set_title(label + " vs grid size")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Individual samples for every grid size.
    for ax, field, label, color in [
        (axes[1, 0], "transient_lengths", "transient (mu)", "tab:blue"),
        (axes[1, 1], "loop_lengths", "loop (lambda)", "tab:red"),
    ]:
        for s in sizes:
            vals = data[str(s)][field]
            ax.scatter(np.full(len(vals), s), vals, s=6, color=color, alpha=0.25,
                       edgecolors="none")
        ax.set_xlabel("grid size")
        ax.set_ylabel(label)
        ax.set_title("individual " + label + " samples")
        if args.logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(sizes)} sizes: {sizes.min()}-{sizes.max()})")


if __name__ == "__main__":
    main()
