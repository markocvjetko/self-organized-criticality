#!/usr/bin/env python
"""Merge the per-seed JSON files produced by sweep_seeds.slurm into one file.

Each array task writes ``data/data_seed<SEED>.json`` with the schema created in
``gol_cycle_bara.py`` (a dict keyed by grid size). This script concatenates the
samples for each size across all seeds, re-stitches the per-batch global offsets,
and recomputes the summary statistics so the merged file looks exactly like one
big run.

Usage:
    python merge_seeds.py "data/data_seed*.json" --out data/merged.json
    python merge_seeds.py data/*.json --out data/merged.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# Stats keys are derived, not merged directly; params/batches handled specially.
ARRAY_KEYS = ("transient_lengths", "loop_lengths")


def _recompute_stats(entry: dict) -> None:
    """Refresh n_samples and the mean/std/min/max fields from the arrays."""
    mu = np.asarray(entry["transient_lengths"], dtype=np.float64)
    lam = np.asarray(entry["loop_lengths"], dtype=np.float64)
    entry["n_samples"] = int(len(mu))
    if len(mu) == 0:
        return
    entry["transient_mean"] = float(np.mean(mu))
    entry["transient_std"] = float(np.std(mu))
    entry["transient_min"] = int(np.min(mu))
    entry["transient_max"] = int(np.max(mu))
    entry["loop_mean"] = float(np.mean(lam))
    entry["loop_std"] = float(np.std(lam))
    entry["loop_min"] = int(np.min(lam))
    entry["loop_max"] = int(np.max(lam))


def merge(paths: list[str], strict: bool = True) -> dict:
    merged: dict = {}
    seen_seeds: dict[str, set] = {}

    for path in sorted(paths):
        with open(path) as f:
            data = json.load(f)

        for size_key, entry in data.items():
            if size_key not in merged:
                # Deep-ish copy: start fresh arrays/batches, keep params.
                merged[size_key] = {
                    "transient_lengths": list(entry["transient_lengths"]),
                    "loop_lengths": list(entry["loop_lengths"]),
                    "n_samples": 0,
                    "transient_mean": None, "transient_std": None,
                    "transient_min": None, "transient_max": None,
                    "loop_mean": None, "loop_std": None,
                    "loop_min": None, "loop_max": None,
                    "params": entry["params"],
                    "batches": [dict(b) for b in entry["batches"]],
                }
                seen_seeds[size_key] = set()
            else:
                tgt = merged[size_key]
                if entry["params"] != tgt["params"]:
                    msg = (f"params mismatch for size={size_key} in {path}\n"
                           f"  existing: {tgt['params']}\n"
                           f"  incoming: {entry['params']}")
                    if strict:
                        raise ValueError(msg)
                    print(f"WARNING (skipping incompatible file): {msg}")
                    continue

                # Concatenate samples, then re-base each incoming batch's
                # global_start/global_end onto the running length.
                offset = len(tgt["transient_lengths"])
                tgt["transient_lengths"].extend(entry["transient_lengths"])
                tgt["loop_lengths"].extend(entry["loop_lengths"])
                for b in entry["batches"]:
                    nb = dict(b)
                    nb["global_start"] = b["global_start"] + offset
                    nb["global_end"] = b["global_end"] + offset
                    tgt["batches"].append(nb)

            # Track seeds to warn about accidental duplicates.
            for b in entry["batches"]:
                s = b.get("seed")
                if s in seen_seeds[size_key]:
                    print(f"WARNING: duplicate seed={s} for size={size_key} "
                          f"(from {path}) — samples will be double-counted")
                seen_seeds[size_key].add(s)

    for entry in merged.values():
        _recompute_stats(entry)

    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+",
                   help="input JSON files or globs (e.g. 'data/data_seed*.json')")
    p.add_argument("--out", required=True, help="output merged JSON path")
    p.add_argument("--no-strict", action="store_true",
                   help="skip files with mismatched params instead of erroring")
    args = p.parse_args()

    # Expand globs that the shell didn't (e.g. when quoted).
    paths: list[str] = []
    for pattern in args.inputs:
        hits = glob.glob(pattern)
        paths.extend(hits if hits else [pattern])
    paths = sorted(set(paths))

    if not paths:
        raise SystemExit("no input files matched")

    out_path = Path(args.out).resolve()
    paths = [p for p in paths if Path(p).resolve() != out_path]  # don't eat our own output

    print(f"merging {len(paths)} files -> {args.out}")
    merged = merge(paths, strict=not args.no_strict)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)

    for size_key in sorted(merged, key=int):
        e = merged[size_key]
        print(f"  size={size_key}: n_samples={e['n_samples']} "
              f"({len(e['batches'])} batches)")


if __name__ == "__main__":
    main()
