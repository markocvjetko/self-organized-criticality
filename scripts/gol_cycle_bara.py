#!/usr/bin/env python
"""Measure transient length and cycle length of Game of Life trajectories.

Because the state space is finite, every Game of Life trajectory is eventually
periodic: after a transient of ``mu`` steps it enters a cycle of length
``lambda``. This script detects both for a batch of random initial boards.

Design notes (why it is fast on very large grids)
-------------------------------------------------
* Cycle detection uses **Brent's algorithm**, which needs only O(1) stored
  states (a "tortoise" and a "hare") rather than the full trajectory history.
  Storing history would cost ``steps * batch * H * W`` bytes — infeasible for
  large grids. Brent's keeps memory at ~3 board copies regardless of cycle
  length, and uses fewer evaluations than Floyd's algorithm.
* The whole detection runs inside a single ``jax.lax.while_loop`` under ``jit``,
  so the entire search stays on the accelerator with no Python/host round trips.
* Boards are stored as ``uint8`` to minimise memory bandwidth (the bottleneck
  for large grids). The neighbour count is computed with a separable
  roll-and-sum (4 shifts instead of 8), max value 8 still fits in ``uint8``.
* The batch is processed together; finished elements are frozen via masks while
  the loop continues until the slowest element has found its cycle.
"""
from __future__ import annotations

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import sys
from pathlib import Path


def make_step(toroidal: bool):
    """Return a single-step Game of Life update for boards of shape (B, H, W)."""

    def step(grid: jnp.ndarray) -> jnp.ndarray:
        # Pad with a zero border for fixed boundaries; for toroidal boundaries
        # the rolls already wrap, so no padding is needed.
        g = grid if toroidal else jnp.pad(grid, ((0, 0), (1, 1), (1, 1)))

        # Separable 3x3 neighbour sum: sum across columns, then across rows.
        # `full` includes the cell itself; subtract it to get the 8-neighbour count.
        row = g + jnp.roll(g, 1, axis=2) + jnp.roll(g, -1, axis=2)
        full = row + jnp.roll(row, 1, axis=1) + jnp.roll(row, -1, axis=1)
        n = full - g

        if not toroidal:
            n = n[:, 1:-1, 1:-1]  # discard the padding ring (and any births in it)

        alive = grid == 1
        new = (n == 3) | (alive & (n == 2))
        return new.astype(jnp.uint8)

    return step


def _equal(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Per-batch-element board equality -> (B,) bool."""
    return jnp.all(a == b, axis=(1, 2))


@partial(jax.jit, static_argnums=(1, 2))
def detect_cycle(x0: jnp.ndarray, toroidal: bool, max_steps: int):
    """Brent's algorithm, batched.

    Returns ``(lam, mu)`` arrays of shape (B,): cycle length and transient
    length. Elements that do not converge within ``max_steps`` get value -1.
    """
    step = make_step(toroidal)
    B = x0.shape[0]

    def grid_mask(m):  # broadcast a (B,) mask over the board
        return m[:, None, None]

    # ----- Phase 1: find the cycle length lambda -----
    # Carry: tortoise, hare, power (next teleport distance), lam, found, steps.
    def p1_cond(c):
        _, _, _, _, found, steps = c
        return jnp.any(~found) & (steps < max_steps)

    def p1_body(c):
        tort, hare, power, lam, found, steps = c
        eq = _equal(tort, hare)
        active = (~found) & (~eq)
        # Teleport the tortoise to the hare every power-of-two steps.
        reset = active & (power == lam)
        tort = jnp.where(grid_mask(reset), hare, tort)
        power = jnp.where(reset, power * 2, power)
        lam = jnp.where(reset, 0, lam)
        # Advance the hare and grow the measured cycle length.
        hare = jnp.where(grid_mask(active), step(hare), hare)
        lam = jnp.where(active, lam + 1, lam)
        return tort, hare, power, lam, found | eq, steps + 1    # Warm up the JIT so the timed run measures execution, not compilation.


    ones = jnp.ones(B, jnp.int32)
    init1 = (x0, step(x0), ones, ones, jnp.zeros(B, bool), jnp.int32(0))
    _, _, _, lam, found1, _ = jax.lax.while_loop(p1_cond, p1_body, init1)

    # ----- Phase 2a: advance a fresh hare by lambda steps ahead of the tortoise -----
    max_lam = jnp.max(lam)

    def adv_cond(c):
        _, i = c
        return i < max_lam

    def adv_body(c):
        hare, i = c
        do = i < lam
        return jnp.where(grid_mask(do), step(hare), hare), i + 1

    hare, _ = jax.lax.while_loop(adv_cond, adv_body, (x0, jnp.int32(0)))

    # ----- Phase 2b: step tortoise and hare in lockstep until they meet -----
    # The number of steps until they coincide is the transient length mu.
    def p2_cond(c):
        _, _, _, done, steps = c
        return jnp.any(~done) & (steps < max_steps)

    def p2_body(c):
        tort, hare, mu, done, steps = c
        eq = _equal(tort, hare)
        active = (~done) & (~eq)
        tort = jnp.where(grid_mask(active), step(tort), tort)
        hare = jnp.where(grid_mask(active), step(hare), hare)
        mu = jnp.where(active, mu + 1, mu)
        
        return tort, hare, mu, done | eq, steps + 1

    init2 = (x0, hare, jnp.zeros(B, jnp.int32), jnp.zeros(B, bool), jnp.int32(0))
    _, _, mu, found2, _ = jax.lax.while_loop(p2_cond, p2_body, init2)

    converged = found1 & found2
    lam = jnp.where(converged, lam, -1)
    mu = jnp.where(converged, mu, -1)
    status = jnp.where(converged, 0, 1)

    return lam, mu, status


def random_boards(key, batch, size, density):
    return (jax.random.uniform(key, (batch, size, size)) < density).astype(jnp.uint8)

def append_size_result_json(
    json_path,
    size,
    mu,
    lam,
    args,
):
    """
    Append one batch of cycle-detection results to a JSON file.

    Top-level JSON structure:

        {
          "25": {
            "transient_lengths": [...],
            "loop_lengths": [...],
            "n_samples": ...,

            "transient_mean": ...,
            "transient_std": ...,
            "transient_min": ...,
            "transient_max": ...,

            "loop_mean": ...,
            "loop_std": ...,
            "loop_min": ...,
            "loop_max": ...,

            "params": {
              "size": 25,
              "density": 0.3,
              "toroidal": true,
              "max_steps": 1000000
            },

            "batches": [
              {
                "seed": 42,
                "batch": 32,
                "local_indices": [0, ..., 31],
                "global_start": 0,
                "global_end": 32,
              }
            ]
          }
        }

    Args:
        json_path:
            Path to output JSON file.

        size:
            Grid side length.

        mu:
            Transient/preperiod lengths, shape (B,).

        lam:
            Period/cycle lengths, shape (B,).

        args:
            argparse args object with at least:
                batch, density, seed, toroidal, max_steps


    Returns:
        The full loaded/updated JSON dictionary.
    """
    json_path = Path(json_path)

    if json_path.exists():
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    mu = np.asarray(mu, dtype=np.int64)
    lam = np.asarray(lam, dtype=np.int64)

    if mu.shape != lam.shape:
        raise ValueError(f"mu and lam must have same shape, got {mu.shape} and {lam.shape}")

    batch_size = int(mu.shape[0])
    size_key = str(int(size))

    current_params = {
        "size": int(size),
        "density": float(args.density),
        "toroidal": bool(args.toroidal),
        "max_steps": int(args.max_steps),
    }

    if size_key not in data:
        data[size_key] = {
            "transient_lengths": [],
            "loop_lengths": [],
            "n_samples": 0,

            "transient_mean": None,
            "transient_std": None,
            "transient_min": None,
            "transient_max": None,

            "loop_mean": None,
            "loop_std": None,
            "loop_min": None,
            "loop_max": None,

            "params": current_params,
            "batches": [],
        }
    else:
        old_params = data[size_key]["params"]

        if old_params != current_params:
            raise ValueError(
                f"Refusing to append incompatible run for size={size_key}.\n"
                f"Existing params:\n{old_params}\n"
                f"New params:\n{current_params}"
            )

    entry = data[size_key]

    global_start = len(entry["transient_lengths"])
    global_end = global_start + batch_size

    batch_record = {
        "seed": int(args.seed),
        "batch": batch_size,
        "local_indices": list(range(batch_size)),
        "global_start": int(global_start),
        "global_end": int(global_end),
    }

    entry["transient_lengths"].extend(mu.tolist())
    entry["loop_lengths"].extend(lam.tolist())
    entry["batches"].append(batch_record)

    all_mu = np.asarray(entry["transient_lengths"], dtype=np.float64)
    all_lam = np.asarray(entry["loop_lengths"], dtype=np.float64)

    entry["n_samples"] = int(len(all_mu))

    entry["transient_mean"] = float(np.mean(all_mu))
    entry["transient_std"] = float(np.std(all_mu))
    entry["transient_min"] = int(np.min(all_mu))
    entry["transient_max"] = int(np.max(all_mu))

    entry["loop_mean"] = float(np.mean(all_lam))
    entry["loop_std"] = float(np.std(all_lam))
    entry["loop_min"] = int(np.min(all_lam))
    entry["loop_max"] = int(np.max(all_lam))

    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=32, help="number of boards")
    p.add_argument("--size", type=int, default=1024, help="grid side length (square)")
    p.add_argument("--density", type=float, default=0.3, help="initial alive fraction")
    p.add_argument("--toroidal", action="store_true", help="periodic boundaries")
    p.add_argument("--max-steps", type=int, default=50_000_000,
                   help="give up (and report -1) after this many steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--minsize",type=int, default=10, help="minimal grid side length")
    p.add_argument("--maxsize",type=int, default=100, help="maximal grid side length")
    p.add_argument("--json-path", type=str, default="gol_cycle_results.json", help="path to JSON file where results are appended after each size")
    args = p.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"batch={args.batch} grid={args.size}x{args.size} "
          f"density={args.density} toroidal={args.toroidal}")

    sizes = np.arange(args.minsize, args.maxsize+1)
    
    for size in sizes:
        key = jax.random.PRNGKey(args.seed)
        x0 = random_boards(key, args.batch, size, args.density)

        lam, mu, status = detect_cycle(x0, args.toroidal, args.max_steps)
        has_failed = jnp.any(status == 1)
        if has_failed:
            failed_idx = jax.device_get(jnp.where(status == 1)[0])
            print("some transient was too long, terminating")
            print(f"size={size}, failed indices={failed_idx.tolist()}")
            sys.exit(1)



        lam.block_until_ready()

    
        lam = jax.device_get(lam)
        mu = jax.device_get(mu)
        status = jax.device_get(status)

        if np.any(status == 1):
            failed_idx = np.flatnonzero(status == 1)
            print("some transient was too long, terminating")
            print(f"size={size}, failed indices={failed_idx.tolist()}")
            sys.exit(1)

        append_size_result_json(
            json_path=args.json_path,
            size=size,
            mu=mu,
            lam=lam,
            args=args,
        )

        print(f"saved/appended results for size={size} to {args.json_path}")


if __name__ == "__main__":
    main()
