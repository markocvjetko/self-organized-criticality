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
    return lam, mu


def random_boards(key, batch, size, density):
    return (jax.random.uniform(key, (batch, size, size)) < density).astype(jnp.uint8)


def _frame_label(t: int, mu_i: int, lam_i: int) -> tuple[str, str]:
    """Return (filename_tag, title_phase) describing the trajectory point t.

    Uses the detected transient ``mu_i`` and cycle length ``lam_i`` to classify
    state x_t. States 0..mu_i-1 are the transient; from mu_i onward we are in the
    cycle, and (t - mu_i) % lam_i is the position within it.
    """
    if lam_i < 0:  # this board never converged within max_steps
        return "unconverged", f"step {t} (not converged)"
    if t < mu_i:
        return "transient", f"transient {t + 1}/{mu_i}"
    k = (t - mu_i) % lam_i
    return "cycle", f"cycle {k}/{lam_i}"


def render_trajectories(x0, lam, mu, toroidal: bool, out_dir: str, max_frames: int):
    """Replay each board from x0 and save every state as a labeled PNG.

    For a converged board we render states x_0 .. x_{mu+lam}, i.e. the full
    transient plus one full cycle (the last frame equals x_mu, visually closing
    the loop). Non-converged boards are rendered for ``max_frames`` steps. Every
    board's frames go in its own ``board_<i>/`` subfolder; each frame is capped
    at ``max_frames`` to keep long transients from exploding into millions of
    images.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    step = make_step(toroidal)
    B = int(x0.shape[0])
    lam = np.asarray(jax.device_get(lam))
    mu = np.asarray(jax.device_get(mu))

    # Frames to render per board, then clamp to the safety cap.
    want = np.where(lam >= 0, mu + lam + 1, max_frames).astype(np.int64)
    frames = np.minimum(want, max_frames)
    truncated = want > frames
    n_total = int(frames.max()) if B else 0
    width = max(4, len(str(max(0, n_total - 1))))

    out = Path(out_dir)
    dirs = []
    for i in range(B):
        d = out / f"board_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)

    print(f"\nrendering up to {n_total} frames/board into {out}/ ...")
    g = x0
    for t in range(n_total):
        g_host = np.asarray(jax.device_get(g))
        for i in range(B):
            if t >= frames[i]:
                continue
            tag, phase = _frame_label(t, int(mu[i]), int(lam[i]))
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(g_host[i], cmap="binary", interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"board {i} | step {t} | {phase}\n(mu={int(mu[i])}, lambda={int(lam[i])})",
                         fontsize=9)
            fig.savefig(dirs[i] / f"step_{t:0{width}d}_{tag}.png",
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        g = step(g)

    for i in range(B):
        note = "  (TRUNCATED at max-render-frames)" if truncated[i] else ""
        print(f"  board {i}: {int(frames[i])} frames -> {dirs[i]}{note}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=32, help="number of boards")
    p.add_argument("--size", type=int, default=1024, help="grid side length (square)")
    p.add_argument("--density", type=float, default=0.3, help="initial alive fraction")
    p.add_argument("--toroidal", action="store_true", help="periodic boundaries")
    p.add_argument("--max-steps", type=int, default=1_000_000,
                   help="give up (and report -1) after this many steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render-dir", type=str, default=None,
                   help="if set, save every board state as a labeled PNG under DIR/board_<i>/")
    p.add_argument("--render-max-frames", type=int, default=500,
                   help="cap on frames rendered per board (transient+cycle can be huge)")
    args = p.parse_args()

    print(f"device: {jax.devices()[0]}")
    print(f"batch={args.batch} grid={args.size}x{args.size} "
          f"density={args.density} toroidal={args.toroidal}")

    key = jax.random.PRNGKey(args.seed)
    x0 = random_boards(key, args.batch, args.size, args.density)

    # Warm up the JIT so the timed run measures execution, not compilation.
    t0 = time.perf_counter()
    lam, mu = detect_cycle(x0, args.toroidal, args.max_steps)
    lam.block_until_ready()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    lam, mu = detect_cycle(x0, args.toroidal, args.max_steps)
    lam.block_until_ready()
    run_s = time.perf_counter() - t0

    lam = jax.device_get(lam)
    mu = jax.device_get(mu)

    print(f"compile+first run: {compile_s:.3f}s | timed run: {run_s:.3f}s\n")
    print(f"{'idx':>4}  {'transient (mu)':>14}  {'cycle (lambda)':>14}")
    for i, (m, l) in enumerate(zip(mu, lam)):
        note = "  (not converged)" if l < 0 else ""
        print(f"{i:>4}  {int(m):>14}  {int(l):>14}{note}")

    ok = lam >= 0
    if ok.any():
        print(f"\nconverged {int(ok.sum())}/{len(lam)}  "
              f"| transient: mean={mu[ok].mean():.1f} max={int(mu[ok].max())}  "
              f"| cycle: mean={lam[ok].mean():.1f} max={int(lam[ok].max())}")

    if args.render_dir is not None:
        render_trajectories(x0, lam, mu, args.toroidal,
                            args.render_dir, args.render_max_frames)


if __name__ == "__main__":
    main()
