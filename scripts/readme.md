How it works

Algorithm — Brent's cycle detection. The key constraint for "very large grids" is memory: storing the full trajectory to find a repeat would cost steps × batch × H × W bytes. Brent's algorithm finds both the cycle length λ and transient μ using only O(1) stored boards (a tortoise + a hare), regardless of how long the trajectory is, and uses fewer step evaluations than Floyd's.

- Phase 1 finds the cycle length λ (teleporting tortoise at power-of-two intervals).
- Phase 2a runs a fresh hare λ steps ahead of the tortoise.
- Phase 2b steps both in lockstep; the number of steps until they meet is the transient μ.

Speed choices, kept explicit in the code:

- Entire detection runs inside jax.lax.while_loop under jit → stays on the GPU, no host round-trips.
- Boards stored as uint8 (memory bandwidth is the bottleneck at scale).
- Neighbour sum via a separable roll-and-sum — 4 shifts instead of 8; max value 8 still fits uint8. Toroidal uses wrapping rolls; fixed boundary pads a zero ring and slices it back off (which also discards out-of-bounds births).
- Whole batch runs together; finished elements are frozen with masks until the slowest one converges.

Results

- Correctness: 64×64 and 32×32 boards matched a dict-based brute-force reference exactly across both boundary conditions and a range of μ/λ values.
- Scale: 4 × 2048² toroidal boards, transients ~10k–15k steps, all converged in ~8.7s on the CUDA device.

Usage

```
python scripts/gol_cycle.py --batch 4 --size 2048 --density 0.3 --toroidal --max-steps 200000
````
--toroidal selects periodic boundaries (omit for fixed/zero). Non-converging elements within --max-steps are reported as -1.

- One note: the default --max-steps 1_000_000 is a safety cap — for very large grids whose transients you don't yet know, set it generously, since an element that hits the cap is reported as -1 rather than silently truncated.

