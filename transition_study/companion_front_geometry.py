"""The front geometry of the companion paper (B = 10, N = 4*2^10 + r cells, prepared sea in [0, N/2), random half in
[N/2, N), 2e5 steps; its ten runs with r <= r_c): where the surviving activity of the random half sits relative to the
two boundaries with the sea at the end of the run.  Cells that changed during the last 2000 steps; distance = number of
quiet cells of the random half between the boundary and the nearest changing cell; and the sea cells that changed or
hold a word outside [0, r)."""
import numpy as np
import nuca_fast as nf
B = 10; base = 1 << B; dist = []
for r in (376, 380, 382, 383, 384):
    for seed in (1, 2):
        N = 4 * base + r; half = N // 2
        rng = np.random.default_rng(seed)
        c = np.zeros(N, dtype=np.int64); n = np.zeros(N, dtype=np.int64)
        n[:half] = rng.integers(0, r, half)
        c[half:] = rng.integers(0, 2, N - half); n[half:] = rng.integers(0, base, N - half)
        nf.run(c, n, r, B, 198000, stop_dead=False)
        mark = np.zeros(N, dtype=np.int64)
        nf.evolve_mark(c, n, np.int64(r), np.int64(B), np.int64(2000), mark)
        idx = np.flatnonzero(mark[half:]) + half
        sea_changed = np.flatnonzero(mark[:half]); conv = int((n[:half] >= r).sum())
        d1, d2 = int(idx.min() - half), int(N - 1 - idx.max())
        dist += [d1, d2]
        print("r=%d seed %d: nearest changing cell of the random half: %3d and %3d cells from the two boundaries; sea cells that changed in the last 2000 steps: %s; sea cells with a word outside [0, r): %d" % (
            r, seed, d1, d2, list(sea_changed), conv))
dist = sorted(dist)
print("all twenty boundaries, sorted:", dist)
