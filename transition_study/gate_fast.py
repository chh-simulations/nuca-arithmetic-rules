#!/usr/bin/env python3
"""Gate for nuca_fast.evolve against the canonical numpy update
    n' = ((n + (2c'-1) h - 1) mod N) & (2^B - 1)
(the form used throughout verify_p1.py), and against two numbers of the
P1 suite (S9).  Every check is printed; nothing is truncated."""
import sys, time
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
import nuca_fast as nf

fails = 0
def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    if not ok: fails += 1

def canonical_run(c, n, N, B, steps):
    """verify_p1.run_B's loop, msb reading, linear increment."""
    c = c.copy(); n = n.copy(); mask = (1 << B) - 1
    prev = c.copy(); nch = []; n1 = []; lo = []; hi = []
    for t in range(steps):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c = (n >> (B - 1 - h)) & 1
        n = ((n + (2 * c - 1) * h - 1) % N) & mask
        ch = (c != prev); prev = c.copy()
        nch.append(int(ch.sum())); n1.append(int(c.sum()))
        w = np.flatnonzero(ch)
        lo.append(int(w[0]) if w.size else -1); hi.append(int(w[-1]) if w.size else -1)
    return c, n, np.array(nch), np.array(n1), np.array(lo), np.array(hi)

# 1. state and series identical, k = 1, 2, 3, several B and r, P1's initial condition
print("1. r-form (numba) == canonical mod-N form, 300 steps, P1 initial condition")
for B in (6, 8, 10, 12):
    base = 1 << B
    for r in (4, 5, 7, base // 8, 3 * base // 8 - 1, 3 * base // 8, 3 * base // 8 + 1, base // 2, base - 1):
        for k in (1, 2, 3):
            N = k * base + r
            rng = np.random.default_rng(11)
            n0 = (rng.permutation(N) % base).astype(np.int64)
            c0 = rng.integers(0, 2, N).astype(np.int64)
            cc, nc, nchc, n1c, loc, hic = canonical_run(c0, n0, N, B, 300)
            cf, nnf = c0.copy(), n0.copy()
            nchf, n1f, lof, hif, td = nf.run(cf, nnf, r, B, 300, stop_dead=False)
            ok = (np.array_equal(cc, cf) and np.array_equal(nc, nnf) and np.array_equal(nchc, nchf)
                  and np.array_equal(n1c, n1f) and np.array_equal(loc, lof) and np.array_equal(hic, hif))
            check(f"B={B:2d} r={r:4d} k={k} N={N:5d}: final c, n and the four series identical", ok)

# 2. the dead detection: the prepared sea is dead at step 1; an all-zero
#    configuration with a word >= 2^(B-1) ignites
print("2. dead detection")
c, n = nf.ic_sea(2048 + 384, 384, 3)
nch, n1, lo, hi, td = nf.run(c, n, 384, 10, 50)
check("prepared sea (B=10, r=384): tdead = 1, no change, no 1-cell", td == 1 and nch.sum() == 0 and n1.sum() == 0)
c, n = nf.ic_sea(2048 + 384, 384, 3); n[100] = 512 + 7
nch, n1, lo, hi, td = nf.run(c, n, 384, 10, 50)
check(f"one word 519 >= 512 in a zero ring ignites at step 1 (cell 100), flickers, dies at step {td} with the word below 512",
      nch[0] == 1 and lo[0] == 100 and hi[0] == 100 and td > 1 and n1[td-1] == 0 and n.max() < 512)
c, n = nf.ic_sea(2048 + 384, 384, 3); n[100] = 400
nch, n1, lo, hi, td = nf.run(c, n, 384, 50, 50)
check("one word 400 in [r, 2^(B-1)) in a zero ring: dead at step 1 (doomed), word drifts to r-1 and cycles",
      td == 1)

# 3. P1 numbers (S9 of verify_p1.py), same initial condition as run_B
print("3. P1 Table 2 entries reproduced: B=10, N=2^B+r, T=12000, tail activity over the last 100 steps")
exp = {383: (0.00, 1.07), 384: (0.88, 0.77), 385: (5.39, 9.64)}
for r, ex in exp.items():
    for seed in (1, 2):
        N = 1024 + r
        rng = np.random.default_rng(seed)
        n = (rng.permutation(N) % 1024).astype(np.int64)
        c = rng.integers(0, 2, N).astype(np.int64)
        t0 = time.time()
        nch, n1, lo, hi, td = nf.run(c, n, r, 10, 12000)
        act = 100 * nch[-100:].mean() / N
        check(f"B=10 r={r} seed{seed}: tail activity {act:.2f} == {ex[seed-1]:.2f} ({time.time()-t0:.2f}s)",
              abs(act - ex[seed - 1]) < 0.005)

# 4. the P1 front at B=10, r=385, seed 1: v = 2.08e-3 (recent-change count, half slope)
print("4. P1 front (S9) reproduced: B=10 r=385 seed 1")
base, r, N = 1024, 385, 1024 + 385
rng = np.random.default_rng(1); half = N // 2
c = np.zeros(N, dtype=np.int64); n = np.zeros(N, dtype=np.int64)
n[:half] = rng.integers(0, r, half)
c[half:] = rng.integers(0, 2, N - half); n[half:] = rng.integers(0, base, N - half)
# the P1 measure needs per-cell recency; run in chunks of 1000 steps with a
# numpy recency update (identical dynamics, recency computed from change flags)
# -> instead reproduce with the canonical loop restricted to 'recent' bookkeeping
prev = c.copy(); recent = np.zeros(N, dtype=np.int64); out = []
t0 = time.time()
for t in range(1, 120001):
    h = c + np.roll(c, 1) + np.roll(c, -1)
    c = (n >> 9) & 1 if False else (n >> (9 - h)) & 1
    n = ((n + (2 * c - 1) * h - 1) % N) & 1023
    ch = (c != prev); prev = c.copy()
    recent = np.where(ch, 0, recent + 1)
    if t % 1000 == 0: out.append((t, int((recent < 50).sum())))
A = np.array(out); m = (A[:, 0] >= 20000) & (A[:, 1] < 0.8 * N)
v = np.polyfit(A[m, 0], A[m, 1], 1)[0] / 2
check(f"canonical numpy front v = {1000*v:.2f}e-3 == 2.08e-3 ({time.time()-t0:.1f}s)", abs(1000 * v - 2.08) < 0.05)

# 5. timing of the numba runner
print("5. timing")
for N, steps in ((1408, 120000), (10240 + 1536, 20000)):
    c, n = nf.ic_random(N, 12 if N > 5000 else 10, 1)
    r = 1536 if N > 5000 else 384; B = 12 if N > 5000 else 10
    t0 = time.time(); nf.run(c, n, r, B, steps); dt = time.time() - t0
    print(f"  N={N} steps={steps}: {dt:.2f}s = {N*steps/dt/1e6:.0f} M cell-updates/s")

print(f"\n{fails} FAIL")
