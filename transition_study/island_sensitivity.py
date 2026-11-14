"""P3 Sec. V A: the +1 perturbation of one core word of the transplanted island (suite S3), followed beyond 3000 steps:
how many cells and words differ, and how far from the island the differences reach."""
import numpy as np
import nuca_fast as nf
B = 10; base = 1 << B; r = 376; N = 2 * base + r
c, n = nf.ic_random(N, B, 1)
nf.run(c, n, r, B, 200000, stop_dead=False)
isl = nf.islands(c, n, r, B, steps=4000)
a, b = isl[0]; a -= 8; b += 9
rng = np.random.default_rng(100); cs = np.zeros(N, dtype=np.int64); ns = rng.integers(0, r, N).astype(np.int64)
x0 = N // 2; cs[x0:x0 + (b - a)] = c[a:b]; ns[x0:x0 + (b - a)] = n[a:b]
cA, nA = cs.copy(), ns.copy(); cB, nB = cs.copy(), ns.copy(); mid = x0 + (b - a) // 2; nB[mid] = (nB[mid] + 1) & (base - 1)
t = 0
for T in (10, 100, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 2000000):
    nf.run(cA, nA, r, B, T - t, stop_dead=False); nf.run(cB, nB, r, B, T - t, stop_dead=False); t = T
    dw = np.flatnonzero(nA != nB); dc = np.flatnonzero(cA != cB)
    print("t=%8d: cells differing %2d, words differing %2d, span of differing words [%d, %d] relative to the transplanted segment [0, %d)" % (
        T, dc.size, dw.size, (dw.min() - x0) if dw.size else 0, (dw.max() - x0) if dw.size else 0, b - a))
