#!/usr/bin/env python3
"""Spreading from a localized random patch in the prepared sea.

A ring of N = k 2^B + r cells is prepared as the sea (cells 0, words
i.i.d. uniform in [0, r)); a patch of `w0` cells in the middle gets
random cells and random words in [0, 2^B).  Each run records, at the
times in `times`, whether it is alive (a run is dead once it has no
1-cell and every word is below 2^(B-1)), the number of cells changed in
that step, the number of 1-cells, and the half-extent of the changed
cells about the patch centre.  Runs stop when dead.

usage: spread.py B r nruns T [w0=32] [k=16] [seed0=1] [procs=2] [tag]
output: spread_B{B}_r{r}_w{w0}_{tag}.npz with arrays
    times, alive[nruns, nt], nch[nruns, nt], n1[nruns, nt], ext[nruns, nt], tdead[nruns]
"""
import sys, time
import numpy as np
from multiprocessing import Pool
import nuca_fast as nf


def one(args):
    B, r, k, w0, seed, T, times = args
    base = 1 << B; N = k * base + r
    rng = np.random.default_rng(seed)
    c = np.zeros(N, dtype=np.int64)
    n = rng.integers(0, r, N).astype(np.int64)
    x0 = N // 2; a = x0 - w0 // 2
    c[a:a + w0] = rng.integers(0, 2, w0)
    n[a:a + w0] = rng.integers(0, base, w0)
    nch, n1, lo, hi, td = nf.run(c, n, r, B, T, stop_dead=True)
    idx = times - 1
    alive = np.ones(len(times), dtype=bool) if td == 0 else (times < td)
    ext = np.where(lo[idx] >= 0, np.maximum(hi[idx] - x0, x0 - lo[idx]), 0)
    return alive, nch[idx], n1[idx], ext, td


def main():
    B, r, nruns, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    w0 = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    k = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    seed0 = int(sys.argv[7]) if len(sys.argv) > 7 else 1
    procs = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    tag = sys.argv[9] if len(sys.argv) > 9 else 'a'
    times = np.unique(np.round(np.logspace(0, np.log10(T), 120)).astype(int))
    jobs = [(B, r, k, w0, seed0 + i, T, times) for i in range(nruns)]
    t0 = time.time()
    with Pool(procs) as p:
        res = p.map(one, jobs, chunksize=1)
    alive = np.array([x[0] for x in res]); nch = np.array([x[1] for x in res])
    n1 = np.array([x[2] for x in res]); ext = np.array([x[3] for x in res]); td = np.array([x[4] for x in res])
    fn = f"spread_B{B}_r{r}_w{w0}_{tag}.npz"
    np.savez(fn, times=times, alive=alive, nch=nch, n1=n1, ext=ext, tdead=td, B=B, r=r, k=k, w0=w0, seed0=seed0)
    P = alive.mean(0); Nn = nch.mean(0); Rext = np.array([ext[alive[:, j], j].mean() if alive[:, j].any() else 0 for j in range(len(times))])
    print(f"B={B} r={r} N={k*(1<<B)+r} w0={w0} runs={nruns} T={T} ({time.time()-t0:.0f}s); dead runs: {(td>0).sum()}")
    print("      t      P(t)     N_ch(t)   N_1(t)   R_ext(t|alive)")
    for j, t in enumerate(times):
        if j % 6 == 0 or j == len(times) - 1:
            print(f"{t:8d}  {P[j]:.4f}  {Nn[j]:9.2f}  {n1.mean(0)[j]:8.2f}  {Rext[j]:9.1f}")


if __name__ == '__main__':
    main()
