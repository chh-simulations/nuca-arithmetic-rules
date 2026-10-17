#!/usr/bin/env python3
"""Front propagation into the prepared sea.

A ring of N = k 2^B + r cells: the middle block [N/4, 3N/4) is random
(cells 0/1, words uniform in [0, 2^B)), the rest is the prepared sea
(cells 0, words uniform in [0, r)), which is absorbing, so the outermost
cells that change in a step are the two fronts.  Their positions are
recorded every `every` steps; the velocity is the slope of each front
over the second half of the run (or of the part before the front has
come within 64 cells of the ring's antipode), averaged over the two
fronts.

usage: front.py B nruns T [k=24] [every=1000] [seed0=1] [procs=2] r1 r2 ...
output: front_B{B}.npz (appended per r): pos[r][run] = (t, xlo, xhi) series
"""
import sys, time, os
import numpy as np
from multiprocessing import Pool
import nuca_fast as nf


def one(args):
    B, r, k, seed, T, every = args
    base = 1 << B; N = k * base + r
    rng = np.random.default_rng(seed)
    c = np.zeros(N, dtype=np.int64); n = rng.integers(0, r, N).astype(np.int64)
    a, b = N // 4, 3 * N // 4
    c[a:b] = rng.integers(0, 2, b - a); n[a:b] = rng.integers(0, base, b - a)
    nch, n1, lo, hi, td = nf.run(c, n, r, B, T, stop_dead=True)
    idx = np.arange(every - 1, T, every)
    return idx + 1, lo[idx], hi[idx], nch[idx], td


def velocity(t, lo, hi, N, a, b):
    """slope of the right front (hi - b) and of the left front (a - lo), each
    fitted over the second half of the valid range (front not yet within 64
    cells of the antipode, i.e. hi < N - 64 and lo > 64); returns (v_right, v_left)."""
    out = []
    for pos in (hi - b, a - lo):
        valid = (hi < N - 64) & (lo > 64) & (lo >= 0)
        tt, pp = t[valid], pos[valid]
        if tt.size < 10:
            out.append(float('nan')); continue
        h = tt.size // 2
        out.append(float(np.polyfit(tt[h:], pp[h:], 1)[0]))
    return out


def main():
    B, nruns, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    k = int(sys.argv[4]); every = int(sys.argv[5]); seed0 = int(sys.argv[6]); procs = int(sys.argv[7])
    rs = [int(x) for x in sys.argv[8:]]
    base = 1 << B
    fn = f"front_B{B}.npz"
    store = dict(np.load(fn, allow_pickle=True)) if os.path.exists(fn) else {}
    for r in rs:
        N = k * base + r; a, b = N // 4, 3 * N // 4
        jobs = [(B, r, k, seed0 + i, T, every) for i in range(nruns)]
        t0 = time.time()
        with Pool(procs) as p:
            res = p.map(one, jobs, chunksize=1)
        vs = []
        for (t, lo, hi, nch, td) in res:
            vr, vl = velocity(t, lo, hi, N, a, b)
            vs.append((vr, vl))
        store[f"r{r}"] = np.array([np.stack([t, lo, hi, nch]) for (t, lo, hi, nch, td) in res], dtype=object)
        store[f"v{r}"] = np.array(vs)
        np.savez(fn, **store)
        e1 = (r - base // 4) / r
        vs = np.array(vs)
        print(f"B={B} r={r} e1={e1:.5f} N={N} T={T} runs={nruns} ({time.time()-t0:.0f}s): "
              f"v_right {np.array2string(vs[:,0]*1e3, precision=3)}  v_left {np.array2string(vs[:,1]*1e3, precision=3)}  "
              f"mean {np.nanmean(vs)*1e3:.4f}e-3  final hi-b {[int(x[2][-1]-b) for x in res]}", flush=True)


if __name__ == '__main__':
    main()
