#!/usr/bin/env python3
"""Developed-phase fronts against a NATURAL sea: the sea left behind by a
random initial state that collapsed at a residue below threshold (its
clock phases are not uniform: the flicker around the dying islands
drained them), compared with the prepared sea of uniform phases.

For each seed: a random ring of N = k 2^B + r_sea cells is run at r_sea
for t_sea steps and must be dead (no cell on, every word below r_sea);
its words are the natural sea.  The active phase is prepared at
r_prep = 2^(B-1) for t_prep steps; the middle half of a ring of N cells
gets the active phase and the outer halves the natural sea (or, for the
control, uniform words in [0, r)); the ring is run at residue r >= r_sea
for T steps and the two fronts (outermost changed cells) are followed;
each front's velocity is the net displacement between T/10 and T over
0.9 T.  Also recorded: the phase histogram of the natural sea in eight
bins of [0, r_sea).

usage: front_natural.py B k r_sea t_sea t_prep nseeds r:T [r:T ...]
output: front_natural_B{B}.jsonl
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def fronts(c, n, r, B, T, a, b):
    nch, n1, lo, hi, td = nf.run(c, n, r, B, T, stop_dead=True)
    every = max(1, T // 200)
    idx = np.arange(every - 1, T, every)
    t = idx + 1; L = a - lo[idx]; R = hi[idx] - b
    ok = lo[idx] >= 0
    j0 = int(np.searchsorted(t, T / 10))
    vL = (L[-1] - L[j0]) / (t[-1] - t[j0]); vR = (R[-1] - R[j0]) / (t[-1] - t[j0])
    return float(vL), float(vR), int(L[-1]), int(R[-1]), int(td)


def main():
    B, k, rsea, tsea, tprep, nseeds = (int(float(x)) for x in sys.argv[1:7])
    jobs = [(int(x.split(':')[0]), int(float(x.split(':')[1]))) for x in sys.argv[7:]]
    base = 1 << B; rprep = base // 2; rc = 3 * (1 << (B - 3))
    out = open(f"front_natural_B{B}.jsonl", "a")
    for r, T in jobs:
        N = k * base + r; a, b = N // 4, 3 * N // 4
        res = dict(B=B, r=r, rc=rc, r_sea=rsea, t_sea=tsea, N=N, T=T, runs=[])
        for seed in range(1, nseeds + 1):
            t0 = time.time()
            # natural sea
            Ns = k * base + rsea
            cs, ns = nf.ic_random(Ns, B, 500 + seed)
            nf.run(cs, ns, rsea, B, tsea, stop_dead=False)
            dead = bool(cs.sum() == 0 and ns.max() < rsea)
            hist = np.histogram(ns, bins=8, range=(0, rsea))[0].tolist() if dead else None
            # active phase
            c0, n0 = nf.ic_random(k * base + rprep, B, seed)
            nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
            rec = dict(seed=seed, natural_sea_dead=dead, sea_hist=hist)
            for kind in ('natural', 'uniform'):
                if kind == 'natural' and not dead:
                    continue
                c = c0[:N].copy(); n = n0[:N].copy()
                if kind == 'natural':
                    c[:a] = 0; n[:a] = ns[:a]; c[b:] = 0; n[b:] = ns[b:N] if N <= Ns else np.concatenate([ns[b:], ns[:N - Ns]])[:N - b]
                else:
                    rng = np.random.default_rng(10_000 + seed)
                    c[:a] = 0; n[:a] = rng.integers(0, r, a); c[b:] = 0; n[b:] = rng.integers(0, r, N - b)
                vL, vR, L, R, td = fronts(c, n, r, B, T, a, b)
                rec[kind] = dict(v=[vL, vR], final=[L, R], tdead=td)
            res['runs'].append(rec)
            print(f"B={B} r={r} seed={seed}: natural sea dead={dead}; " + "; ".join(f"{kind} v={rec[kind]['v'][0]:.2e},{rec[kind]['v'][1]:.2e}" for kind in ('natural', 'uniform') if kind in rec) + f"  ({time.time()-t0:.0f}s)", flush=True)
        for kind in ('natural', 'uniform'):
            v = np.array([x[kind]['v'] for x in res['runs'] if kind in x], dtype=float).ravel()
            res[f'v_{kind}_mean'] = float(v.mean()) if v.size else None
            res[f'v_{kind}_sem'] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None
        out.write(json.dumps(res) + "\n"); out.flush()
        print(f"B={B} r={r} (r_c{r-rc:+d}) T={T:.0e}: natural {res['v_natural_mean']} +- {res['v_natural_sem']}, uniform {res['v_uniform_mean']} +- {res['v_uniform_sem']}", flush=True)
    out.close()


if __name__ == '__main__':
    main()
