#!/usr/bin/env python3
"""Stability of the developed active phase below the threshold.  The
active phase is prepared at r_prep = 2^(B-1) (e_1 = 1/2) for t_prep
steps on a ring of k 2^B + r_prep cells; the configuration and words of
the first N' = k 2^B + r cells are then run at residue r (the state is
untouched; only the arithmetic of the wrap changes).  Recorded: the
activity in log windows and, at five checkpoints, the active domains
(clusters of cells changed within 4000 steps): their number and the
fraction of the ring they cover.  The retreat velocity per front follows
from the loss of covered fraction between checkpoints divided by twice
the number of domains (holes grow at two fronts each).

usage: quench.py B k t_prep T nseeds r1 r2 ...
output: quench_B{B}.jsonl
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def main():
    B, k, tprep, T, nseeds = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rs = [int(x) for x in sys.argv[6:]]
    base = 1 << B; rprep = base // 2; rc = 3 * base // 8
    checkpoints = [int(T * f) for f in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)]
    with open(f"quench_B{B}.jsonl", "a") as out:
        for seed in range(1, nseeds + 1):
            c0, n0 = nf.ic_random(k * base + rprep, B, seed)
            nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
            nchp, _, _, _, _ = nf.run(c0, n0, rprep, B, 2000, stop_dead=False)
            act_prep = 100 * float(nchp.mean()) / len(c0)
            for r in rs:
                N = k * base + r; t0 = time.time()
                c, n = c0[:N].copy(), n0[:N].copy()
                t = 0; series = []; cps = []
                for cp in checkpoints:
                    steps = cp - t - 4000
                    nch, n1, lo, hi, td = nf.run(c, n, r, B, steps, stop_dead=False)
                    series.append((t, t + steps, 100 * float(nch.mean()) / N, 100 * float(nch[-steps // 5:].mean()) / N))
                    isl = nf.islands(c, n, r, B, steps=4000)
                    cover = sum(b - a + 1 for a, b in isl) / N
                    cps.append(dict(t=cp, domains=len(isl), cover=round(cover, 5), act_last_fifth=round(series[-1][3], 4)))
                    t = cp
                # retreat velocity from the last two checkpoints: d(cover)/dt * N / (2 * domains)
                v = (cps[-1]['cover'] - cps[-2]['cover']) * N / (cps[-1]['t'] - cps[-2]['t']) / (2 * max(1, cps[-1]['domains']))
                rec = dict(B=B, r=r, rc=rc, seed=seed, N=N, T=T, act_prep=round(act_prep, 3), checkpoints=cps, v_retreat=v, secs=round(time.time() - t0, 1))
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"B={B} r={r} (r_c{r-rc:+d}) seed={seed} N={N} T={T} ({rec['secs']:.0f}s): act_prep {act_prep:.1f}%; "
                      + " ".join(f"[{x['t']}: {x['domains']}d {100*x['cover']:.1f}% {x['act_last_fifth']:.1f}%]" for x in cps)
                      + f"; v_front {v:+.2e}", flush=True)


if __name__ == '__main__':
    main()
