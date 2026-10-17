#!/usr/bin/env python3
"""Creep below the direct threshold, measured in bulk: from a random
initial state at N = k 2^B + r the activity collapses to islands; the
late growth of the number of 1-cells per step, divided by the number of
islands at t_mid and by the 1-density of the active phase (~0.70),
estimates the mean growth per island; half of it is the creep velocity
per front (islands that die contribute negatively).

usage: creep.py B k T nseeds tmid r1 r2 ...
output: creep_B{B}.jsonl
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def main():
    B, k, T, nseeds, tmid = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rs = [int(x) for x in sys.argv[6:]]
    base = 1 << B; rc = 3 * base // 8
    nwin = 20
    with open(f"creep_B{B}.jsonl", "a") as out:
        for r in rs:
            N = k * base + r
            for seed in range(1, nseeds + 1):
                t0 = time.time()
                c, n = nf.ic_random(N, B, seed)
                nf.run(c, n, r, B, tmid, stop_dead=False)
                isl = nf.islands(c, n, r, B, steps=4000)
                nch, n1, lo, hi, td = nf.run(c, n, r, B, T - tmid - 4000, stop_dead=False)
                isl2 = nf.islands(c, n, r, B, steps=4000)
                # window means of n1 and nch over the run after tmid
                edges = np.linspace(0, len(n1), nwin + 1).astype(int)
                n1w = [float(n1[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
                nchw = [float(nch[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
                tw = [tmid + 4000 + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])]
                # slope of n1 over the second half
                h = nwin // 2
                slope = float(np.polyfit(tw[h:], n1w[h:], 1)[0])
                per_island = slope / max(1, len(isl)) / 0.70 / 2   # cells per step per front
                rec = dict(B=B, r=r, rc=rc, seed=seed, N=N, T=T, tmid=tmid, islands_mid=len(isl), islands_end=len(isl2),
                           widths_mid=sorted(b - a + 1 for a, b in isl), widths_end=sorted(b - a + 1 for a, b in isl2),
                           t=tw, n1=n1w, nch=nchw, slope_n1=slope, v_est=per_island, secs=round(time.time() - t0, 1))
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"B={B} r={r} (r_c{r-rc:+d}) seed={seed} N={N}: islands {len(isl)} -> {len(isl2)}; n1 {n1w[0]:.0f} -> {n1w[-1]:.0f}; "
                      f"slope {slope:.3e}/step -> v_est {per_island:.2e} cells/step/front; max width {max(rec['widths_end']) if isl2 else 0} ({rec['secs']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
