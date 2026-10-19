#!/usr/bin/env python3
"""Front of the DEVELOPED active phase against the prepared sea.

The active phase is prepared at r_prep = 2^(B-1) (e_1 = 1/2) for t_prep
steps on a ring of k 2^B + r_prep cells; the first N = k 2^B + r cells
are kept, the outer halves [0, N/4) and [3N/4, N) are replaced by the
prepared sea (cells 0, words uniform in [0, r)), and the ring is run at
residue r.  The fronts are the outermost changed cells; positions every
`every` steps; each front's velocity is the slope over the second half
of the valid range (front.velocity).  Also recorded: the net displacement
of each front between t = T/10 and T, divided by 0.9 T.

usage: front_dev.py B k t_prep T every nseeds procs r1 r2 ...
output: front_dev_B{B}.jsonl (appended)
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
import nuca_fast as nf
import front as FR


def one(args):
    B, r, k, tprep, seed, T, every = args
    base = 1 << B; rprep = base // 2
    c0, n0 = nf.ic_random(k * base + rprep, B, seed)
    nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
    N = k * base + r; a, b = N // 4, 3 * N // 4
    c = c0[:N].copy(); n = n0[:N].copy()
    rng = np.random.default_rng(10_000 + seed)
    c[:a] = 0; n[:a] = rng.integers(0, r, a)
    c[b:] = 0; n[b:] = rng.integers(0, r, N - b)
    nch, n1, lo, hi, td = nf.run(c, n, r, B, T, stop_dead=True)
    idx = np.arange(every - 1, T, every)
    t, lo_, hi_ = idx + 1, lo[idx], hi[idx]
    vr, vl = FR.velocity(t, lo_, hi_, N, a, b)
    j0 = len(t) // 10
    ok = (lo_ >= 0)
    if ok[j0] and ok[-1]:
        dr_ = ((hi_[-1] - b) - (hi_[j0] - b)) / (t[-1] - t[j0])
        dl_ = ((a - lo_[-1]) - (a - lo_[j0])) / (t[-1] - t[j0])
    else:
        dr_ = dl_ = float('nan')
    return dict(seed=seed, v_fit=[vr, vl], v_net=[dr_, dl_], tdead=td,
                final=[int(a - lo_[-1]) if lo_[-1] >= 0 else None, int(hi_[-1] - b) if hi_[-1] >= 0 else None])


def main():
    B, k, tprep, T, every, nseeds, procs = (int(x) for x in sys.argv[1:8])
    rs = [int(x) for x in sys.argv[8:]]
    rc = 3 * (1 << (B - 3))
    with open(f"front_dev_B{B}.jsonl", "a") as out:
        for r in rs:
            t0 = time.time()
            with Pool(procs) as p:
                res = p.map(one, [(B, r, k, tprep, s, T, every) for s in range(1, nseeds + 1)], chunksize=1)
            vf = np.array([x['v_fit'] for x in res], dtype=float).ravel()
            vn = np.array([x['v_net'] for x in res], dtype=float).ravel()
            rec = dict(B=B, r=r, rc=rc, k=k, T=T, nseeds=nseeds, runs=res,
                       v_fit_mean=float(np.nanmean(vf)), v_fit_sem=float(np.nanstd(vf) / np.sqrt(np.sum(~np.isnan(vf)))),
                       v_net_mean=float(np.nanmean(vn)), v_net_sem=float(np.nanstd(vn) / np.sqrt(np.sum(~np.isnan(vn)))))
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"B={B} r={r} (r_c{r-rc:+d}) e1={(r-(1<<(B-2)))/r:.4f}: v_fit {rec['v_fit_mean']:.3e} +- {rec['v_fit_sem']:.1e}, "
                  f"v_net {rec['v_net_mean']:.3e} +- {rec['v_net_sem']:.1e}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
