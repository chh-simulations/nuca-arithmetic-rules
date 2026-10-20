#!/usr/bin/env python3
"""Developed-phase fronts over long times, memory-bounded (chunked).

As front_dev.py: the active phase is prepared at r_prep = 2^(B-1) for
t_prep steps on a ring of k 2^B + r_prep cells, the first N = k 2^B + r
cells are kept, the outer halves [0, N/4) and [3N/4, N) are replaced by
the prepared sea, and the ring is run at residue r.  The run proceeds in
chunks of `chunk` steps; at the end of every chunk the positions of the
two fronts are recorded as the outermost cells that changed during the
last 2000 steps of the chunk (left front: a - lo, right front: hi - b,
both counted outward into the sea).  Each front's velocity is the slope
of its position over the second half of the run; the net velocity is the
displacement between t = T/10 and T divided by 0.9 T.

usage: front_dev_long.py B r k t_prep T chunk nseeds procs
output: front_dev_long_B{B}.jsonl (appended; one record per residue with
        the per-seed front histories)
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
import nuca_fast as nf

NONE = -(1 << 40)   # sentinel for 'no changed cell' in the checkpoint arrays (front positions can be negative)


def one(args):
    B, r, k, tprep, seed, T, chunk = args[:7]
    use_ckpt = args[7] if len(args) > 7 else True     # the verification suite regenerates from scratch
    base = 1 << B; rprep = base // 2
    c0, n0 = nf.ic_random(k * base + rprep, B, seed)
    nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
    N = k * base + r; a, b = N // 4, 3 * N // 4
    c = c0[:N].copy(); n = n0[:N].copy()
    rng = np.random.default_rng(10_000 + seed)
    c[:a] = 0; n[:a] = rng.integers(0, r, a)
    c[b:] = 0; n[b:] = rng.integers(0, r, N - b)
    t = 0; hist = []; tdead = 0
    # checkpoint after every chunk (the container may be restarted): state, time, history
    import os
    os.makedirs('ckpt', exist_ok=True)
    ck = f"ckpt/frontdev_B{B}_r{r}_k{k}_s{seed}.npz"  # T-independent, so a run can be extended to a larger T
    if use_ckpt and os.path.exists(ck):
        z = np.load(ck)
        if int(z['t']) <= T:
            c[:] = z['c']; n[:] = z['n']; t = int(z['t'])
            hist = [(int(x[0]), None if x[1] == NONE else int(x[1]), None if x[2] == NONE else int(x[2]), int(x[3])) for x in z['hist']]
    while t < T:
        steps = min(chunk, T - t)
        nch, n1, lo, hi, td = nf.run(c, n, r, B, steps, stop_dead=True)
        if td:
            tdead = t + td; break
        t += steps
        l, h = lo[-2000:], hi[-2000:]
        ok = l >= 0
        if ok.any():
            hist.append((t, int(a - l[ok].min()), int(h[ok].max() - b), int(nch[-2000:].mean())))
        else:
            hist.append((t, None, None, 0))
        if use_ckpt:
            hk = np.array([(x[0], NONE if x[1] is None else x[1], NONE if x[2] is None else x[2], x[3]) for x in hist], dtype=np.int64).reshape(-1, 4)
            np.savez(ck + '.tmp.npz', c=c, n=n, t=t, hist=hk); os.replace(ck + '.tmp.npz', ck)
    H = np.array([(x[0], x[1], x[2]) for x in hist if x[1] is not None], dtype=float)
    out = dict(seed=seed, tdead=tdead, hist=hist)
    if len(H) >= 4:
        tt = H[:, 0]; half = tt >= T / 2
        out['v_fit'] = [float(np.polyfit(tt[half], H[half, 2], 1)[0]), float(np.polyfit(tt[half], H[half, 1], 1)[0])]
        j0 = int(np.searchsorted(tt, T / 10))
        out['v_net'] = [float((H[-1, 2] - H[j0, 2]) / (tt[-1] - tt[j0])), float((H[-1, 1] - H[j0, 1]) / (tt[-1] - tt[j0]))]
        out['final'] = [int(H[-1, 1]), int(H[-1, 2])]
    return out


def main():
    B, r, k, tprep, T, chunk, nseeds, procs = (int(float(x)) for x in sys.argv[1:9])
    rc = 3 * (1 << (B - 3)); e1 = (r - (1 << (B - 2))) / r
    t0 = time.time()
    import os, glob
    os.makedirs('records', exist_ok=True)
    rec_path = lambda seed: f"records/frontdev_B{B}_r{r}_k{k}_T{T}_s{seed}.json"
    have = {int(g.rsplit('_s', 1)[1][:-5]): json.load(open(g)) for g in glob.glob(f"records/frontdev_B{B}_r{r}_k{k}_T{T}_s*.json")}
    jobs = [(B, r, k, tprep, s, T, chunk) for s in range(1, nseeds + 1) if s not in have]
    with Pool(procs) as p:
        for x in p.imap_unordered(one, jobs, chunksize=1):
            with open(rec_path(x['seed']), 'w') as f:
                json.dump(x, f); f.flush(); os.fsync(f.fileno())
            have[x['seed']] = x
    res = [have[s] for s in range(1, nseeds + 1)]
    vf = np.array([x.get('v_fit', [np.nan, np.nan]) for x in res], dtype=float).ravel()
    vn = np.array([x.get('v_net', [np.nan, np.nan]) for x in res], dtype=float).ravel()
    rec = dict(B=B, r=r, rc=rc, e1=round(e1, 5), k=k, N=k * (1 << B) + r, t_prep=tprep, T=T, chunk=chunk, nseeds=nseeds, runs=res,
               v_fit_mean=float(np.nanmean(vf)), v_fit_sem=float(np.nanstd(vf) / np.sqrt(np.sum(~np.isnan(vf)))),
               v_net_mean=float(np.nanmean(vn)), v_net_sem=float(np.nanstd(vn) / np.sqrt(np.sum(~np.isnan(vn)))),
               secs=round(time.time() - t0))
    with open(f"front_dev_long_B{B}.jsonl", "a") as out:
        out.write(json.dumps(rec) + "\n"); out.flush(); os.fsync(out.fileno())
    print(f"B={B} r={r} (r_c{r-rc:+d}) e1={e1:.4f} N={rec['N']} T={T:.0e} fronts={2*nseeds}: "
          f"v_fit {rec['v_fit_mean']:.3e} +- {rec['v_fit_sem']:.1e}, v_net {rec['v_net_mean']:.3e} +- {rec['v_net_sem']:.1e}, "
          f"scaled v*2^(B-3) = {rec['v_net_mean']*(1<<(B-3)):.2e}; final displacements {[x.get('final') for x in res]}  ({rec['secs']}s)", flush=True)


if __name__ == '__main__':
    main()
