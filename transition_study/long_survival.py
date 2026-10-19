#!/usr/bin/env python3
"""Long seed-survival runs, memory-bounded.

As spread.py (a random patch of w0 cells, words uniform in [0, 2^B), in
the prepared sea of a ring of k 2^B + r cells), but run in chunks of
`chunk` steps up to T, stopping when the ring is dead.  At the end of
every chunk the extent of the active region (number of cells between the
outermost cells that changed during the last 2000 steps of the chunk,
inclusive) is recorded.  A run is classified at T as
  dead      : the ring died (tdead recorded)
  growing   : extent at T exceeds extent at T/10 by >= 40 cells, or
              covers >= 80% of the ring
  static    : alive and not growing
and the extent history is stored so the classification can be redone.

usage: long_survival.py B r nruns T [w0=32] [k=4] [chunk=1000000] [procs=2] [seed0=1] [seeds=s1,s2,...]
output: long_survival_B{B}_r{r}.jsonl (one line per run); with an explicit
seed list (used to follow the survivors of a shorter run further; the
runs are deterministic, so the earlier history is reproduced) the output
is long_survival_B{B}_r{r}_cont.jsonl.
"""
import sys, time, json
import numpy as np
from multiprocessing import Pool
import nuca_fast as nf


def one(args):
    B, r, k, w0, seed, T, chunk = args
    base = 1 << B; N = k * base + r
    rng = np.random.default_rng(seed)
    c = np.zeros(N, dtype=np.int64); n = rng.integers(0, r, N).astype(np.int64)
    x0 = N // 2; a = x0 - w0 // 2
    c[a:a + w0] = rng.integers(0, 2, w0); n[a:a + w0] = rng.integers(0, base, w0)
    t = 0; hist = []; tdead = 0
    # checkpoint after every chunk (the container may be restarted): state, time, history
    import os
    os.makedirs('ckpt', exist_ok=True)
    ck = f"ckpt/survival_B{B}_r{r}_s{seed}_T{T}.npz"
    if os.path.exists(ck):
        z = np.load(ck); c[:] = z['c']; n[:] = z['n']; t = int(z['t']); hist = [tuple(int(v) for v in row) for row in z['hist']]
    while t < T:
        steps = min(chunk, T - t)
        nch, n1, lo, hi, td = nf.run(c, n, r, B, steps, stop_dead=True)
        if td:
            tdead = t + td; break
        t += steps
        l, h = lo[-2000:], hi[-2000:]
        ok = l >= 0
        ext = int(h[ok].max() - l[ok].min() + 1) if ok.any() else 0
        hist.append((t, ext, int(n1[-1])))
        np.savez(ck + '.tmp.npz', c=c, n=n, t=t, hist=np.array(hist, dtype=np.int64).reshape(-1, 3)); os.replace(ck + '.tmp.npz', ck)
    if tdead:
        cls = 'dead'
    else:
        e_end = hist[-1][1]
        e_early = next(e for (tt, e, _) in hist if tt >= T // 10)
        cls = 'growing' if (e_end - e_early >= 40 or e_end >= 0.8 * N) else 'static'
    return dict(B=B, r=r, seed=seed, N=N, T=T, tdead=tdead, cls=cls, hist=hist)


def main():
    a = sys.argv[1:]
    B, r, nruns, T = int(a[0]), int(a[1]), int(a[2]), int(float(a[3]))
    w0 = int(a[4]) if len(a) > 4 else 32
    k = int(a[5]) if len(a) > 5 else 4
    chunk = int(float(a[6])) if len(a) > 6 else 1_000_000
    procs = int(a[7]) if len(a) > 7 else 2
    seed0 = int(a[8]) if len(a) > 8 else 1
    seeds = [int(x) for x in a[9].split('=')[1].split(',')] if len(a) > 9 else [seed0 + i for i in range(nruns)]
    nruns = len(seeds)
    t0 = time.time()
    import os
    fn = f"long_survival_B{B}_r{r}.jsonl" if len(a) <= 9 else f"long_survival_B{B}_r{r}_cont.jsonl"
    # every finished run is written to its own file (opened and closed at once, so that it
    # survives a container restart); the jsonl is rebuilt as the union of those files and
    # of its own lines, at the start and at the end
    import glob
    os.makedirs('records', exist_ok=True)
    rec_path = lambda seed: f"records/{fn[:-6]}_s{seed}_T{T}.json"
    def union():
        recs = {}
        if os.path.exists(fn):
            for l in open(fn):
                x = json.loads(l)
                if x['T'] == T: recs[x['seed']] = x
        for g in glob.glob(f"records/{fn[:-6]}_s*_T{T}.json"):
            x = json.load(open(g)); recs[x['seed']] = x
        return recs
    def write_all(recs):
        with open(fn + '.tmp', 'w') as f:
            for s_ in sorted(recs): f.write(json.dumps(recs[s_]) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(fn + '.tmp', fn)
    recs = union(); write_all(recs)
    jobs = [(B, r, k, w0, s, T, chunk) for s in seeds if s not in recs]
    res = [recs[s] for s in seeds if s in recs]
    with Pool(procs) as p:
        for rec in p.imap_unordered(one, jobs, chunksize=1):
            with open(rec_path(rec['seed']), 'w') as f:
                json.dump(rec, f); f.flush(); os.fsync(f.fileno())
            with open(fn, 'a') as out:
                out.write(json.dumps(rec) + "\n"); out.flush(); os.fsync(out.fileno())
            res.append(rec)
    write_all(union())
    cnt = {c: sum(1 for x in res if x['cls'] == c) for c in ('dead', 'static', 'growing')}
    rc = 3 * (1 << (B - 3))
    print(f"B={B} r={r} (r_c{r-rc:+d}) runs={nruns} T={T:.0e}: {cnt}; P_grow={cnt['growing']/nruns:.4f}; "
          f"latest death {max((x['tdead'] for x in res), default=0)}; static extents {sorted(x['hist'][-1][1] for x in res if x['cls']=='static')}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
