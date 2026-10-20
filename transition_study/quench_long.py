#!/usr/bin/env python3
"""The developed active phase below threshold over long times (chunked,
checkpointed).  As quench.py: the active phase is prepared at r_prep =
2^(B-1) for t_prep steps on k 2^B + r_prep cells and the first
N = k 2^B + r cells are run at residue r.  Every `chunk` steps the active
domains are marked (cells changed within the last 4000 steps of the
chunk, gap 3) and their number and the fraction of the ring they cover
are recorded, together with the activity over the chunk.

usage: quench_long.py B k t_prep T chunk seed r1 [r2 ...]
output: quench_long_B{B}.jsonl (one record per r and seed, written when
        the run completes; checkpoints in ckpt/, per-run files in records/)
"""
import sys, time, json, os
import numpy as np
import nuca_fast as nf


def run(B, r, k, tprep, T, chunk, seed):
    base = 1 << B; rprep = base // 2
    os.makedirs('ckpt', exist_ok=True); os.makedirs('records', exist_ok=True)
    ck = f"ckpt/quench_B{B}_r{r}_k{k}_s{seed}.npz"
    N = k * base + r
    if os.path.exists(ck) and int(np.load(ck)['t']) <= T:
        z = np.load(ck); c = z['c'].copy(); n = z['n'].copy(); t = int(z['t']); hist = [tuple(float(v) for v in row) for row in z['hist']]
    else:
        c0, n0 = nf.ic_random(k * base + rprep, B, seed)
        nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
        c, n = c0[:N].copy(), n0[:N].copy(); t = 0; hist = []
    while t < T:
        steps = min(chunk, T - t)
        nch, n1, lo, hi, td = nf.run(c, n, r, B, steps - 4000, stop_dead=False)
        act = 100 * float(nch.mean()) / N
        isl = nf.islands(c, n, r, B, steps=4000)   # advances the state by 4000 steps
        t += steps
        covered = 100 * sum(b - a + 1 for a, b in isl) / N
        hist.append((t, len(isl), covered, act, 100 * float(n1[-1]) / N))
        np.savez(ck + '.tmp.npz', c=c, n=n, t=t, hist=np.array(hist, dtype=float).reshape(-1, 5)); os.replace(ck + '.tmp.npz', ck)
    return dict(B=B, r=r, k=k, N=N, t_prep=tprep, T=T, chunk=chunk, seed=seed, hist=hist)


def main():
    B, k, tprep, T, chunk, seed = (int(float(x)) for x in sys.argv[1:7])
    rs = [int(x) for x in sys.argv[7:]]
    rc = 3 * (1 << (B - 3))
    for r in rs:
        rec_path = f"records/quench_B{B}_r{r}_k{k}_T{T}_s{seed}.json"
        if os.path.exists(rec_path):
            print(f"B={B} r={r}: done already ({rec_path})", flush=True); continue
        t0 = time.time()
        rec = run(B, r, k, tprep, T, chunk, seed); rec['secs'] = round(time.time() - t0)
        with open(rec_path, 'w') as f:
            json.dump(rec, f); f.flush(); os.fsync(f.fileno())
        with open(f"quench_long_B{B}.jsonl", "a") as out:
            out.write(json.dumps(rec) + "\n"); out.flush(); os.fsync(out.fileno())
        h = rec['hist']
        print(f"B={B} r={r} (r_c{r-rc:+d}) e1={(r-(1<<(B-2)))/r:.4f} N={rec['N']} T={T:.0e}: coverage " +
              ", ".join(f"{x[2]:.1f}% ({int(x[1])} dom.) at {x[0]:.0e}" for x in h[::max(1, len(h)//6)]) + f"; final {h[-1][2]:.2f}% ({int(h[-1][1])} domains)  ({rec['secs']}s)", flush=True)


if __name__ == '__main__':
    main()
