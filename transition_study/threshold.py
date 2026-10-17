#!/usr/bin/env python3
"""Bulk runs around the threshold at word length B: activity in log-spaced
windows, island count and widths at an intermediate time and at the end.

Prediction recorded before running (16 Oct, after the B=10 runs):
  the creep channel (an ignited sea cell with word r_c - 2 climbing by
  +2 at a single h = 3 step into the range where bit B-3 is set) exists
  iff r >= r_c - 1 = 3*2^(B-3) - 1; the direct channel (ignited words
  with bit B-3 set) iff r >= r_c + 1.  So: no growth at r_c - 2, slow
  growth at r_c - 1 and r_c, fast invasion at r_c + 1, at every B.

usage: threshold.py B k Tmult seeds r1 r2 ...   (N = k 2^B + r, T = Tmult * 2^B)
output: appended lines in threshold_B{B}.jsonl
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def main():
    B, k, Tmult, nseeds = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rs = [int(x) for x in sys.argv[5:]]
    base = 1 << B; T = Tmult * base; rc = 3 * base // 8
    edges = np.unique(np.round(np.logspace(2, np.log10(T), 50)).astype(int))
    tmid = int(50 * base)   # island count time: 50 clock periods of 2^B steps
    with open(f"threshold_B{B}.jsonl", "a") as out:
        for r in rs:
            N = k * base + r
            for seed in range(1, nseeds + 1):
                t0 = time.time()
                c, n = nf.ic_random(N, B, seed)
                nch1, _, _, _, _ = nf.run(c, n, r, B, tmid, stop_dead=False)
                isl_mid = nf.islands(c, n, r, B, steps=4000)
                nch2, _, _, _, _ = nf.run(c, n, r, B, T - tmid - 4000, stop_dead=False)
                isl_end = nf.islands(c, n, r, B, steps=4000)
                nch = np.concatenate([nch1, np.zeros(4000, dtype=np.int32), nch2])  # the 4000 marking steps are not recorded
                rho = []
                for a, b in zip(edges[:-1], edges[1:]):
                    seg = nch[a:b]; seg = seg[seg > 0] if (a <= tmid + 4000 and b > tmid) else seg
                    rho.append(100 * float(seg.mean()) / N if seg.size else float('nan'))
                wm = [b - a + 1 for a, b in isl_mid]; we = [b - a + 1 for a, b in isl_end]
                rec = dict(B=B, r=r, rc=rc, seed=seed, N=N, T=T, secs=round(time.time() - t0, 1),
                           edges=edges.tolist(), rho=[round(x, 4) for x in rho], tmid=tmid,
                           islands_mid=len(isl_mid), widths_mid=sorted(wm), islands_end=len(isl_end), widths_end=sorted(we),
                           act_mid=round(100 * float(nch1[-10000:].mean()) / N, 4), act_end=round(100 * float(nch2[-10000:].mean()) / N, 4))
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"B={B} r={r} (r_c{r-rc:+d}) seed={seed} N={N} T={T} ({rec['secs']:.0f}s): act {rec['act_mid']:.3f}% at {tmid} -> {rec['act_end']:.3f}% at end; "
                      f"islands {len(isl_mid)} (max w {max(wm) if wm else 0}) -> {len(isl_end)} (max w {max(we) if we else 0})", flush=True)


if __name__ == '__main__':
    main()
