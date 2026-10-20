#!/usr/bin/env python3
"""Word distribution and activity of the developed active phase: a ring
of N = 4 2^B + r cells from a random state, run for 200 2^B steps at r
(above threshold), then measured over 2000 steps: cells changing per
step, cells on, and the fraction of words within 32 of 2^(B-1) and of
3 2^(B-2).

usage: phase_words.py B r [B r ...]
output: phase_words.jsonl (appended)
"""
import sys, json
import numpy as np
import nuca_fast as nf


def measure(B, r, seed=1):
    base = 1 << B; N = 4 * base + r; T = 200 * base
    c, n = nf.ic_random(N, B, seed)
    nf.run(c, n, r, B, T, stop_dead=False)
    nch, n1, _, _, _ = nf.run(c, n, r, B, 2000, stop_dead=False)
    lo = np.abs(n - base // 2) <= 32; hi = np.abs(n - 3 * base // 4) <= 32
    return dict(B=B, r=r, N=N, T=T, seed=seed, changing_pct=round(100 * float(nch.mean()) / N, 2), on_pct=round(100 * float(n1.mean()) / N, 2),
                near_half_pct=round(100 * float(lo.mean()), 1), near_three_quarters_pct=round(100 * float(hi.mean()), 1),
                below_r_pct=round(100 * float((n < r).mean()), 1))


def main():
    a = [int(x) for x in sys.argv[1:]]
    with open('phase_words.jsonl', 'a') as out:
        for B, r in zip(a[::2], a[1::2]):
            rec = measure(B, r)
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"B={B} r={r}: changing {rec['changing_pct']}%, on {rec['on_pct']}%, words within 32 of 2^(B-1): {rec['near_half_pct']}%, of 3 2^(B-2): {rec['near_three_quarters_pct']}%, below r: {rec['below_r_pct']}%", flush=True)


if __name__ == '__main__':
    main()
