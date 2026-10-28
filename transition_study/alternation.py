#!/usr/bin/env python3
"""Period-two alternation in the bulk of the developed active phase (Sec. IV F):
the probability that a cell has changed state at each of the last m steps, for
m = 1..8, against the value p^m that a random sequence with the same rate of
change p would give.  The phase is prepared at r = 2^(B-1) on a ring of
4 2^B + r cells for 200 2^B steps from a random state (seed 1), then observed
for 20000 steps.

usage: alternation.py [B]        (default 10)
output: alternation.jsonl (appended)
"""
import sys, json
import numpy as np
import nuca_fast as nf

B = int(sys.argv[1]) if len(sys.argv) > 1 else 10
base = 1 << B; r = base // 2; N = 4 * base + r; T = 20000
c, n = nf.ic_random(N, B, 1)
nf.run(c, n, r, B, 200 * base, stop_dead=False)
S = np.zeros((T, N), dtype=np.int8)
for t in range(T):
    nf.run(c, n, r, B, 1, stop_dead=False); S[t] = c
alt = (S[1:] != S[:-1]).astype(np.int8)
p = float(alt.mean())
rows = []
run = np.zeros(N, dtype=np.int32); counts = {m: 0 for m in range(1, 9)}
for t in range(alt.shape[0]):
    run = np.where(alt[t] == 1, run + 1, 0)
    for m in counts: counts[m] += int(np.sum(run >= m))
tot = alt.shape[0] * N
rec = dict(B=B, r=r, N=N, t_prep=200 * base, T=T, seed=1, on=float(S.mean()), change_rate=p,
           alternation=[dict(m=m, prob=counts[m] / tot, random=p ** m) for m in range(1, 9)])
with open('alternation.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
print(f"B={B} r={r} N={N}: on {rec['on']:.3f}, change rate {p:.3f}; P(alternated for the last m steps): " +
      ", ".join(f"m={x['m']}: {x['prob']:.4g} (random {x['random']:.4g})" for x in rec['alternation']))
