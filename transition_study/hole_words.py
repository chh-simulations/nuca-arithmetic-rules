#!/usr/bin/env python3
"""The holes of the developed phase quenched below threshold: from the
final states of the long quench runs (checkpoints ckpt/quench_B{B}_r{r}_
k{k}_s{seed}.npz, written by quench_long.py at t = T), the ring is
advanced by 4000 steps, the cells that changed are marked, and the holes
are the maximal runs of unmarked cells longer than `gap` = 3 (the
complement of the active domains of quench_long.py).  For every hole:
its length, the number of its cells with the word in the sea range
[0, r), in the lower pinned band (within 32 of 2^(B-1)), in the upper
band (within 32 of 3 2^(B-2)), and elsewhere, and the number of cells on.
The record also classifies the hole cells of the whole ring, and gives,
from the stored coverage history, the number of chunk-to-chunk increases
and decreases of the hole count (lower bounds on openings and closings).

usage: hole_words.py B k seed r [r ...]
output: hole_words.jsonl (appended)
"""
import sys, json
import numpy as np
import nuca_fast as nf


def holes_of(mark, gap=3):
    N = mark.size; idx = np.flatnonzero(mark == 0)
    if idx.size == 0: return []
    runs = []; start = idx[0]; last = idx[0]
    for i in idx[1:]:
        if i > last + 1:
            runs.append([int(start), int(last)]); start = i
        last = i
    runs.append([int(start), int(last)])
    # ring: merge a run ending at N-1 with one starting at 0
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == N - 1:
        runs[0] = [runs[-1][0], runs[0][1] + N]; runs.pop()
    return [(a, b) for a, b in runs if b - a + 1 > gap]


def final_state_path(B, r, k, seed):
    """the checkpoint written by quench_long.py at t = T, or its copy in final_states/ (the repository)"""
    import os
    for d in ('ckpt', 'final_states', 'results/final_states'):
        p = f"{d}/quench_B{B}_r{r}_k{k}_s{seed}.npz"
        if os.path.exists(p): return p
    raise FileNotFoundError(f"no final state for B={B} r={r} k={k} seed={seed}")


def analyse(B, r, k, seed):
    base = 1 << B; band_lo = base // 2; band_hi = 3 * base // 4
    z = np.load(final_state_path(B, r, k, seed)); c = z['c'].copy(); n = z['n'].copy(); t = int(z['t']); hist = z['hist']
    N = c.size
    mark = np.zeros(N, dtype=np.int64)
    nf.evolve_mark(c, n, np.int64(r), np.int64(B), np.int64(4000), mark)
    H = holes_of(mark)
    rows = []
    for a, b in H:
        cells = np.arange(a, b + 1) % N
        w = n[cells]; on = c[cells]
        rows.append(dict(start=int(a), length=int(b - a + 1), sea_range=int((w < r).sum()), lower_band=int((np.abs(w - band_lo) <= 32).sum()),
                         upper_band=int((np.abs(w - band_hi) <= 32).sum()), other=int(((w >= r) & (np.abs(w - band_lo) > 32) & (np.abs(w - band_hi) > 32)).sum()), on=int(on.sum()),
                         words_sample=[int(x) for x in w[:12]]))
    tot = dict(holes=len(H), cells=sum(x['length'] for x in rows), sea_range=sum(x['sea_range'] for x in rows), lower_band=sum(x['lower_band'] for x in rows),
               upper_band=sum(x['upper_band'] for x in rows), other=sum(x['other'] for x in rows), on=sum(x['on'] for x in rows))
    cnt = hist[:, 1]; d = np.diff(cnt)
    return dict(B=B, r=r, k=k, seed=seed, N=N, t=t, holes=rows, totals=tot, lengths=sorted(x['length'] for x in rows),
                count_increases=int(d[d > 0].sum()), count_decreases=int(-d[d < 0].sum()), chunks=int(len(cnt)), chunk=int(hist[1, 0] - hist[0, 0]) if len(hist) > 1 else None)


def main():
    B, k, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    for r in (int(x) for x in sys.argv[4:]):
        rec = analyse(B, r, k, seed); rows = rec['holes']; tot = rec['totals']; t = rec['t']; N = rec['N']
        with open('hole_words.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print(f"B={B} r={r} t={t:.0e} N={N}: {tot['holes']} holes, lengths {rec['lengths']}; hole cells {tot['cells']}: in [0,r) {tot['sea_range']}, lower band {tot['lower_band']}, upper band {tot['upper_band']}, other {tot['other']}, on {tot['on']}; "
              f"hole-count increases {rec['count_increases']} / decreases {rec['count_decreases']} over {rec['chunks']} chunks of {rec['chunk']}", flush=True)
        for x in rows:
            print(f"    hole at {x['start']}: length {x['length']}, sea-range {x['sea_range']}, lower {x['lower_band']}, upper {x['upper_band']}, other {x['other']}, on {x['on']}, words {x['words_sample']}")


if __name__ == '__main__':
    main()
