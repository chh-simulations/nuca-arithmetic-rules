#!/usr/bin/env python3
"""The word-defined front from the final states of the developed-phase
runs (checkpoints ckpt/frontdev_B{B}_r{r}_k{k}_s{seed}.npz, or their
copies in final_states/): on each side, the outermost cell whose word
lies outside the sea range [0, r).

One convention for both sides.  The prepared block is [a, b) with
a = N/4, b = 3N/4, and the position of a front is the number of cells
beyond the block boundary:

    left   p = a - i          right   p = i - (b - 1)

so that p = 1 is the first cell of the prepared sea, p = 0 the block's
own edge cell, and p < 0 a cell inside the block (a retreat).  The
block's edge cells may hold words below r in the prepared state, so the
initial position p(0) <= 0 is measured on the prepared state, and the
displacement of a front is p(T) - p(0).  Beside it the script counts
the cells of the prepared sea that hold a word outside the sea range at
T (the converted cells), and gives the changed-cell front of the last
2000 steps in the same convention.

Correction of 13 November 2026.  The first form of this script counted
the right side as i - b (one cell short of the left side's a - i) and,
for a front with no converted sea cell, returned minus the initial edge
offset instead of the final position, so that a front that had
retreated into the block was entered as if it had not moved.  The
records written by that form are superseded by the present ones.

usage: word_front.py B r k [B r k ...]
output: word_front.jsonl (appended)
"""
import sys, json, glob
import numpy as np
import nuca_fast as nf

_PREP = {}


def initial_state(B, r, k, seed, tprep):
    """the developed block of front_dev_long.py: a random ring run at r = 2^(B-1) for tprep steps, cut to N cells"""
    base = 1 << B; rprep = base // 2
    key = (B, k, seed, tprep)
    if key not in _PREP:
        c0, n0 = nf.ic_random(k * base + rprep, B, seed)
        nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
        _PREP[key] = (c0, n0)
    c0, n0 = _PREP[key]
    N = k * base + r
    return c0[:N].copy(), n0[:N].copy()


def positions(n, r, a, b):
    """positions (left, right) of the outermost cells with a word outside [0, r); None if there is no such word"""
    out = np.flatnonzero(n >= r)
    if out.size == 0:
        return None, None
    return int(a - out.min()), int(out.max() - (b - 1))


def final_states(B, r, k):
    """the checkpoints of front_dev_long.py at t = T (ckpt/), or their copies in final_states/ (the repository)"""
    for d in ('ckpt', 'final_states', 'results/final_states'):
        fs = sorted(glob.glob(f'{d}/frontdev_B{B}_r{r}_k{k}_s*.npz'))
        if fs: return fs
    return []


def measure(B, r, k):
    N = k * (1 << B) + r; a, b = N // 4, 3 * N // 4
    tprep = 500000 if B >= 11 else 200000
    rows = []
    for f in final_states(B, r, k):
        seed = int(f.rsplit('_s', 1)[1][:-4]); z = np.load(f)
        if int(z['t']) == 0: continue
        c0, n0 = initial_state(B, r, k, seed, tprep)
        # the prepared ring: the block [a, b) of the developed state, sea words below r outside it
        block = np.zeros(N, dtype=n0.dtype); block[a:b] = n0[a:b]
        p0 = positions(block, r, a, b)
        p = positions(z['n'], r, a, b)
        if p0[0] is None or p[0] is None:
            continue                                # no word outside the sea range: the active phase is gone
        conv = [int((z['n'][:a] >= r).sum()), int((z['n'][b:] >= r).sum())]
        h = z['hist']; changed = [int(h[-1][1]), int(h[-1][2]) + 1]
        rows.append(dict(seed=seed, t=int(z['t']), position=[p[0], p[1]], initial_position=[p0[0], p0[1]],
                         word_front=[p[0] - p0[0], p[1] - p0[1]], converted_sea_cells=conv, changed_front=changed))
    rec = dict(B=B, r=r, k=k, N=N, convention='p = cells beyond the block boundary; word_front = p(T) - p(0)', runs=rows)
    disp = np.array([x['word_front'] for x in rows], dtype=float).ravel()
    conv = np.array([x['converted_sea_cells'] for x in rows], dtype=float).ravel()
    rec['mean_word_front'] = float(disp.mean()) if disp.size else None
    rec['sem_word_front'] = float(disp.std(ddof=1) / np.sqrt(disp.size)) if disp.size > 1 else None
    rec['mean_converted'] = float(conv.mean()) if conv.size else None
    rec['sem_converted'] = float(conv.std(ddof=1) / np.sqrt(conv.size)) if conv.size > 1 else None
    return rec


def main():
    a_ = [int(x) for x in sys.argv[1:]]
    with open('word_front.jsonl', 'a') as out:
        for B, r, k in zip(a_[::3], a_[1::3], a_[2::3]):
            rec = measure(B, r, k); rows = rec['runs']; N = rec['N']
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"B={B} r={r} k={k} N={N}: displacements {[x['word_front'] for x in rows]} (mean {rec['mean_word_front']:.2f} +- {rec['sem_word_front']:.2f}), "
                  f"converted sea cells {[x['converted_sea_cells'] for x in rows]} (mean {rec['mean_converted']:.2f} +- {rec['sem_converted']:.2f}), "
                  f"initial positions {[x['initial_position'] for x in rows]}, changed-cell fronts {[x['changed_front'] for x in rows]}, t={[x['t'] for x in rows]}", flush=True)


if __name__ == '__main__':
    main()
