#!/usr/bin/env python3
"""The word-defined front from the final states of the developed-phase
runs (checkpoints ckpt/frontdev_B{B}_r{r}_k{k}_s{seed}.npz): the
outermost cell on each side whose word lies outside the sea range [0, r),
counted outward from the boundaries a = N/4 and b = 3N/4 of the prepared
block.  Since the block's own edge cells may have words below r, the
initial word front is measured on the prepared state as well, and the
displacement is the difference.  Compares with the changed-cell front.

usage: word_front.py B r k [B r k ...]
output: word_front.jsonl (appended)
"""
import sys, json, glob
import numpy as np
import nuca_fast as nf


def initial_state(B, r, k, seed, tprep):
    base = 1 << B; rprep = base // 2
    c0, n0 = nf.ic_random(k * base + rprep, B, seed)
    nf.run(c0, n0, rprep, B, tprep, stop_dead=False)
    N = k * base + r
    return c0[:N].copy(), n0[:N].copy()


def word_front(n, r, a, b):
    out = np.flatnonzero(n >= r)
    left = out[out < a]; right = out[out >= b]
    L = int(a - left.min()) if left.size else None
    R = int(right.max() - b) if right.size else None
    return L, R


def final_states(B, r, k):
    """the checkpoints of front_dev_long.py at t = T (ckpt/), or their copies in final_states/ (the repository)"""
    import os
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
        # the prepared block's own edge: words outside [0, r) inside [a, b)
        L0, R0 = word_front(n0, r, a, b)            # None: no word >= r outside the block initially
        inside = np.flatnonzero(n0[a:b] >= r)
        e0L = int(inside.min()) if inside.size else 0; e0R = int(b - a - 1 - inside.max()) if inside.size else 0
        L, R = word_front(z['n'], r, a, b)
        h = z['hist']; cL, cR = int(h[-1][1]), int(h[-1][2])
        rows.append(dict(seed=seed, t=int(z['t']), word_front=[L if L is not None else -e0L, R if R is not None else -e0R],
                         initial_block_edge_offset=[e0L, e0R], changed_front=[cL, cR]))
    rec = dict(B=B, r=r, k=k, N=N, runs=rows)
    disp = np.array([x['word_front'] for x in rows], dtype=float).ravel()
    rec['mean_word_front'] = float(disp.mean()) if disp.size else None
    rec['sem_word_front'] = float(disp.std(ddof=1) / np.sqrt(disp.size)) if disp.size > 1 else None
    return rec


def main():
    a_ = [int(x) for x in sys.argv[1:]]
    with open('word_front.jsonl', 'a') as out:
        for B, r, k in zip(a_[::3], a_[1::3], a_[2::3]):
            rec = measure(B, r, k); rows = rec['runs']; N = rec['N']
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"B={B} r={r} k={k} N={N}: word fronts {[x['word_front'] for x in rows]} (mean {rec['mean_word_front']:.1f} +- {rec['sem_word_front']:.1f}), changed-cell fronts {[x['changed_front'] for x in rows]}, t={[x['t'] for x in rows]}", flush=True)


if __name__ == '__main__':
    main()
