#!/usr/bin/env python3
"""Exact reachability of word values for a single cell driven by arbitrary
neighbours (the pinning floor).

A cell's update depends only on its own state (c, n) and on the states
a, b of its two neighbours: h = a + c + b, c' = bit (B-1-h) of n,
x = n + (2c'-1) h - 1, n' = x & (2^B - 1) if x >= 0 else x + r.  Treating
(a, b) as free inputs over-approximates every real neighbourhood, so the
set of (c, n) reachable from the sea states {(0, n): 0 <= n < r} under
ALL input sequences bounds what any sea cell can ever become, whatever
the active phase next to it does.  For each B and r this script computes
that set exactly (a fixed point of the successor map over the 2 * 2^B
states) and reports the largest reachable word, and whether the cell can
reach the sustained-climb band [5 2^(B-4), 2^B) (bit B-4 readable at
h = 3 inside the ignition band), the direct band [r_c, 2^B), and the
active band [2^(B-1), 2^B).

usage: floor_exact.py Bmin Bmax        (all r from 2^(B-2) to 2^(B-1) - 1)
       floor_exact.py B r [r2 ...]     (print the reachable set summary)
output: floor_exact.jsonl (appended; one record per B with the per-r table)
"""
import sys, json
import numpy as np


def reachable(B, r):
    base = 1 << B; mask = base - 1
    n = np.arange(base, dtype=np.int64)
    # successor tables for the four inputs (a, b), for c = 0 and c = 1
    succ = []
    for c in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                h = a + c + b
                cp = (n >> (B - 1 - h)) & 1
                x = n + (2 * cp - 1) * h - 1
                npn = np.where(x >= 0, x & mask, x + r)
                succ.append((c, cp, npn))
    R = np.zeros((2, base), dtype=bool)
    R[0, :r] = True
    while True:
        new = R.copy()
        for c, cp, npn in succ:
            src = R[c]
            # states (cp[i], npn[i]) reachable from (c, i) for every reachable i
            for v in (0, 1):
                sel = src & (cp == v)
                if sel.any():
                    new[v, npn[sel]] = True
        if np.array_equal(new, R):
            break
        R = new
    return R


def summary(B, r):
    R = reachable(B, r)
    base = 1 << B; rc = 3 * (1 << (B - 3)); floor = 5 * (1 << (B - 4))
    words = np.flatnonzero(R.any(axis=0))
    on_words = np.flatnonzero(R[1])
    return dict(r=r, u=round(r / (1 << (B - 4)), 4), e1=round((r - (1 << (B - 2))) / r, 5),
                max_word=int(words.max()), max_on_word=int(on_words.max()) if on_words.size else -1,
                n_reach=int(words.size), above_r=int((words >= r).sum()),
                reach_climb_band=bool(words.max() >= floor), reach_direct_band=bool(words.max() >= rc),
                reach_active_band=bool(words.max() >= (base >> 1)))


def main():
    a = sys.argv[1:]
    if len(a) == 2 and int(a[1]) <= 24:          # Bmin Bmax
        Bmin, Bmax = int(a[0]), int(a[1])
        for B in range(Bmin, Bmax + 1):
            rc = 3 * (1 << (B - 3)); floor = 5 * (1 << (B - 4))
            rows = [summary(B, r) for r in range(1 << (B - 2), 1 << (B - 1))]
            first_climb = next((x['r'] for x in rows if x['reach_climb_band']), None)
            first_direct = next((x['r'] for x in rows if x['reach_direct_band']), None)
            first_active = next((x['r'] for x in rows if x['reach_active_band']), None)
            maxw_below = max(x['max_word'] for x in rows if x['r'] <= floor)
            rec = dict(B=B, rc=rc, floor=floor, first_r_reaching_climb_band=first_climb, first_r_reaching_direct_band=first_direct,
                       first_r_reaching_active_band=first_active, max_word_for_r_le_floor=maxw_below, rows=rows)
            with open('floor_exact.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"B={B:2d}: r_c={rc}, floor 5*2^(B-4)={floor}; max reachable word for r <= floor: {maxw_below} "
                  f"(floor-1 = {floor-1}); first r reaching the climb band: {first_climb} (= floor+1? {first_climb == floor + 1}); "
                  f"first r reaching r_c: {first_direct}; first r reaching 2^(B-1): {first_active}", flush=True)
    else:
        B = int(a[0])
        for r in a[1:]:
            print(json.dumps(summary(B, int(r))))


if __name__ == '__main__':
    main()
