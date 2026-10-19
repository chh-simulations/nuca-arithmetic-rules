#!/usr/bin/env python3
"""Exact reachability for TWO adjacent sea cells (1 and 2), updated by the
exact rule between themselves, with free inputs a (left neighbour of
cell 1) and d (right neighbour of cell 2).  This over-approximates every
real neighbourhood more tightly than the single-cell model of
floor_exact.py (cell 2 is a real sea cell, so it can be on only when its
word is in the ignition band and it must turn off after a step with
h >= 2 unless its word reads a set bit).  Reports, for each r, the
largest word either cell can reach from the sea states.

usage: floor_exact2.py B r1 [r2 ...]
"""
import sys, json
import numpy as np


def step_tables(B, r):
    base = 1 << B; mask = base - 1
    n = np.arange(base, dtype=np.int64)
    T = {}
    for c in (0, 1):
        for h in (c, c + 1, c + 2):
            cp = (n >> (B - 1 - h)) & 1
            x = n + (2 * cp - 1) * h - 1
            T[(c, h)] = (cp, np.where(x >= 0, x & mask, x + r))
    return T


def reachable2(B, r):
    base = 1 << B
    T = step_tables(B, r)
    S = 2 * base                      # states of one cell: index c * base + n
    R = np.zeros((S, S), dtype=bool)  # R[s1, s2]
    R[:r, :r] = True                  # both off, words in [0, r)
    idx = np.arange(S); c_of = idx // base; n_of = idx % base
    while True:
        new = R.copy()
        s1, s2 = np.nonzero(R)
        c1, n1, c2, n2 = c_of[s1], n_of[s1], c_of[s2], n_of[s2]
        for a in (0, 1):
            for d in (0, 1):
                h1 = a + c1 + c2; h2 = c1 + c2 + d
                cp1 = np.empty_like(c1); np1 = np.empty_like(n1); cp2 = np.empty_like(c2); np2 = np.empty_like(n2)
                for c in (0, 1):
                    for h in (c, c + 1, c + 2):
                        m = (c1 == c) & (h1 == h)
                        if m.any():
                            cp, nn = T[(c, h)]; cp1[m] = cp[n1[m]]; np1[m] = nn[n1[m]]
                        m = (c2 == c) & (h2 == h)
                        if m.any():
                            cp, nn = T[(c, h)]; cp2[m] = cp[n2[m]]; np2[m] = nn[n2[m]]
                new[cp1 * base + np1, cp2 * base + np2] = True
        if np.array_equal(new, R):
            break
        R = new
    s1, s2 = np.nonzero(R)
    w1 = n_of[s1]; w2 = n_of[s2]
    return int(max(w1.max(), w2.max())), int(w1.max()), int(w2.max()), int(R.sum())


if __name__ == '__main__':
    B = int(sys.argv[1]); base = 1 << B; rc = 3 * (1 << (B - 3)); floor = 5 * (1 << (B - 4))
    for r in (int(x) for x in sys.argv[2:]):
        mx, m1, m2, cnt = reachable2(B, r)
        print(json.dumps(dict(B=B, r=r, u=round(r / (1 << (B - 4)), 4), max_word=mx, max_word_cell1=m1, max_word_cell2=m2,
                              states=cnt, reach_climb_band=mx >= floor, reach_direct_band=mx >= rc, reach_active_band=mx >= base // 2)), flush=True)
