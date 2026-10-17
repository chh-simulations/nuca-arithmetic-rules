#!/usr/bin/env python3
"""Reduced model of the front: cell 0 is a driven edge, on with
probability p at every step independently; cells 1..L-1 are sea cells
(initially 0, clock phases uniform in [0, r)) updated by the exact rule
of nuca_fast; the right neighbour of cell L-1 is a permanent 0.  A
success is a cell whose word reaches 2^(B-1) - 32 (the pinned band); the
sea is then redrawn.  run_edge returns (successes, steps with a word in
[r_c, band), maximal word).

usage: edge_model.py            (the B = 10 table, r = 376..400, p = 0.5, 0.7)
       edge_model.py scaling    (B = 9..12 at fixed dr, T scaled as 2^B)
"""
import sys
import numpy as np
from numba import njit

@njit(cache=True)
def run_edge(r, B, L, T, p, seed):
    np.random.seed(seed)
    mask = (1 << B) - 1; rc = 3 * (1 << (B - 3)); band = (1 << (B - 1)) - 32
    c = np.zeros(L, dtype=np.int64); n = np.zeros(L, dtype=np.int64)
    for i in range(1, L): n[i] = np.random.randint(0, r)
    cn = np.zeros(L, dtype=np.int64)
    succ = 0; reached_rc = 0; maxw = 0
    for t in range(T):
        c[0] = 1 if np.random.random() < p else 0
        n[0] = 0
        for i in range(1, L):
            left = c[i - 1]; right = c[i + 1] if i < L - 1 else 0
            h = left + c[i] + right
            ci = (n[i] >> (B - 1 - h)) & 1
            x = n[i] + (2 * ci - 1) * h - 1
            if x >= 0: n[i] = x & mask
            else: n[i] = x + r
            cn[i] = ci
        for i in range(1, L):
            c[i] = cn[i]
            if n[i] > maxw: maxw = n[i]
            if n[i] >= rc and n[i] < band: reached_rc += 1
            if n[i] >= band:
                succ += 1
                # reset
                for j in range(1, L):
                    c[j] = 0; n[j] = np.random.randint(0, r)
                break
    return succ, reached_rc, maxw

def table():
  B = 10; L = 48; T = 4_000_000
  for p in (0.5, 0.7):
      print(f"p = {p}: successes per 4e6 steps (and steps with a word in [r_c, band)), seeds 1-3")
      for r in (376, 378, 380, 381, 382, 383, 384, 385, 386, 388, 392, 400):
          res = [run_edge(r, B, L, T, p, s) for s in (1, 2, 3)]
          print(f"  r={r}: succ {[x[0] for x in res]}  rc-words {[x[1] for x in res]}  maxw {[x[2] for x in res]}")


def scaling():
    for B in (9, 10, 11, 12):
        base = 1 << B; rc = 3 * base // 8; T = 4_000_000 * base // 1024
        rows = []
        for dr in (-2, -1, 0, 1, 2, 4, 8, 16, 32, 64):
            r = rc + dr * max(1, base // 1024) if abs(dr) > 2 else rc + dr
            res = [run_edge(r, B, 48, T, 0.5, s)[0] for s in (1, 2)]
            rows.append((r, dr, res))
        print(f"B={B} r_c={rc} T={T}: " + "  ".join(f"r_c{dr:+d}{'x' if abs(dr) > 2 and base > 1024 else ''}:{sum(res)/2:.0f}" for r, dr, res in rows), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "scaling":
        scaling()
    else:
        table()
