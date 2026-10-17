#!/usr/bin/env python3
"""Fast runner for the B-bit NuCA (msb reading, linear increment).

Update, for a ring of N cells with B-bit words n_i and cells c_i:
    h_i   = c_{i-1} + c_i + c_{i+1}                (old configuration)
    c_i'  = bit (B-1-h_i) of n_i
    n_i'  = ((n_i + (2 c_i' - 1) h_i - 1) mod N) & (2^B - 1)
With r = N mod 2^B and r >= 4 the word update is the same as
    x = n_i + (2 c_i' - 1) h_i - 1
    n_i' = x & (2^B - 1)  if x >= 0  else  x + r
so the local dynamics depends on N only through the residue r, and the
ring size can be any N = k 2^B + r.  This module uses the r-form; the
gate (gate_fast.py) checks it against the canonical form digit for
digit at k = 1, 2, 3.

evolve(c, n, r, B, steps) advances the state in place and returns four
int32 arrays of length `steps` with, for every step, the number of
cells that changed, the number of 1-cells, and the leftmost and
rightmost index among the cells that changed (-1 if none).  When the
state is dead (no 1-cell and every word below 2^(B-1), hence doomed to
the all-zero sea) the run stops and the remaining entries are 0, 0,
-1, -1; the return value `tdead` is the step at which that was first
seen (0 if never).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def evolve(c, n, r, B, steps, nch, n1, xlo, xhi, stop_dead):
    N = c.shape[0]
    mask = (1 << B) - 1
    top = 1 << (B - 1)
    cn = np.empty(N, dtype=c.dtype)
    tdead = 0
    for t in range(steps):
        cnt = 0
        ones = 0
        lo = -1
        hi = -1
        nmax = 0
        for i in range(N):
            left = c[N - 1] if i == 0 else c[i - 1]
            right = c[0] if i == N - 1 else c[i + 1]
            h = left + c[i] + right
            ci = (n[i] >> (B - 1 - h)) & 1
            x = n[i] + (2 * ci - 1) * h - 1
            if x >= 0:
                n[i] = x & mask
            else:
                n[i] = x + r
            if n[i] > nmax:
                nmax = n[i]
            cn[i] = ci
            if ci != c[i]:
                cnt += 1
                if lo < 0:
                    lo = i
                hi = i
            ones += ci
        for i in range(N):
            c[i] = cn[i]
        nch[t] = cnt
        n1[t] = ones
        xlo[t] = lo
        xhi[t] = hi
        if ones == 0 and nmax < top and tdead == 0:
            tdead = t + 1
            if stop_dead:
                for s in range(t + 1, steps):
                    nch[s] = 0
                    n1[s] = 0
                    xlo[s] = -1
                    xhi[s] = -1
                break
    return tdead


@njit(cache=True)
def evolve_mark(c, n, r, B, steps, mark):
    """Advance by `steps` (no series, no dead stop) and set mark[i] = 1 for
    every cell that changed at least once."""
    N = c.shape[0]
    mask = (1 << B) - 1
    cn = np.empty(N, dtype=c.dtype)
    for t in range(steps):
        for i in range(N):
            left = c[N - 1] if i == 0 else c[i - 1]
            right = c[0] if i == N - 1 else c[i + 1]
            h = left + c[i] + right
            ci = (n[i] >> (B - 1 - h)) & 1
            x = n[i] + (2 * ci - 1) * h - 1
            if x >= 0:
                n[i] = x & mask
            else:
                n[i] = x + r
            cn[i] = ci
            if ci != c[i]:
                mark[i] = 1
        for i in range(N):
            c[i] = cn[i]


def islands(c, n, r, B, steps=4000, gap=3):
    """Advance by `steps` (in place) and return the list of (start, end)
    index pairs of the clusters of cells that changed during them, clusters
    being separated by more than `gap` unchanged cells (ring not unwrapped)."""
    mark = np.zeros(c.shape[0], dtype=np.int64)
    evolve_mark(c, n, np.int64(r), np.int64(B), np.int64(steps), mark)
    idx = np.flatnonzero(mark)
    if idx.size == 0:
        return []
    out = []; start = idx[0]; last = idx[0]
    for i in idx[1:]:
        if i - last > gap:
            out.append((int(start), int(last))); start = i
        last = i
    out.append((int(start), int(last)))
    return out


def run(c, n, r, B, steps, stop_dead=True):
    """Advance (c, n) in place by `steps`; return (nch, n1, xlo, xhi, tdead).
    With stop_dead the run stops at the first dead step (the words of a
    dead ring only drift, nothing else happens); without it the words are
    advanced to the end and tdead still records the first dead step."""
    nch = np.zeros(steps, dtype=np.int32)
    n1 = np.zeros(steps, dtype=np.int32)
    xlo = np.zeros(steps, dtype=np.int32)
    xhi = np.zeros(steps, dtype=np.int32)
    tdead = evolve(c, n, np.int64(r), np.int64(B), np.int64(steps), nch, n1, xlo, xhi, stop_dead)
    return nch, n1, xlo, xhi, int(tdead)


def ic_random(N, B, seed, p1=0.5):
    """Random initial condition: cells 1 with probability p1, words i.i.d.
    uniform in [0, 2^B).  (The P1 suite drew words as a permutation of
    0..N-1 reduced mod 2^B; the gate uses that construction, this
    function the i.i.d. one.)"""
    rng = np.random.default_rng(seed)
    c = (rng.random(N) < p1).astype(np.int64)
    n = rng.integers(0, 1 << B, N).astype(np.int64)
    return c, n


def ic_sea(N, r, seed):
    """The prepared sea: all cells 0, words i.i.d. uniform in [0, r)."""
    rng = np.random.default_rng(seed)
    c = np.zeros(N, dtype=np.int64)
    n = rng.integers(0, r, N).astype(np.int64)
    return c, n


def ic_interface(N, B, r, seed, active=(0.25, 0.75)):
    """Prepared sea outside [active[0] N, active[1] N), random inside."""
    rng = np.random.default_rng(seed)
    c = np.zeros(N, dtype=np.int64)
    n = rng.integers(0, r, N).astype(np.int64)
    a, b = int(active[0] * N), int(active[1] * N)
    c[a:b] = rng.integers(0, 2, b - a)
    n[a:b] = rng.integers(0, 1 << B, b - a)
    return c, n, a, b
