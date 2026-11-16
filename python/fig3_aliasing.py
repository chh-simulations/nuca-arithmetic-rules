#!/usr/bin/env python3
# fig3_aliasing.py - Figure 3 of P1: rank against time at N = 800 in the
# lsb reading, seed 1: the fast linear cell and the slow quadratic cell,
# with the frozen-phase attractor value of the same seed (71 distinct
# words; verify_p1.py, sections S1 and S11) marked.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

def rank_trace(dim, steps, seed, msb, g):
    rng = np.random.default_rng(seed)
    n = (rng.permutation(dim) % 256).astype(np.int64)
    c = rng.integers(0, 2, dim).astype(np.int64)
    rk = np.empty(steps)
    for t in range(steps):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c = (n >> ((7 - h) if msb else h)) & 1
        n = ((n + (2 * c - 1) * g(h) - 1) % dim) & 0xFF
        rk[t] = len(np.unique(n))
    return rk

slow = rank_trace(800, 5000, 1, False, lambda h: h * (1 - h))
fast = rank_trace(800, 5000, 1, False, lambda h: h)
t = np.arange(1, 5001)
print('first step with rank <= 71: fast %d, slow %d' % (t[fast <= 71][0], t[slow <= 71][0]))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(t[::10], slow[::10], color='b', lw=1, label='lsb, $h(1{-}h)$ (slow)')
ax.plot(t[::10], fast[::10], color='g', lw=1, label='lsb, $h$ (fast)')
ax.axhline(71, color='k', ls='--', lw=0.8, label='frozen-phase attractor (71 words)')
ax.set_xlim(0, 5000)
ax.set_xlabel('time step'); ax.set_ylabel('rank (distinct rule words)')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('fig3_aliasing.pdf')
print('saved fig3_aliasing.pdf')
