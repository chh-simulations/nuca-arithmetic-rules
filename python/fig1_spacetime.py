#!/usr/bin/env python3
# fig1_spacetime.py - Figure 1 of P1: spacetime diagrams of the
# configuration layer at N = 800, linear increment, seed 1, in the two
# readings, first 220 steps. The dynamics is that of Section 2 (the same
# vectorized form that verify_p1.py gates against the canonical loop).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

def spacetime(dim, steps, seed, msb):
    rng = np.random.default_rng(seed)
    n = (rng.permutation(dim) % 256).astype(np.int64)
    c = rng.integers(0, 2, dim).astype(np.int64)
    rows = []
    for _ in range(steps):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c = (n >> ((7 - h) if msb else h)) & 1
        n = ((n + (2 * c - 1) * h - 1) % dim) & 0xFF
        rows.append(c.copy())
    return np.array(rows)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1))
for ax, msb, title in ((axes[0], True, 'msb reading: frozen phase (residue 32)'),
                       (axes[1], False, 'lsb reading: persistently active phase')):
    ax.imshow(spacetime(800, 220, 1, msb), cmap='gray_r', aspect='auto', interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('cell'); ax.set_ylabel('time step')
fig.tight_layout()
fig.savefig('fig1_spacetime.pdf')
print('saved fig1_spacetime.pdf')
