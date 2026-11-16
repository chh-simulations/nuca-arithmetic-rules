#!/usr/bin/env python3
# fig4_activity.py - Figure 4 of P1: the lsb-reading tail activity at the
# eight widths of Table 1, linear and quadratic increments (two-seed
# means over the last 100 of 800 steps; the values of Table 1, asserted
# in verify_p1.py, section S4).
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'DejaVu Sans'

def tail_activity(dim, steps, seed, g):
    rng = np.random.default_rng(seed)
    n = (rng.permutation(dim) % 256).astype(np.int64)
    c = rng.integers(0, 2, dim).astype(np.int64)
    acts = []
    for t in range(steps):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c2 = (n >> h) & 1                                  # the lsb reading
        n = ((n + (2 * c2 - 1) * g(h) - 1) % dim) & 0xFF
        if t >= steps - 100: acts.append(np.mean(c2 != c))
        c = c2
    return 100 * float(np.mean(acts))

widths = [512, 544, 640, 768, 800, 896, 1024, 1056]
lin = [np.mean([tail_activity(N, 800, s, lambda h: h) for s in (1, 2)]) for N in widths]
quad = [np.mean([tail_activity(N, 800, s, lambda h: h * (1 - h)) for s in (1, 2)]) for N in widths]
for N, a, b in zip(widths, lin, quad):
    print('N = %4d: linear %.1f  quadratic %.1f' % (N, a, b))
fig, ax = plt.subplots(figsize=(6.4, 2.8))
ax.plot(widths, lin, 'o-', color='C0', ms=5, label='lsb activity, $g = h$')
ax.plot(widths, quad, 's-', color='C1', ms=5, label='lsb activity, $g = h(1{-}h)$')
ax.set_ylim(0, 100)
ax.set_xlabel('lattice width $N$'); ax.set_ylabel('tail activity (% of $N$)')
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('fig4_activity.pdf')
print('saved fig4_activity.pdf')
