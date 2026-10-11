#!/usr/bin/env python3
# fig2_wordlength.py - Figure 2 of P1: the attraction threshold across
# word lengths (a) and the interface velocity at B = 10 (b).
# Regenerates every point by simulation (about three minutes); the same
# numbers are asserted in verify_p1.py, section S9.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 9

def run_B(dim, steps, seed, B, tail=100):
    mask = (1 << B) - 1
    rng = np.random.default_rng(seed)
    n = (rng.permutation(dim) % (1 << B)).astype(np.int64)
    c = rng.integers(0, 2, dim).astype(np.int64)
    prev = c.copy(); acts = []
    for t in range(steps):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c = (n >> (B - 1 - h)) & 1
        n = ((n + (2 * c - 1) * h - 1) % dim) & mask
        acts.append(float(np.mean(c != prev))); prev = c.copy()
    return 100 * float(np.mean(acts[-tail:]))

def front(B, r, seed, steps=120000, win=50, every=1000):
    base = 1 << B; N = base + r; mask = base - 1
    rng = np.random.default_rng(seed); half = N // 2
    c = np.zeros(N, dtype=np.int64); n = np.zeros(N, dtype=np.int64)
    n[:half] = rng.integers(0, r, half)
    c[half:] = rng.integers(0, 2, N - half); n[half:] = rng.integers(0, base, N - half)
    prev = c.copy(); recent = np.zeros(N, dtype=np.int64); out = []
    for t in range(1, steps + 1):
        h = c + np.roll(c, 1) + np.roll(c, -1)
        c = (n >> (B - 1 - h)) & 1
        n = ((n + (2 * c - 1) * h - 1) % N) & mask
        ch = (c != prev); prev = c.copy()
        recent = np.where(ch, 0, recent + 1)
        if t % every == 0:
            out.append((t, int((recent < win).sum())))
    A = np.array(out)
    m = (A[:, 0] >= 20000) & (A[:, 1] < 0.8 * N)
    return np.polyfit(A[m, 0], A[m, 1], 1)[0] / 2 if m.sum() >= 5 else float('nan')

# (a) threshold triple, B = 6..12
panel_a = {}
for B in range(6, 13):
    base = 1 << B; rc = 3 * base // 8; T = max(1500, 3000 * base // 256)
    for d in range(-2, 4):
        r = rc + d
        panel_a[(B, d)] = [run_B(base + r, T, s, B) for s in (1, 2)]
        print("B=%d r=%d (r_c%+d) T=%d: %s" % (B, r, d, T, panel_a[(B, d)]))

# (b) interface velocity at B = 10
rs = [376, 380, 382, 383, 384, 385, 386, 388, 391, 400]
panel_b = {}
for r in rs:
    panel_b[r] = [front(10, r, s) for s in (1, 2)]
    print("r=%d e=%.4f v=%s" % (r, (r - 256) / r, panel_b[r]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
markers = {6: 'v', 7: '^', 8: 'o', 9: 's', 10: 'D', 11: 'p', 12: 'h'}
for B in range(6, 13):
    xs = []; ys = []
    for d in range(-2, 4):
        for v in panel_a[(B, d)]:
            xs.append(d); ys.append(v)
    ax1.plot(xs, ys, markers[B], ms=5, mfc='none', label='$B=%d$' % B)
ax1.axvspan(-2.5, 0.5, color='0.92', zorder=0)
ax1.set_xlabel('$r - r_c$,  $r_c = 3\\cdot 2^{B-3}$')
ax1.set_ylabel('tail activity at $T_B$ (% of $N$)')
ax1.set_xticks(range(-2, 4))
ax1.legend(fontsize=7, ncol=2, frameon=False)
ax1.set_title('(a) threshold across word lengths', fontsize=10)

es = [(r - 256) / r for r in rs]
for k, r in enumerate(rs):
    ax2.plot([es[k]] * 2, [1000 * v for v in panel_b[r]], 'o', ms=4, color='k', mfc='none')
ax2.plot(es, [1000 * np.nanmean(panel_b[r]) for r in rs], '-', color='0.4', lw=1)
ax2.axvline(1 / 3, color='0.6', ls='--', lw=1)
ax2.set_xlabel('sea excitability $e_1 = (r - 2^{B-2})/r$,  $B = 10$')
ax2.set_ylabel('front velocity $v$ ($10^{-3}$ cells / step)')
ax2.set_title('(b) interface velocity', fontsize=10)
fig.tight_layout()
fig.savefig('fig2_wordlength.pdf')
print('saved fig2_wordlength.pdf')
