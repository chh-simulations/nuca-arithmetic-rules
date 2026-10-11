"""
hca_probes.py — two probes on top of hca_phase_map.py (same directory).

P1  Palindrome distance. Palindromic words are the 16 fixed points of the
    reflector R_8 - the sector where the two readings coincide. The
    probe tracks the mean distance of the rule population to the nearest
    palindrome: for each word, the number of mismatched mirror bit-pairs
    (0..4; a random word averages 2.0). Zero would mean the dynamics lives
    entirely in the R-invariant sector.

P2  Convergence windows. Rank (as % of 256) in successive time windows, to
    see slow contraction directly at times beyond 1000 steps.

Usage:
    python hca_probes.py             (both probes at width 800)
"""
import numpy as np
from hca_phase_map import kernel, d2b, b2d


def run_probe(dim, steps, seed, ordering, incr):
    """hca_phase_map.run with palindrome tracking added; dynamics unchanged."""
    msb = (ordering == 'msb')
    rng = np.random.default_rng(seed)
    rules = np.array([d2b(int(v), msb) for v in rng.permutation(dim) % 256], dtype=int)
    c = rng.integers(0, 2, dim)
    K = kernel(dim)
    rank, modc, pal = [], [], []
    prev = c.copy()
    for _ in range(steps):
        h = (K @ c).astype(int)
        for j in range(dim):
            row = rules[j]
            hj = int(h[j]); c[j] = row[hj] if hj < 8 else 0
            num = b2d(row, msb); sgn = 2 * c[j] - 1
            val = (num + sgn * incr(hj) - 1) % dim
            rules[j] = d2b(val & 0xFF, msb)
        rank.append(len(set(b2d(rules[j], msb) for j in range(dim))))
        modc.append(int(np.sum(c != prev))); prev = c.copy()
        pal.append(np.sum(rules[:, :4] != rules[:, 7:3:-1], axis=1).mean())
    return np.array(rank), np.array(modc), np.array(pal)


if __name__ == "__main__":
    DIM = 800
    print("P1. Palindrome distance (mean mismatched mirror pairs, 0..4;")
    print("    random baseline 2.0), width 800, 800 steps, start -> tail(100)\n")
    for ordering in ('msb', 'lsb'):
        for name, fn in [('linear h', lambda h: h), ('poly h(1-h)', lambda h: h * (1 - h))]:
            _, _, p = run_probe(DIM, 800, 1, ordering, fn)
            print(f"    {ordering} {name:12}: {p[:5].mean():.2f} -> {p[-100:].mean():.2f}")
    print()
    print("P2. Rank convergence windows (% of 256), width 800, 5000 steps,")
    print("    active ordering; windows at t ~ 600 / 1000 / 3000 / 5000\n")
    for name, fn in [('linear h', lambda h: h),
                     ('poly h(1-h)', lambda h: h * (1 - h)),
                     ('h^3', lambda h: h ** 3)]:
        r, _, _ = run_probe(DIM, 5000, 1, 'lsb', fn)
        w = [r[a:b].mean() / 256 * 100 for a, b in
             [(500, 600), (900, 1000), (2900, 3000), (4900, 5000)]]
        print(f"    lsb {name:12}: " + " -> ".join(f"{v:5.1f}%" for v in w))
