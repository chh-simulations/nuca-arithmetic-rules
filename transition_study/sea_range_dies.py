"""Random sea-range configurations die at or below the pinning floor.

The observation of Sec. III that is not part of the proof of the pinning theorem: a ring whose cells are random
(0 or 1 with probability 1/2) and whose words are uniform in the sea range [0, r) is dead, every cell 0, within a
few hundred steps, at residues from 2^(B-2) + 2 to the floor 5 * 2^(B-4).  Plain numpy, the rule as written
(reduction modulo N, then the low B bits), independent of the compiled runner.

    python sea_range_dies.py          # B = 8 and 10, five residues each, eight configurations per residue

Rings of N = 4 * 2^B + r cells.  A configuration is dead at step t if no cell is 1 at steps t and t + 1; the
all-zero ring with words below r <= 2^(B-1) is invariant (the companion paper's frozen sea), so death is final.
"""
import numpy as np


def step(c, n, N, B):
    h = np.roll(c, 1) + c + np.roll(c, -1)
    c2 = (n >> (B - 1 - h)) & 1
    n2 = ((n + (2 * c2 - 1) * h - 1) % N) & ((1 << B) - 1)
    return c2, n2


def death_time(B, r, seed, k=4, tmax=100000):
    N = k * (1 << B) + r
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, N)
    n = rng.integers(0, r, N)
    for t in range(tmax):
        c2, n2 = step(c, n, N, B)
        if not c.any() and not c2.any():
            return t, int(n2.max()) < r
        c, n = c2, n2
    return None, False


def main():
    worst = {}
    for B in (8, 10):
        q, floor = 1 << (B - 2), 5 * (1 << (B - 4))
        residues = sorted({q + 2, q + 3, (q + floor) // 2, floor - 1, floor})
        for r in residues:
            res = [death_time(B, r, seed) for seed in range(1, 9)]
            times = [t for t, _ in res]
            ok = all(t is not None for t in times) and all(inside for _, inside in res)
            print("B=%2d r=%4d (e_1 = %.4f): dead at step %s; all words below r afterwards: %s" % (
                B, r, (r - q) / r, times, ok))
            assert ok
            worst[B] = max(worst.get(B, 0), max(times))
    print("latest death: %d steps at B = 8, %d steps at B = 10" % (worst[8], worst[10]))


if __name__ == '__main__':
    main()
