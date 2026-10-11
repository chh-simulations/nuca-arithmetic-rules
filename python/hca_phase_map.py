"""
hca_phase_map.py - the phase-map probe.

The heterogeneous hyper-rule dynamics, faithful to the original MATLAB
implementation of Raptis (hca.m / ruleupdate.m / cfilter.m), with two
things promoted to parameters:
the increment (any map of h) and the lattice width.

  kernel   : cfilter([1 1 0..0 1])   (symmetric circulant; h = K@c in 0..3)
  lookup   : c(j) = rules(j, h(j)+1)
  update   : int = mod(num + sgn*incr(h) - 1, dim),  sgn = 2c - 1,
             stored back as 8 bits (values >= 256 truncated to the low
             byte, matching a truncating de2bi).
  ordering : 'lsb' or 'msb' — the two bit-readings of the stored word.
             They are conjugate under the reflector: the value of a word
             read in one order is R_8 of its value read in the other
             (verified for all 256 words; see section C below).

Section A reproduces the increment sweep (Section 8 of the paper): order
is not the boundary - h(1-h) fails in the active ordering, h(1-h)(2-h)
minimizes in both.

Section B reproduces the width sweep (Section 4 of the paper): the
organizing variable is width mod 256; at 768 or 1024 the frozen columns
of the reference run disappear.

Reported per run: tail rank (distinct rule integers, last 100 steps, as %
of 256) and tail activity (config bits changed per step, as % of width).

Pure Python + numpy/scipy; a run at width 800 x 800 steps takes seconds.
"""
import numpy as np
from scipy.linalg import toeplitz


def kernel(dim):
    mask = np.zeros(dim); mask[0] = 1; mask[1] = 1; mask[-1] = 1
    return toeplitz(np.concatenate(([mask[0]], mask[1:][::-1])), mask)

def d2b(n, msb):
    return [(n >> (7 - i)) & 1 for i in range(8)] if msb else [(n >> i) & 1 for i in range(8)]

def b2d(bits, msb):
    return (sum(int(v) << (7 - i) for i, v in enumerate(bits)) if msb
            else sum(int(v) << i for i, v in enumerate(bits)))


def run(dim, steps, seed, ordering, incr):
    """ordering: 'msb' or 'lsb'. incr: function of h in {0..3}."""
    msb = (ordering == 'msb')
    rng = np.random.default_rng(seed)
    rules = np.array([d2b(int(v), msb) for v in rng.permutation(dim) % 256], dtype=int)
    c = rng.integers(0, 2, dim)
    K = kernel(dim)
    rank, modc, prev = [], [], c.copy()
    for _ in range(steps):
        h = (K @ c).astype(int)
        for j in range(dim):
            row = rules[j]
            hj = int(h[j]); c[j] = row[hj] if hj < 8 else 0
            num = b2d(row, msb); sgn = 2 * c[j] - 1
            val = (num + sgn * incr(hj) - 1) % dim      # the modulus is dim
            rules[j] = d2b(val & 0xFF, msb)             # 8-bit truncation
        rank.append(len(set(b2d(rules[j], msb) for j in range(dim))))
        modc.append(int(np.sum(c != prev))); prev = c.copy()
    return np.array(rank), np.array(modc)


def tail(dim, steps, ordering, incr, seeds=(1, 2)):
    rk, ac = [], []
    for s in seeds:
        r, m = run(dim, steps, s, ordering, incr)
        rk.append(r[-100:].mean() / 256 * 100)
        ac.append(m[-100:].mean() / dim * 100)
    return np.mean(rk), np.mean(ac)


INCREMENTS = {
    'h            (linear, deg 1)': lambda h: h,
    'h(1-h)       (yours,  deg 2)': lambda h: h * (1 - h),
    'h^2                  (deg 2)': lambda h: h * h,
    '3h           (big lin deg 1)': lambda h: 3 * h,
    'h(1-h)(2-h)  (alt.,   deg 3)': lambda h: h * (1 - h) * (2 - h),
    '7h           (big lin deg 1)': lambda h: 7 * h,
    'h^3                  (deg 3)': lambda h: h ** 3,
}

if __name__ == "__main__":
    print("A. Increment sweep at width 800 (the reference run's width), 600 steps.")
    print("   tail rank % of 256 / tail activity % of width\n")
    for ordering in ('msb', 'lsb'):
        print(f"   ordering: {ordering}")
        for name, fn in INCREMENTS.items():
            r, a = tail(800, 600, ordering, fn)
            print(f"     {name}:  rank {r:5.1f}%   act {a:5.1f}%")
        print()

    print("B. Width sweep, linear and h(1-h) increments, 800 steps.")
    print("   watch the column 'mod 256'\n")
    for name, fn in [('linear h', lambda h: h), ('poly h(1-h)', lambda h: h * (1 - h))]:
        for ordering in ('msb', 'lsb'):
            print(f"   increment {name}, ordering {ordering}")
            for dim in (512, 544, 640, 768, 800, 896, 1024, 1056):
                r, a = tail(dim, 800, ordering, fn)
                print(f"     width {dim:>4}  (mod 256 = {dim % 256:>3}):  rank {r:5.1f}%   act {a:5.1f}%")
            print()

    print("C. Two exact identities (Section 3 of the paper), checkable here.")
    def refl(n, L):                       # the reflector R_L on L-bit words
        return int(bin(n)[2:].zfill(L)[::-1], 2)
    # C1: the LSB<->MSB reading conversion IS the reflector, all 256 words
    c1 = all(b2d(d2b(n, False), True) == refl(n, 8) for n in range(256))
    print(f"   C1  MSB-reading of a word == R_8(LSB-reading), all 256 words: {c1}")
    # C2: the antihomomorphism over concatenation, all pairs at L=4 -> 8
    c2 = all(refl(x + y * 16, 8) == refl(y, 4) + refl(x, 4) * 16
             for x in range(16) for y in range(16))
    print(f"   C2  R_8(|x| + |y|*2^4) == R_4(|y|) + R_4(|x|)*2^4, all pairs: {c2}")
