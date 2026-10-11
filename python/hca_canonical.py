"""
Canonical simulator for the heterogeneous hyper-rule CA (Raptis).
Faithful to the original MATLAB implementation of Raptis (hca.m /
ruleupdate.m / cfilter.m):

  kernel   : cfilter([1 1 0..0 1])  (symmetric circulant; h = K@c in 0..3)
  lookup   : c(j) = rules(j, h(j)+1)
  update   : int = mod(num + sgn*h - 1,        dim)   [linear variant]
             int = mod(num + sgn*h*(1 - h) - 1, dim)   [quadratic variant]
             sgn = 2c - 1;  stored back as 8 bits (values >= 256 truncated mod 256,
             matching a truncating de2bi; the strict comm-toolbox de2bi would
             reject 9-10 bit values, which hca.m produces whenever dim > 256).
  NOTE     : mod is taken with d = dim (NOT 256). At dim = 512 the two coincide
             (512 = 2*256); at dim = 800 they do not: e.g. -3 -> mod 800 -> 797
             -> 8-bit truncation -> 29, whereas mod 256 gives 253. This is why
             lattice size changes the dynamics.

encoding : 'lsb' or 'msb' bit order for the stored 8-bit rule words. The two
           encodings are genuinely different dynamics (the lookup reads the
           other nibble; the integer arithmetic differs). fliplr of a de2bi
           row converts one encoding to the other — so reflect(update(...))
           composed around the update runs the opposite-encoding dynamics.

mode     : 'plain'     — the original map M
           'sandwich'  — R[M(R[rule])] per step (fliplr in and out)
           'one_sided' — reflect(update(...)) only (stored word flipped)

Reported per step: rank (distinct rule integers) and modcount (config bits
changed vs previous step) - the two curves of the reference run (Raptis,
notes, Fig. 2).

Runtime: dim=800, 1000 steps ~ 1-2 min per run (pure Python). numpy, scipy.
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


def run(dim=800, steps=1000, seed=1, encoding='msb', variant='linear', mode='plain'):
    msb = (encoding == 'msb')
    poly = (variant == 'poly')
    rng = np.random.default_rng(seed)
    rules = np.array([d2b(int(v), msb) for v in rng.permutation(dim) % 256], dtype=int)
    c = rng.integers(0, 2, dim)
    K = kernel(dim)
    rank, modc, prev = [], [], c.copy()
    for _ in range(steps):
        h = (K @ c).astype(int)
        for j in range(dim):
            row = rules[j][::-1] if mode == 'sandwich' else rules[j]
            hj = int(h[j]); c[j] = row[hj] if hj < 8 else 0
            num = b2d(row, msb); sgn = 2 * c[j] - 1
            incr = sgn * hj * (1 - hj) if poly else sgn * hj
            val = (num + incr - 1) % dim              # the modulus is dim, not 256
            nb = d2b(val & 0xFF, msb)                 # 8-bit truncation
            if mode in ('sandwich', 'one_sided'):
                nb = nb[::-1]
            rules[j] = nb
        rank.append(len(set(b2d(rules[j], msb) for j in range(dim))))
        modc.append(int(np.sum(c != prev))); prev = c.copy()
    return np.array(rank), np.array(modc)


if __name__ == "__main__":
    DIM, STEPS = 800, 1000
    print(f"dim={DIM}, {STEPS} steps, seed 1. rank / modcount: start -> tail mean\n")
    for enc in ('lsb', 'msb'):
        for var in ('linear', 'poly'):
            for mode in ('plain', 'sandwich', 'one_sided'):
                r, m = run(DIM, STEPS, 1, enc, var, mode)
                print(f"  {enc} {var:6} {mode:9}: rank {r[:5].mean():4.0f}->{r[-100:].mean():4.0f}"
                      f"   modcount {m[:5].mean():4.0f}->{m[-100:].mean():4.0f}")
        print()
    print("Reading: in one encoding both curves fall to a plateau near 65 for both")
    print("variants (reference Fig. 2); in the other, modcount stays high for both;")
    print("one_sided shows no minimization (the 'None' rows of the four-way table).")
