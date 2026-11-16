"""
verify_p1.py - regenerates every number of the paper
"Phase structure and contraction timescales of a non-uniform cellular
automaton with arithmetic rule dynamics" (P. Stoychev), version 3.
Self-contained (numpy only). Every check prints measured against
expected and PASS, WARN or FAIL; a copy of the output is written to
verify_p1_results.txt.

Run:   python verify_p1.py            (all sections, about half an hour)
       python verify_p1.py --quick    (one seed where two are used)
       python verify_p1.py S8 S9      (chosen sections only)

Sections and the parts of the paper they cover:
  S1  canonical simulator against the reference run of Raptis
      (private communication, Fig. 2): width 800, 1000 steps          (Section 2)
  S2  exact identities: reflector, antihomomorphism, palindrome fixed
      points, the partial conjugation of Theorem 1    (Section 3)
  S3  increment sweep, width 800, 600 steps            (Section 8)
  S4  width-class sweep, 800 steps, Table 1            (Section 4)
  S5  long-run convergence, width 800, 5000 steps      (Section 6)
  S6  width 1000 (residue 232), 1000 steps             (Section 8)
  S7  palindrome-distance order parameter              (Section 7)
  S8  mechanism of the frozen phase: confinement, consistency table,
      out-of-sample residues, core structure, threshold (Section 5)
  S9  the law across word lengths: regimes at B = 9, 10; threshold
      r_c = 3*2^(B-3) at B = 6..12; growth at r_c+1; the exact
      excitability crossing; interface velocity at B = 10, counted
      in the prepared half of the ring, and where the remnants of
      the random half sit at the end of those runs (Section 5)
  S10 eight-seed ensembles for every cell of Table 1    (Section 4)
  S11 statements added or corrected in version 3: the full
      conjugation of Theorem 2; Propositions 1 and 2 and Corollary 2
      at every residue and on every cycle, with simulations at the
      exceptional residues; widths below the word modulus; Table 1
      at 2 x 10^4 steps; the fine scan over eight seeds and the slow
      invasion at and below r_c to 2 x 10^5 steps; the excitability
      along single cycles; the frozen-phase rank over eight seeds;
      the palindrome distance in time and the palindromic start;
      the lsb activity at every residue; Rule 110 on 8 cells, the
      toy widths and the resweep of Section 9   (Sections 3-7, 9)

Tolerances: identities exact; counts +-1.5; percentages +-0.6 to 0.8
points and palindrome distances +-0.06 in S1-S7 (numpy-version RNG
nuances; PCG64 streams are normally identical); +-0.15 points or tighter
in S8-S11, where vectorized runners are used.
"""
import sys, datetime
import numpy as np

OUT = open("verify_p1_results.txt", "w")
def emit(s=""):
    print(s); OUT.write(s + "\n"); OUT.flush()

TALLY = {"PASS": 0, "WARN": 0, "FAIL": 0}
def check(label, got, exp, tol):
    d = abs(got - exp)
    tag = "PASS" if d <= tol else ("WARN" if d <= 2 * tol else "FAIL")
    TALLY[tag] += 1
    emit(f"  [{tag}] {label}: got {got:.2f}, expected {exp:.2f} (tol {tol})")
def checkbool(label, ok):
    tag = "PASS" if ok else "FAIL"; TALLY[tag] += 1
    emit(f"  [{tag}] {label}: {ok}")

# ---------------- canonical dynamics (identical to hca_canonical.py) --------
def kernel(dim):
    # numpy-only build of the periodic 3-neighbour circulant; verified
    # byte-identical to the canonical scipy.linalg.toeplitz construction.
    K = np.zeros((dim, dim))
    idx = np.arange(dim)
    K[idx, idx] = 1
    K[idx, (idx + 1) % dim] = 1
    K[idx, (idx - 1) % dim] = 1
    return K

def d2b(n, msb):
    return [(n >> (7 - i)) & 1 for i in range(8)] if msb else [(n >> i) & 1 for i in range(8)]

def b2d(bits, msb):
    return (sum(int(v) << (7 - i) for i, v in enumerate(bits)) if msb
            else sum(int(v) << i for i, v in enumerate(bits)))

def run(dim=800, steps=1000, seed=1, encoding='msb', variant='linear', mode='plain',
        incr_fn=None, track_pal=False):
    """incr_fn overrides the variant increment when given (function of h)."""
    msb = (encoding == 'msb'); poly = (variant == 'poly')
    rng = np.random.default_rng(seed)
    rules = np.array([d2b(int(v), msb) for v in rng.permutation(dim) % 256], dtype=int)
    c = rng.integers(0, 2, dim)
    K = kernel(dim)
    rank, modc, pal, prev = [], [], [], c.copy()
    for _ in range(steps):
        h = (K @ c).astype(int)
        for j in range(dim):
            row = rules[j][::-1] if mode == 'sandwich' else rules[j]
            hj = int(h[j]); c[j] = row[hj] if hj < 8 else 0
            num = b2d(row, msb); sgn = 2 * c[j] - 1
            if incr_fn is not None:
                incr = sgn * incr_fn(hj)
            else:
                incr = sgn * hj * (1 - hj) if poly else sgn * hj
            val = (num + incr - 1) % dim
            nb = d2b(val & 0xFF, msb)
            if mode in ('sandwich', 'one_sided'):
                nb = nb[::-1]
            rules[j] = nb
        rank.append(len(set(b2d(rules[j], msb) for j in range(dim))))
        modc.append(int(np.sum(c != prev))); prev = c.copy()
        if track_pal:
            pal.append(np.sum(rules[:, :4] != rules[:, 7:3:-1], axis=1).mean())
    return np.array(rank), np.array(modc), np.array(pal)

def tailpct(dim, steps, seeds, encoding, incr_fn):
    rk, ac = [], []
    for s in seeds:
        r, m, _ = run(dim, steps, s, encoding, incr_fn=incr_fn)
        rk.append(r[-100:].mean() / 256 * 100); ac.append(m[-100:].mean() / dim * 100)
    return float(np.mean(rk)), float(np.mean(ac))

INCR = {
    'h':        lambda h: h,
    'h(1-h)':   lambda h: h * (1 - h),
    'h^2':      lambda h: h * h,
    '3h':       lambda h: 3 * h,
    'alt3':     lambda h: h * (1 - h) * (2 - h),
    '7h':       lambda h: 7 * h,
    'h^3':      lambda h: h ** 3,
}

# ---------------------------------------------------------------- sections --
def S1():
    emit("\nS1. Canonical simulator vs the reference run of Raptis (private communication, Fig. 2): width 800, 1000 steps, seed 1")
    r, m, _ = run(800, 1000, 1, 'msb', 'linear', 'plain')
    check("msb-lin start rank", r[:5].mean(), 242, 1.5)
    check("msb-lin tail rank", r[-100:].mean(), 71, 1.5)
    check("msb-lin start modc", m[:5].mean(), 276, 1.5)
    check("msb-lin tail modc", m[-100:].mean(), 54, 1.5)
    r, m, _ = run(800, 1000, 1, 'lsb', 'poly', 'plain')
    check("lsb-poly tail rank", r[-100:].mean(), 91, 1.5)
    check("lsb-poly tail modc", m[-100:].mean(), 529, 2.0)
    r, m, _ = run(800, 1000, 1, 'msb', 'linear', 'one_sided')
    check("one-sided tail rank (no minimization)", r[-100:].mean(), 228, 2.0)
    check("one-sided tail modc", m[-100:].mean(), 414, 2.0)

def S2():
    emit("\nS2. Exact identities")
    refl = lambda n, L: int(bin(n)[2:].zfill(L)[::-1], 2)
    checkbool("C1 MSB-reading == R_8(LSB-reading), all 256 words",
              all(b2d(d2b(n, False), True) == refl(n, 8) for n in range(256)))
    checkbool("C2 antihomomorphism, all pairs L=4->8",
              all(refl(x + y * 16, 8) == refl(y, 4) + refl(x, 4) * 16
                  for x in range(16) for y in range(16)))
    checkbool("palindromes = fixed points of R_8, exactly 16",
              sum(1 for n in range(256) if refl(n, 8) == n) == 16)
    # R M R (partial sandwich, lookup untouched) == plain @ opposite encoding
    def run_partial(dim, steps, init_ints, seed_c, encoding, variant, reflect):
        msb = (encoding == 'msb'); poly = (variant == 'poly')
        rng = np.random.default_rng(seed_c)
        rules = np.array([d2b(int(v), msb) for v in init_ints], dtype=int)
        c = rng.integers(0, 2, dim); K = kernel(dim); traj = []
        for _ in range(steps):
            h = (K @ c).astype(int)
            for j in range(dim):
                stored = rules[j]
                hj = int(h[j]); c[j] = stored[hj] if hj < 8 else 0
                w = stored[::-1] if reflect else stored
                num = b2d(w, msb); sgn = 2 * c[j] - 1
                incr = sgn * hj * (1 - hj) if poly else sgn * hj
                nb = d2b(((num + incr - 1) % dim) & 0xFF, msb)
                rules[j] = nb[::-1] if reflect else nb
            traj.append(rules.copy())
        return traj
    rng = np.random.default_rng(1); v = rng.permutation(800) % 256
    bitrev = lambda n: int(bin(n)[2:].zfill(8)[::-1], 2)
    for variant in ('linear', 'poly'):
        tA = run_partial(800, 120, v, 7, 'lsb', variant, True)
        tB = run_partial(800, 120, [bitrev(int(x)) for x in v], 7, 'msb', variant, False)
        checkbool(f"R-M-R == plain@opposite, identical trajectories, {variant}",
                  all(np.array_equal(a, b) for a, b in zip(tA, tB)))

def S3(seeds):
    emit(f"\nS3. Increment sweep, width 800, 600 steps, seeds {seeds} (rank% / act%)")
    exp = {  # (msb rank, msb act, lsb rank, lsb act)
      'h':      (29.5,  7.7, 13.0, 48.5), 'h(1-h)': (33.5, 14.5, 50.8, 67.0),
      'h^2':    (29.0,  2.8, 19.2, 55.6), '3h':     (20.7,  6.8, 17.9, 51.3),
      'alt3':   (12.5,  0.0, 14.3, 46.0), '7h':     (19.5,  2.2, 30.9, 52.6),
      'h^3':    (24.9,  0.2, 76.5, 46.2)}
    for name, fn in INCR.items():
        for enc, (ei, ea) in zip(('msb', 'lsb'), [exp[name][:2], exp[name][2:]]):
            r, a = tailpct(800, 600, seeds, enc, fn)
            check(f"{enc} {name} rank%", r, ei, 0.6); check(f"{enc} {name} act%", a, ea, 0.6)

def S4(seeds):
    emit(f"\nS4. Width-class sweep, 800 steps, seeds {seeds} (linear & h(1-h))")
    exp = {  # dim: {(variant,enc): (rank%, act%)}
      512:  {('h','msb'): (12.8, 47.3), ('h','lsb'): (75.6, 48.4), ('h(1-h)','msb'): (34.2, 67.6), ('h(1-h)','lsb'): (67.1, 66.5)},
      544:  {('h','msb'): (23.7,  6.3), ('h','lsb'): (12.8, 48.7), ('h(1-h)','msb'): (27.4, 13.3), ('h(1-h)','lsb'): (37.6, 67.6)},
      640:  {('h','msb'): (10.6, 46.1), ('h','lsb'): (45.5, 48.4), ('h(1-h)','msb'): (55.6, 17.4), ('h(1-h)','lsb'): (49.8, 66.5)},
      768:  {('h','msb'): (13.5, 47.3), ('h','lsb'): (83.8, 48.5), ('h(1-h)','msb'): (46.9, 65.7), ('h(1-h)','lsb'): (74.0, 67.5)},
      800:  {('h','msb'): (29.6,  7.6), ('h','lsb'): (13.0, 48.2), ('h(1-h)','msb'): (32.8, 14.5), ('h(1-h)','lsb'): (43.1, 67.6)},
      896:  {('h','msb'): (11.1, 46.5), ('h','lsb'): (46.9, 48.4), ('h(1-h)','msb'): (56.6, 20.1), ('h(1-h)','lsb'): (54.4, 66.7)},
      1024: {('h','msb'): (13.8, 47.3), ('h','lsb'): (87.6, 48.2), ('h(1-h)','msb'): (54.6, 66.0), ('h(1-h)','lsb'): (80.4, 66.3)},
      1056: {('h','msb'): (31.9,  7.7), ('h','lsb'): (13.1, 48.2), ('h(1-h)','msb'): (34.5, 15.8), ('h(1-h)','lsb'): (47.0, 67.3)}}
    for dim, cells in exp.items():
        for (name, enc), (ei, ea) in cells.items():
            r, a = tailpct(dim, 800, seeds, enc, INCR[name])
            check(f"dim {dim} (mod {dim%256}) {enc} {name} rank%", r, ei, 0.8)
            check(f"dim {dim} (mod {dim%256}) {enc} {name} act%", a, ea, 0.8)

def S5():
    emit("\nS5. Long-run convergence, width 800, 5000 steps, seed 1 (rank% windows)")
    windows = [(500, 600), (900, 1000), (1900, 2000), (2900, 3000), (4900, 5000)]
    exp = {('lsb', 'h(1-h)'): (50.2, 35.5, 23.9, 20.2, 14.4),
           ('lsb', 'h^3'):    (76.2, 77.0, 76.6, 76.0, 76.0),
           ('lsb', 'h'):      (13.0, 13.0, 13.0, 12.9, 13.0),
           ('msb', 'h^3'):    (25.4, 24.9, 24.9, 24.9, 24.9)}
    for (enc, name), evs in exp.items():
        r, m, _ = run(800, 5000, 1, enc, incr_fn=INCR[name])
        for (a, b), ev in zip(windows, evs):
            check(f"{enc} {name} rank% @t~{b}", r[a:b].mean() / 256 * 100, ev, 0.6)
    r, m, _ = run(800, 5000, 2, 'lsb', incr_fn=INCR['h(1-h)'])
    check("lsb h(1-h) seed2 rank% @600", r[500:600].mean() / 256 * 100, 51.5, 0.8)
    check("lsb h(1-h) seed2 rank% @5000 (still fallen)", r[4900:5000].mean() / 256 * 100, 25.0, 1.0)

def S6():
    emit("\nS6. Width-1000 cells (mod 256 = 232), 1000 steps, seed 1")
    exp = {('msb', 'h'): (9.6, 45.8), ('msb', 'h(1-h)'): (45.4, 68.0),
           ('msb', 'alt3'): (88.1, 28.5), ('msb', 'h^3'): (85.2, 40.2),
           ('lsb', 'h'): (81.9, 48.7), ('lsb', 'h(1-h)'): (74.5, 68.2),
           ('lsb', 'alt3'): (88.7, 46.1), ('lsb', 'h^3'): (93.1, 46.5)}
    for (enc, name), (ei, ea) in exp.items():
        r, a = tailpct(1000, 1000, (1,), enc, INCR[name])
        check(f"dim1000 {enc} {name} rank%", r, ei, 0.8)
        check(f"dim1000 {enc} {name} act%", a, ea, 0.8)

def S7():
    emit("\nS7. Palindrome-distance order parameter, width 800, 800 steps (tail mean, baseline 2.0)")
    exp = {('msb', 'linear', 1): 1.81, ('lsb', 'linear', 1): 2.22,
           ('msb', 'poly', 1): 2.02, ('lsb', 'poly', 1): 2.03,
           ('msb', 'linear', 2): 1.76, ('lsb', 'linear', 2): 2.24}
    for (enc, var, seed), ev in exp.items():
        _, _, p = run(800, 800, seed, enc, var, track_pal=True)
        check(f"{enc} {var} seed{seed} pal-dist tail", p[-100:].mean(), ev, 0.06)


def S8():
    emit("\nS8. Mechanism of the frozen phase (confinement, consistency,")
    emit("    out-of-sample residues, frozen-core structure, threshold)")

    # -- fast vectorized runner, gated below against the canonical run() --
    def run_fast(dim, steps, seed, msb=True, poly=False, tail=100,
                 track_change_tail=None):
        rng = np.random.default_rng(seed)
        n = (rng.permutation(dim) % 256).astype(np.int64)
        c = rng.integers(0, 2, dim).astype(np.int64)
        acts = []; prev = c.copy()
        changed = np.zeros(dim, dtype=bool)
        for t in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            shift = (7 - h) if msb else h
            c = (n >> shift) & 1
            g = h * (1 - h) if poly else h
            n = ((n + (2 * c - 1) * g - 1) % dim) & 0xFF
            if t >= steps - tail:
                acts.append(np.mean(c != prev))
            if track_change_tail is not None and t >= steps - track_change_tail:
                changed |= (c != prev)
            prev = c.copy()
        return c, n, float(np.mean(acts)) * 100, changed

    for enc in ('msb', 'lsb'):
        cf, nf, _, _ = run_fast(160, 30, 7, msb=(enc == 'msb'))
        r_, m_, _ = run(160, 30, 7, enc, 'linear', 'plain')
        # reproduce canonical final state for comparison
        rng = np.random.default_rng(7)
        words = [int(v) for v in rng.permutation(160) % 256]
        cc = rng.integers(0, 2, 160)
        rules = np.array([d2b(w, enc == 'msb') for w in words], dtype=int)
        K = kernel(160)
        for _ in range(30):
            hh = (K @ cc).astype(int)
            for j in range(160):
                hj = int(hh[j]); cc[j] = rules[j][hj]
                num = b2d(rules[j], enc == 'msb'); sgn = 2 * cc[j] - 1
                rules[j] = d2b(((num + sgn * hj - 1) % 160) & 0xFF, enc == 'msb')
        nn = np.array([b2d(r, enc == 'msb') for r in rules])
        checkbool(f"fast runner == canonical, 30 steps, {enc}",
                  bool(np.array_equal(cf, cc) and np.array_equal(nf, nn)))

    # -- confinement lemma and rotations --
    def eventual_cycle(N, v):
        T = [((x + v) % N) & 0xFF for x in range(256)]
        on = set()
        for s in range(256):
            seen = set(); x = s
            while x not in seen: seen.add(x); x = T[x]
            cyc = [x]; y = T[x]
            while y != x: cyc.append(y); y = T[y]
            on.update(cyc)
        return sorted(on)
    ok = True
    for r in (8, 16, 32, 48, 64, 80, 96, 128):
        N = 512 + r
        for v in range(-7, 0):
            ok &= max(eventual_cycle(N, v)) < r
    checkbool("confinement: cycles of T_v inside [0,r), v=-7..-1, r=8..128", ok)
    ok = True
    for r in (8, 32, 128):
        N = 512 + r
        for v in (0, 1, 2):
            ok &= all((((x + v) % N) & 0xFF) == (x + v) % 256 for x in range(256))
    checkbool("rotation: T_v = +v on Z_256 for v=0,1,2", ok)

    # -- consistency tables vs the analytic pattern --
    def consistent(N, v, bitpos, cval):
        cyc = eventual_cycle(N, v)
        return {(x >> bitpos) & 1 for x in cyc} == {cval}
    ok = True
    for r in (8, 16, 32, 48, 64, 80, 96, 128):
        N = 512 + r
        ok &= consistent(N, -1, 7, 0) == (r <= 128)
        ok &= consistent(N, -2, 6, 0) == (r <= 64)
        ok &= consistent(N, -3, 5, 0) == (r <= 32)
        ok &= not consistent(N, 1, 6, 1)   # (1,h=2): rotation, varies
        ok &= not consistent(N, 2, 4, 1)   # (1,h=3): rotation, varies
    checkbool("msb-linear consistency table matches the analytic pattern", ok)
    ok = True
    for r in (8, 16, 32, 48, 64, 80, 96, 128):
        N = 512 + r
        ok &= consistent(N, -1, 7, 0) == (r <= 128)   # (0,h=0) quad v=-1
        ok &= consistent(N, -1, 6, 0) == (r <= 64)    # (0,h=1) quad v=-1
        ok &= not consistent(N, 1, 5, 0)              # (0,h=2) quad v=+1
        ok &= not consistent(N, -1, 6, 1)             # (1,h=1) quad: bit6=1 impossible
        ok &= not consistent(N, -3, 5, 1)             # (1,h=2) quad
        ok &= not consistent(N, -7, 4, 1)             # (1,h=3) quad
    checkbool("msb-quadratic consistency: no frozen 1-cells at any residue", ok)
    ok = True
    for r in (16, 32, 64):
        N = 512 + r
        for v, b in ((-1, 0), (-2, 1), (-3, 2)):
            cyc = eventual_cycle(N, v)
            ok &= len({(x >> b) & 1 for x in cyc}) == 2
    checkbool("lsb: looked-up low bits vary on every confined cycle", ok)

    # -- out-of-sample residues (predictions recorded before the runs) --
    exp_act = {(520, 1): 6.1, (520, 2): 7.0, (528, 1): 7.6, (528, 2): 6.2,
               (560, 1): 4.9, (560, 2): 5.8, (576, 1): 6.5, (576, 2): 5.9,
               (592, 1): 1.4, (592, 2): 2.7, (608, 1): 4.4, (608, 2): 4.3}
    for (N, s), ev in exp_act.items():
        _, _, a, _ = run_fast(N, 800, s)
        check(f"msb-linear tail activity N={N} seed{s}", a, ev, 0.15)
    _, _, a, _ = run_fast(528, 800, 1, msb=False)
    check("lsb control N=528 seed1", a, 48.8, 0.15)

    # -- attraction threshold --
    exp_thr = {(616, 1): 37.1, (616, 2): 36.3, (624, 1): 44.9, (624, 2): 45.7,
               (632, 1): 46.2, (632, 2): 46.5}
    for (N, s), ev in exp_thr.items():
        _, _, a, _ = run_fast(N, 800, s)
        check(f"active-phase warmup window (t=800) N={N} (r={N % 256}) seed{s}",
              a, ev, 0.15)

    # -- fine transition scan, 3000 steps: the frozen/active boundary --
    exp_fine = {(608, 1): 2.0, (608, 2): 2.7, (609, 1): 18.2, (609, 2): 17.4,
                (610, 1): 35.7, (610, 2): 39.2,
                (864, 1): 1.9, (865, 1): 22.3, (866, 1): 39.1}
    for (N, s), ev in exp_fine.items():
        _, _, a, _ = run_fast(N, 3000, s)
        check(f"fine scan (t=3000) N={N} (r={N % 256}) seed{s}", a, ev, 0.15)

    # -- frozen-core structure, 2000 steps, seed 1 --
    words_ok = iso_ok = bit6_ok = True
    m101_got, core1_got = [], []
    for N in (520, 544, 560, 576, 592, 608):
        r = N % 256
        c, n, _, changed = run_fast(N, 2000, 1, tail=200, track_change_tail=200)
        frozen = ~changed
        core = frozen & np.roll(frozen, 1) & np.roll(frozen, -1)
        idx = np.where(core)[0]
        ones = [j for j in idx if c[j] == 1]
        core1_got.append(len(ones))
        iso_ok &= all(c[(j - 1) % N] == 0 and c[(j + 1) % N] == 0 for j in ones)
        m101_got.append(sum(1 for j in idx if c[j] == 0 and
                            c[(j - 1) % N] == 1 and c[(j + 1) % N] == 1))
        zw = n[[j for j in idx if c[j] == 0]]
        if len(zw): words_ok &= bool(np.all(zw < r))
        ow = n[[j for j in ones]]
        if len(ow): bit6_ok &= bool(np.all((ow >> 6) & 1 == 1))
    checkbool("core 0-cell words all below r (six widths)", words_ok)
    checkbool("core 1s all isolated (six widths)", iso_ok)
    checkbool("core 101-motif counts == (3,3,0,0,0,0)", m101_got == [3, 3, 0, 0, 0, 0])
    checkbool("core 1-counts == (42,38,26,18,0,0)", core1_got == [42, 38, 26, 18, 0, 0])
    checkbool("isolated core 1-words all have bit 6 set", bit6_ok)

def S9():
    emit("\nS9. The law across word lengths B (regimes, threshold r_c = 3*2^(B-3),")
    emit("    invasion at r_c+1, excitability crossing, interface velocity)")

    def run_B(dim, steps, seed, B, msb=True, tail=100, core_tail=200, checkpoints=()):
        """B-bit words; identical to S8's run_fast at B = 8 (gated below)."""
        mask = (1 << B) - 1
        rng = np.random.default_rng(seed)
        n = (rng.permutation(dim) % (1 << B)).astype(np.int64)
        c = rng.integers(0, 2, dim).astype(np.int64)
        prev = c.copy(); acts = []; changed = np.zeros(dim, dtype=bool); cp = {}
        for t in range(1, steps + 1):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            shift = (B - 1 - h) if msb else h
            c = (n >> shift) & 1
            n = ((n + (2 * c - 1) * h - 1) % dim) & mask
            ch = (c != prev); prev = c.copy()
            acts.append(float(np.mean(ch)))
            if t > steps - core_tail:
                changed |= ch
            if t in checkpoints:
                cp[t] = 100 * float(np.mean(acts[-tail:]))
        return c, n, 100 * float(np.mean(acts[-tail:])), changed, cp

    # (a) the generalized runner reproduces S8's fast runner at B = 8
    def run_fast8(dim, steps, seed):
        rng = np.random.default_rng(seed)
        n = (rng.permutation(dim) % 256).astype(np.int64)
        c = rng.integers(0, 2, dim).astype(np.int64)
        for _ in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            c = (n >> (7 - h)) & 1
            n = ((n + (2 * c - 1) * h - 1) % dim) & 0xFF
        return c, n
    c8, n8 = run_fast8(608, 500, 1)
    cB, nB, _, _, _ = run_B(608, 500, 1, 8)
    checkbool("run_B(B=8) == S8 fast runner (N=608, 500 steps, seed 1)",
              np.array_equal(c8, cB) and np.array_equal(n8, nB))

    # (b) confinement for B = 9, 10: eventual cycles of T_v inside [0, r)
    conf_ok = True
    for B in (9, 10):
        base = 1 << B
        for r in range(8, base // 2 + 1, 8):
            N = base + r
            for v in range(-7, 0):
                nxt = ((np.arange(base) + v) % N) & (base - 1)
                x = np.arange(base)
                for _ in range(base + 8):
                    x = nxt[x]
                conf_ok &= bool(np.all(x < r))
    checkbool("confinement: cycles of T_v inside [0,r), v=-7..-1, B=9,10, r=8..2^(B-1) step 8", conf_ok)

    # (c) the exact excitability crossing e_1(r_c) = e_2(r_c) = 1/3
    from fractions import Fraction
    cross_ok = True
    for B in range(6, 13):
        base = 1 << B; rc = 3 * base // 8
        def e(bit, r):
            return Fraction(sum(1 for w in range(r) if (w >> bit) & 1), r)
        cross_ok &= (e(B - 2, rc) == Fraction(1, 3) and e(B - 3, rc) == Fraction(1, 3))
        cross_ok &= all(e(B - 2, r) < Fraction(1, 3) < e(B - 3, r) for r in range(base // 4 + 1, rc))
        cross_ok &= all(e(B - 2, r) == e(B - 3, r) for r in range(rc, base // 2 + 1))
        cross_ok &= all(e(B - 1, r) == 0 for r in range(1, base // 2 + 1))
    checkbool("e_1(r_c) = e_2(r_c) = 1/3; e_1 < 1/3 < e_2 below r_c; e_1 = e_2 above; e_0 = 0 (B=6..12)", cross_ok)

    # (d) the three regimes at B = 9, 10 (predictions recorded before the runs)
    exp_reg = {
        9:  {16: (6.77, 8.31), 64: (6.85, 6.77), 96: (5.08, 5.59), 128: (4.53, 4.78),
             160: (0.40, 0.43), 192: (0.43, 1.11), 256: (46.35, 46.32), 0: (46.68, 46.60)},
        10: {32: (7.73, 7.63), 128: (6.63, 6.48), 192: (4.60, 4.72), 256: (4.86, 4.34),
             320: (0.24, 0.60), 384: (0.71, 0.92), 512: (45.79, 46.07), 0: (45.44, 45.56)},
    }
    exp_n1 = {9: {16: (39, 49), 64: (49, 61), 96: (29, 23), 128: (40, 43)},
              10: {32: (70, 78), 128: (128, 96), 192: (63, 58), 256: (59, 58)}}
    exp_m101 = {9: {16: (3, 10), 64: (9, 5)}, 10: {32: (2, 10), 128: (22, 16)}}
    exp_lsb = {9: (64, 47.87), 10: (128, 48.47)}
    for B in (9, 10):
        base = 1 << B; steps = 2000 * (base // 256)
        words_ok = iso_ok = bit_ok = True; n1_got = {}; m101_got = {}
        for r, exps in exp_reg[B].items():
            N = base + r if r > 0 else 2 * base
            for seed, ex in zip((1, 2), exps):
                c, n, act, changed, _ = run_B(N, steps, seed, B)
                check(f"B={B} r={r} N={N} seed{seed} tail activity", act, ex, 0.15)
                core = ~(changed | np.roll(changed, 1) | np.roll(changed, -1))
                zeros = core & (c == 0); ones = core & (c == 1)
                if zeros.any(): words_ok &= bool(np.all(n[zeros] < (r if r > 0 else base)))
                iso = ones & (np.roll(c, 1) == 0) & (np.roll(c, -1) == 0)
                iso_ok &= bool(ones.sum() == iso.sum())
                if iso.any(): bit_ok &= bool(np.all((n[iso] >> (B - 2)) & 1 == 1))
                n1_got.setdefault(r, []).append(int(ones.sum()))
                m101_got.setdefault(r, []).append(int((core & (c == 0) & (np.roll(c, 1) == 1)
                                                      & (np.roll(c, -1) == 1)).sum()))
        checkbool(f"B={B} core 0-cell words all below r", words_ok)
        checkbool(f"B={B} core 1s all isolated", iso_ok)
        checkbool(f"B={B} isolated core 1-words have bit {B-2} set", bit_ok)
        checkbool(f"B={B} core 1-counts regimes 1-2 == {exp_n1[B]}",
                  all(tuple(n1_got[r]) == v for r, v in exp_n1[B].items()))
        checkbool(f"B={B} regime 3 and r=2^(B-1), r=0: no core 1s",
                  all(n1_got[r] == [0, 0] for r in exp_reg[B] if r == 0 or r > base // 4))
        checkbool(f"B={B} 101 motifs regime 1 == {exp_m101[B]}",
                  all(tuple(m101_got[r]) == v for r, v in exp_m101[B].items()))
        checkbool(f"B={B} 101 motifs absent in regimes 2-3",
                  all(m101_got[r] == [0, 0] for r in exp_reg[B] if r > base // 8))
        r_l, ex_l = exp_lsb[B]
        _, _, act_l, _, _ = run_B(base + r_l, steps, 1, B, msb=False)
        check(f"B={B} lsb control r={r_l} tail activity", act_l, ex_l, 0.15)

    # (e) the threshold r_c = 3*2^(B-3) at B = 6..12, scaled time T_B = max(1500, 3000*2^(B-8))
    exp_thr = {
        6:  {22: (0.00, 0.00), 23: (0.00, 0.00), 24: (0.00, 0.00), 25: (29.37, 27.11), 26: (35.60, 36.07), 27: (40.42, 40.26)},
        7:  {46: (0.00, 0.00), 47: (0.00, 0.00), 48: (3.09, 4.60), 49: (24.89, 37.23), 50: (41.13, 42.08), 51: (42.12, 43.95)},
        8:  {94: (0.00, 0.94), 95: (0.00, 0.00), 96: (1.21, 0.00), 97: (24.50, 17.29), 98: (32.83, 40.15), 99: (39.86, 39.06)},
        9:  {190: (0.43, 0.36), 191: (0.51, 0.88), 192: (0.44, 1.39), 193: (10.29, 6.15), 194: (35.03, 32.42), 195: (35.40, 36.04)},
        10: {382: (0.44, 0.47), 383: (0.00, 1.07), 384: (0.88, 0.77), 385: (5.39, 9.64), 386: (30.46, 35.36), 387: (31.56, 31.89)},
        11: {766: (0.42, 0.97), 767: (0.45, 0.86), 768: (0.67, 1.09), 769: (4.84, 4.53), 770: (29.85, 28.42), 771: (28.98, 30.55)},
        12: {1534: (0.16, 0.18), 1535: (0.40, 0.17), 1536: (0.39, 0.40), 1537: (5.36, 4.67), 1538: (27.97, 26.92), 1539: (28.89, 29.69)},
    }
    emit("  (widths N = 2^B + r; at B = 8 this is the class 350-355, a third width class")
    emit("   beside 608-610 and 864-866 of S8)")
    triple_ok = True
    for B in range(6, 13):
        base = 1 << B; rc = 3 * base // 8; T = max(1500, 3000 * base // 256)
        for r, exps in exp_thr[B].items():
            acts = []
            for seed in (1, 2):
                _, _, act, _, _ = run_B(base + r, T, seed, B)
                acts.append(act)
                if exps is not None:
                    check(f"B={B} r={r} (r_c{r-rc:+d}) T={T} seed{seed} tail activity", act, exps[seed - 1], 0.15)
                else:
                    emit(f"  [info] B={B} r={r} (r_c{r-rc:+d}) T={T} seed{seed} tail activity: {act:.2f}")
            if r <= rc:
                triple_ok &= max(acts) <= 5.0
            elif r == rc + 1:
                triple_ok &= min(acts) >= 2.5
            else:
                triple_ok &= min(acts) >= 24.0
    checkbool("B=6..12: activity <= 5% for r <= r_c; >= 2.5% at r_c+1; >= 24% for r >= r_c+2", triple_ok)

    # (f) growth in time at r_c+1 (invasion) and stationarity at r_c, B = 8..12
    exp_grow = {
        8:  {97: ((24.50, 32.68, 41.80), (17.29, 22.93, 35.96)), 96: ((1.21, 0.00, 0.00), (0.00, 0.00, 0.00))},
        9:  {193: ((10.29, 18.53, 29.54), (6.15, 9.47, 16.75)), 192: ((0.44, 0.70, 0.65), (1.39, 1.50, 1.84))},
        10: {385: ((5.39, 8.93, 14.96), (9.64, 15.67, 24.67)), 384: ((0.88, 0.96, 1.30), (0.77, 0.99, 1.18))},
        11: {769: ((4.84, 7.91, 12.89), (4.53, 7.84, 13.28))},
        12: {1537: ((5.36, 8.83, 14.63), (4.67, 8.01, 13.82))},
    }
    grow_ok = stat_ok = True
    for B, d in exp_grow.items():
        base = 1 << B; rc = 3 * base // 8; T = 3000 * base // 256
        for r, exps in d.items():
            for seed in (1, 2):
                _, _, _, _, cp = run_B(base + r, 4 * T, seed, B, checkpoints=(T, 2 * T, 4 * T))
                got = (cp[T], cp[2 * T], cp[4 * T])
                for k, t in enumerate((T, 2 * T, 4 * T)):
                    check(f"B={B} r={r} seed{seed} activity at t={t}", got[k], exps[seed - 1][k], 0.15)
                if r == rc + 1:
                    grow_ok &= got[0] < got[1] < got[2]
                else:
                    stat_ok &= got[2] <= 2.0
    checkbool("r_c+1: activity strictly increasing over T, 2T, 4T (B=8..12, both seeds)", grow_ok)
    checkbool("r_c: activity <= 2% at 4T (B=8..10, both seeds)", stat_ok)

    # (g) prepared interface at B = 10: half of a ring of 4*2^10 + r cells
    #     is the frozen sea (words < r), half random; the number of cells
    #     changed within the last 50 steps is counted in the PREPARED half
    #     only, so that only the two fronts entering it are measured
    #     (domains growing inside the random half before they merge are
    #     not counted); v = half the slope of that count over t >= 40000
    #     while it is below 0.8 of the prepared half.
    def front(B, r, seed, k=4, steps=200000, win=50, every=500, t0=40000):
        base = 1 << B; N = k * base + r; mask = base - 1
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
                out.append((t, int((recent[:half] < win).sum())))
        A = np.array(out)
        m = (A[:, 0] >= t0) & (A[:, 1] < 0.8 * half)
        v = np.polyfit(A[m, 0], A[m, 1], 1)[0] / 2 if m.sum() >= 5 else float('nan')
        # the end state: cells of the random half that changed during the last 2000 steps, their distance
        # (quiet cells in between) from the two boundaries with the prepared sea, the cells of the random
        # half changed within the last `win` steps, and the sea words outside [0, r)
        late = np.flatnonzero(recent[half:] < 2000) + half
        geo = dict(dist=(int(late.min() - half), int(N - 1 - late.max())) if late.size else (None, None),
                   active=int((recent[half:] < win).sum()), random_half=N - half,
                   converted=int((n[:half] >= r).sum()),
                   flickers=[int(x) for x in A[A[:, 0] >= t0, 1] if x > 0])
        return v, A, geo
    exp_v = {376: (0.00, 0.00), 380: (0.00, 0.00), 382: (0.00, 0.00), 383: (0.00, 0.00),
             384: (0.00, 0.00), 385: (0.54, 0.64), 386: (4.97, 4.89), 388: (5.04, 5.39),
             391: (5.98, 6.04), 400: (7.77, 8.10), 416: (9.62, 8.31), 448: (13.27, 13.19),
             480: (14.61, 14.48), 512: (14.15, 14.35)}
    v_ok = True; dists = []; actives = []; converted = 0; flick = {}
    for r, exps in exp_v.items():
        for seed in (1, 2):
            v, A, geo = front(10, r, seed)
            check(f"B=10 interface r={r} e={(r-256)/r:.4f} seed{seed} v (10^-3 cells/step, prepared half)",
                  1000 * v, exps[seed - 1], 0.05)
            if r <= 384:
                v_ok &= abs(v) <= 0.0001
                dists += list(geo['dist']); actives.append(geo['active']); converted += geo['converted']
                if geo['flickers']: flick[(r, seed)] = geo['flickers']
            elif r == 385: v_ok &= v >= 0.0002
            else: v_ok &= v >= 0.003
    checkbool("interface velocity: |v| <= 1e-4 for r <= 384; >= 2e-4 at 385; >= 3e-3 for r >= 386", v_ok)
    # the ten runs with r <= 384: what the zero of the count rests on
    checkbool("r <= 384: the count in the prepared half over t >= 40000 is zero in eight runs; nonzero samples (of 321 each) "
              "only at (r, seed) = (382, 2): one, of 1 cell, and (383, 2): four, of 1, 1, 2 and 1 cells",
              flick == {(382, 2): [1], (383, 2): [1, 1, 2, 1]})
    checkbool("r <= 384: no word of the prepared half lies outside [0, r) at the end of any of the ten runs", converted == 0)
    checkbool("r <= 384: cells of the random half (2236 to 2240 cells) changed within the last 50 steps, per run == "
              "[140, 156, 86, 223, 83, 173, 222, 371, 529, 616]", actives == [140, 156, 86, 223, 83, 173, 222, 371, 529, 616])
    checkbool("r <= 384: distance of the nearest changing cell from the sea at the twenty boundaries, sorted == "
              "[0, 1, 7, 29, 32, 38, 39, 41, 42, 45, 45, 46, 50, 52, 98, 104, 113, 114, 118, 193]",
              sorted(dists) == [0, 1, 7, 29, 32, 38, 39, 41, 42, 45, 45, 46, 50, 52, 98, 104, 113, 114, 118, 193])

def S10():
    emit("\nS10. Ensembles for Table 1: eight seeds per cell, 800 steps; means,")
    emit("     standard deviations, and the position of the two-seed values")

    def run_fast(dim, steps, seed, msb=True, poly=False, tail=100):
        rng = np.random.default_rng(seed)
        n = (rng.permutation(dim) % 256).astype(np.int64)
        c = rng.integers(0, 2, dim).astype(np.int64)
        acts = []; ranks = []; prev = c.copy()
        for t in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            shift = (7 - h) if msb else h
            c = (n >> shift) & 1
            g = h * (1 - h) if poly else h
            n = ((n + (2 * c - 1) * g - 1) % dim) & 0xFF
            if t >= steps - tail:
                acts.append(np.mean(c != prev)); ranks.append(len(np.unique(n)))
            prev = c.copy()
        return 100 * float(np.mean(acts)), 100 * float(np.mean(ranks)) / 256

    # gate: fast runner == canonical run() for the quadratic increment too (both readings)
    for enc in ('msb', 'lsb'):
        r_, m_, _ = run(160, 30, 7, enc, 'poly', 'plain')
        a_fast, k_fast = run_fast(160, 30, 7, msb=(enc == 'msb'), poly=True, tail=30)
        checkbool(f"fast runner == canonical, quadratic increment, {enc}: activity and rank traces",
                  abs(100 * m_[-30:].mean() / 160 - a_fast) < 1e-9 and abs(100 * r_[-30:].mean() / 256 - k_fast) < 1e-9)

    widths = (512, 768, 1024, 640, 896, 544, 800, 1056)
    combos = (('msb', False), ('lsb', False), ('msb', True), ('lsb', True))
    # Table 1 values (two-seed means, rank / activity), transcribed from the paper
    table1 = {
        512:  ((12.8, 47.3), (75.6, 48.4), (34.2, 67.6), (67.1, 66.5)),
        768:  ((13.5, 47.3), (83.8, 48.5), (46.9, 65.7), (74.0, 67.5)),
        1024: ((13.8, 47.3), (87.6, 48.2), (54.6, 66.0), (80.4, 66.3)),
        640:  ((10.6, 46.1), (45.5, 48.4), (55.6, 17.4), (49.8, 66.5)),
        896:  ((11.1, 46.5), (46.9, 48.4), (56.6, 20.1), (54.4, 66.7)),
        544:  ((23.7, 6.3), (12.8, 48.7), (27.4, 13.3), (37.6, 67.6)),
        800:  ((29.6, 7.6), (13.0, 48.2), (32.8, 14.5), (43.1, 67.6)),
        1056: ((31.9, 7.7), (13.1, 48.2), (34.5, 15.8), (47.0, 67.3)),
    }
    sd_lin_active = sd_lin_frozen = sd_quad = sd_rank = 0.0
    within = True
    for N in widths:
        for k, (enc, poly) in enumerate(combos):
            A = []; K = []
            for seed in range(1, 9):
                a, kk = run_fast(N, 800, seed, msb=(enc == 'msb'), poly=poly)
                A.append(a); K.append(kk)
            A = np.array(A); K = np.array(K)
            t_rank, t_act = table1[N][k]
            emit(f"  N={N:4d} {enc} {'h(1-h)' if poly else 'h     '}  activity {A.mean():5.1f} +- {A.std(ddof=1):4.2f}"
                 f"  rank {K.mean():5.1f} +- {K.std(ddof=1):4.2f}   (table: {t_act} / {t_rank})")
            if poly: sd_quad = max(sd_quad, A.std(ddof=1))
            elif A.mean() < 25: sd_lin_frozen = max(sd_lin_frozen, A.std(ddof=1))
            else: sd_lin_active = max(sd_lin_active, A.std(ddof=1))
            sd_rank = max(sd_rank, K.std(ddof=1))
            within &= abs(t_act - A.mean()) <= max(2 * A.std(ddof=1), 0.3) and abs(t_rank - K.mean()) <= max(2 * K.std(ddof=1), 0.3)
    emit(f"  max sd of activity: linear active {sd_lin_active:.2f}, linear frozen {sd_lin_frozen:.2f}, quadratic {sd_quad:.2f}; max sd of rank {sd_rank:.2f}")
    checkbool("every two-seed Table 1 value within two sd (or 0.3 points) of its eight-seed mean", within)
    check("max sd of activity, linear increment, active cells (points)", sd_lin_active, 0.27, 0.02)
    check("max sd of activity, linear increment, frozen cells (points)", sd_lin_frozen, 1.21, 0.02)
    check("max sd of activity, quadratic increment (points)", sd_quad, 3.62, 0.02)
    check("max sd of rank (points of 256)", sd_rank, 3.21, 0.02)

def S11():
    emit("\nS11. Statements added or corrected in version 3: exhaustive forms of the")
    emit("     propositions, the scope of the tables in time, the palindromic start,")
    emit("     the reading-independent activity, the time-periodic remark")

    PALD = np.array([sum(((x >> i) & 1) != ((x >> (7 - i)) & 1) for i in range(4)) for x in range(256)])
    PALS = np.array([x for x in range(256) if int(f'{x:08b}'[::-1], 2) == x])
    lin = lambda h: h
    quad = lambda h: h * (1 - h)

    def start(dim, seed):
        rng = np.random.default_rng(seed)
        n = (rng.permutation(dim) % 256).astype(np.int64)
        c = rng.integers(0, 2, dim).astype(np.int64)
        return c, n

    def evolve(c, n, dim, steps, msb=True, g=lin, rank=False, pal=False, mark=0):
        """the dynamics of Section 2 on integer words (B = 8). Returns the final state and a dict of
        per-step traces: 'act' (activity in %), on request 'rank' (distinct words) and 'pal' (mean
        palindrome distance), and 'changed' (cells that changed during the last `mark` steps)"""
        act = np.empty(steps); rk = np.empty(steps) if rank else None; pd = np.empty(steps) if pal else None
        changed = np.zeros(dim, dtype=bool)
        for t in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            c2 = (n >> ((7 - h) if msb else h)) & 1
            n = ((n + (2 * c2 - 1) * g(h) - 1) % dim) & 0xFF
            act[t] = 100 * np.mean(c2 != c)
            if t >= steps - mark: changed |= (c2 != c)
            c = c2
            if rank: rk[t] = len(np.unique(n))
            if pal: pd[t] = PALD[n].mean()
        return c, n, {'act': act, 'rank': rk, 'pal': pd, 'changed': changed}

    def largest_domain(changed):
        """largest cluster of changing cells on the ring, clusters being separated by three or more quiet cells"""
        N = len(changed); idx = np.flatnonzero(changed)
        if idx.size == 0: return 0
        gaps = np.diff(np.concatenate([idx, [idx[0] + N]]))
        cuts = np.flatnonzero(gaps >= 4)
        if cuts.size == 0: return N
        return max(int((idx[cuts[(k + 1) % cuts.size]] - idx[(cuts[k] + 1) % idx.size]) % N) + 1 for k in range(cuts.size))

    # -- gate: this runner against the canonical run(), both readings, both increments --
    for enc in ('msb', 'lsb'):
        for var, g in (('linear', lin), ('poly', quad)):
            r_, m_, p_ = run(160, 30, 7, enc, var, 'plain', track_pal=True)
            c0, n0 = start(160, 7)
            _, _, tr = evolve(c0, n0, 160, 30, msb=(enc == 'msb'), g=g, rank=True, pal=True)
            checkbool(f"S11 runner == canonical, 30 steps, {enc} {var}: activity, rank and palindrome-distance traces",
                      bool(np.allclose(tr['act'] * 160 / 100, m_) and np.array_equal(tr['rank'], r_) and np.allclose(tr['pal'], p_)))

    # ---------------------------------------------------------------- (a) Theorem 2
    emit("  (a) Theorem 2: the full conjugation rho M rho against M from the reflected words")
    def run_conj(dim, steps, init_ints, seed_c, encoding, variant, conj):
        msb = (encoding == 'msb'); poly = (variant == 'poly')
        rng = np.random.default_rng(seed_c)
        rules = np.array([d2b(int(v), msb) for v in init_ints], dtype=int)
        c = rng.integers(0, 2, dim); K = kernel(dim); traj = []
        for _ in range(steps):
            h = (K @ c).astype(int)
            for j in range(dim):
                w = rules[j][::-1] if conj else rules[j]        # rho before the step
                hj = int(h[j]); c[j] = w[hj]                    # the lookup addresses the reflected word
                num = b2d(w, msb); sgn = 2 * c[j] - 1
                incr = sgn * hj * (1 - hj) if poly else sgn * hj
                nb = d2b(((num + incr - 1) % dim) & 0xFF, msb)
                rules[j] = nb[::-1] if conj else nb             # rho after the step
            traj.append((c.copy(), rules.copy()))
        return traj
    rng = np.random.default_rng(1); v0 = rng.permutation(800) % 256
    bitrev = lambda x: int(bin(int(x))[2:].zfill(8)[::-1], 2)
    for variant in ('linear', 'poly'):
        tA = run_conj(800, 150, v0, 7, 'msb', variant, True)
        tB = run_conj(800, 150, [bitrev(x) for x in v0], 7, 'msb', variant, False)
        checkbool(f"rho M rho == M in reflected coordinates, 150 steps, {variant}: bits identical, words mirror images",
                  all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1][:, ::-1]) for a, b in zip(tA, tB)))

    # ---------------------------------------------------------------- (b) Proposition 1
    emit("  (b) Proposition 1 at every residue")
    conf_ok = map_ok = rot_ok = sharp_ok = True
    for r in range(1, 256):
        N = 512 + r; x = np.arange(256)
        for v in range(-7, 0):
            T = ((x + v) % N) & 0xFF
            P = T.copy()
            for _ in range(9): P = P[P]                 # T^512: every word is on its eventual cycle
            if v >= -r:
                conf_ok &= bool(P.max() < r)
                inside = np.arange(r)
                map_ok &= bool(np.array_equal(T[:r], np.where(inside >= -v, inside + v, inside + v + r)))
            else:
                sharp_ok &= bool(P.max() >= r)          # outside the hypothesis the cycles leave [0, r)
        for v in (0, 1, 2):
            rot_ok &= bool(np.array_equal(((x + v) % N) & 0xFF, (x + v) % 256))
    checkbool("confinement: for every r = 1..255 and v = -7..-1 with v >= -r the eventual cycles lie in [0, r)", conf_ok)
    checkbool("on [0, r) the map is n -> n + v, re-entering at n + v + r (same range)", map_ok)
    checkbool("for v < -r (r = 1..6) the eventual cycles leave [0, r)", sharp_ok)
    checkbool("rotation: T_v = +v on Z_256 for v = 0, 1, 2 at every r = 1..255", rot_ok)

    # ---------------------------------------------------------------- (c) Proposition 2
    emit("  (c) Proposition 2 cycle by cycle, every residue, B = 6..12; configurations at B = 8")
    def cycle_fraction(B, N, v, bit, val):
        """fraction of the words on eventual cycles of T_v whose whole cycle shows `val` at `bit`"""
        base = 1 << B; x = np.arange(base)
        T = ((x + v) % N) & (base - 1)
        K = B + 1
        P = T.copy()
        for _ in range(K): P = P[P]
        on = np.unique(P)
        bad = (((x >> bit) & 1) != val)
        Q = T.copy()
        for _ in range(K):
            bad = bad | bad[Q]; Q = Q[Q]
        return float(np.mean(~bad[on]))
    for B in range(6, 13):
        base = 1 << B; ok = True; partial = []
        for r in range(0, base):
            N = 2 * base + r
            f0 = cycle_fraction(B, N, -1, B - 1, 0)
            f1 = cycle_fraction(B, N, -2, B - 2, 0)
            f2 = cycle_fraction(B, N, -3, B - 3, 0)
            ok &= ((f0 == 1.0) == (1 <= r <= base // 2)) and f0 in (0.0, 1.0)
            ok &= ((f1 == 1.0) == (2 <= r <= base // 4)) and f1 in (0.0, 1.0)
            ok &= ((f2 == 1.0) == (3 <= r <= base // 8))
            if 0.0 < f2 < 1.0: partial.append((r, round(f2, 4)))
            if r in (0, 1, base // 8, base // 4, base // 2, base - 1):
                ok &= cycle_fraction(B, N, 1, B - 3, 1) == 0.0 and cycle_fraction(B, N, 2, B - 4, 1) == 0.0
        exc = (base // 8 + 1, round(2 / 3, 4)) if B % 2 == 0 else (base // 8 + 2, round(1 / 3, 4))
        checkbool(f"B={B}: 0-cells with h=0, 1, 2 consistent on every cycle exactly for 1<=r<=2^(B-1), 2<=r<=2^(B-2), 3<=r<=2^(B-3); "
                  f"h=2 on part of the cycles only at r={exc[0]} (fraction {exc[1]}); 1-cells with h>=2 never", ok and partial == [exc])
    def frozen_for(r, ones, mid=None, steps=2000, seed=5):
        """does a prepared configuration at N = 512 + r keep its cells for `steps` steps?"""
        N = 512 + r; rng = np.random.default_rng(seed)
        c = np.zeros(N, dtype=np.int64); c[list(ones)] = 1
        n = rng.integers(0, r, N).astype(np.int64)                       # sea words on their cycles
        n[list(ones)] = rng.integers(0, 256, len(ones)) | 64             # static words with bit 6 set
        if mid is not None: n[mid[0]] = mid[1]
        c0 = c.copy()
        for _ in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            c = (n >> (7 - h)) & 1
            n = ((n + (2 * c - 1) * h - 1) % N) & 0xFF
            if not np.array_equal(c, c0): return False
        return True
    sea = {r: frozen_for(r, ()) for r in (1, 2, 3, 32, 33, 34, 64, 65, 128, 129)}
    iso = {r: frozen_for(r, (10, 20, 33, 100)) for r in (1, 2, 3, 32, 33, 34, 64, 65, 128)}
    m101 = {r: frozen_for(r, (50, 52, 120), mid=(51, 0)) for r in (2, 3, 32, 34, 64)}
    m33 = {w: frozen_for(33, (50, 52, 120), mid=(51, w)) for w in (0, 1, 2, 3, 4, 5, 30, 31, 32)}
    checkbool("simulation: the all-zero sea is invariant at r = 1, 2, 3, 32, 33, 34, 64, 65, 128 and not at 129",
              all(sea[r] for r in sea if r != 129) and not sea[129])
    checkbool("simulation: isolated 1s are invariant at r = 2, 3, 32, 33, 34, 64 and not at r = 1, 65, 128",
              all(iso[r] for r in (2, 3, 32, 33, 34, 64)) and not any(iso[r] for r in (1, 65, 128)))
    checkbool("simulation: a 101 motif is invariant at r = 3, 32 and not at r = 2, 34, 64",
              m101[3] and m101[32] and not (m101[2] or m101[34] or m101[64]))
    checkbool("simulation at r = 33: a 101 motif is invariant exactly when its middle word is 0 or 1 modulo 3",
              all(m33[w] == (w % 3 != 2) for w in m33))

    # ---------------------------------------------------------------- (d) Corollary 2
    emit("  (d) Corollary 2: the lsb reading and the quadratic increment at every residue")
    found = set()
    for r in range(0, 256):
        N = 512 + r
        for hh, v, bit in ((0, -1, 0), (1, -2, 1), (2, -3, 2)):
            if cycle_fraction(8, N, v, bit, 0) > 0: found.add((hh, r))
    checkbool("lsb: a 0-cell type is consistent on some cycle only for (h, r) = (0,1), (1,2), (2,3), (2,4), (2,6)",
              found == {(0, 1), (1, 2), (2, 3), (2, 4), (2, 6)})
    def lsb_keeps(N, pattern, w1, w0, steps=5000):
        c = np.tile(pattern, N // len(pattern)).astype(np.int64)
        n = np.where(c == 1, w1, w0).astype(np.int64); c0 = c.copy()
        for _ in range(steps):
            h = c + np.roll(c, 1) + np.roll(c, -1)
            c = (n >> h) & 1
            n = ((n + (2 * c - 1) * h - 1) % N) & 0xFF
            if not np.array_equal(c, c0): return False
        return True
    checkbool("lsb: invariant configurations at r = 1 (all zero, N=513), r = 2 (pattern 100, N=258), r = 4 and 6 (alternating, N=516, 518)",
              lsb_keeps(513, [0], 0, 0) and lsb_keeps(258, [1, 0, 0], 2, 0) and lsb_keeps(516, [1, 0], 2, 0) and lsb_keeps(518, [1, 0], 2, 0))
    checkbool("lsb: the alternating ring is not invariant at r = 2 (N=514), nor the all-zero ring at r = 3 (N=515)",
              not lsb_keeps(514, [1, 0], 2, 0) and not lsb_keeps(515, [0], 0, 0))
    for N, ev in ((513, 48.69), (258, 47.46), (516, 54.27), (518, 49.06)):
        c0, n0 = start(N, 1)
        a = evolve(c0, n0, N, 3000, msb=False)[2]['act']
        check(f"lsb from a random state, N={N} (r={N % 256}) seed1, tail activity at t=3000", a[-100:].mean(), ev, 0.15)
    q_ok = True
    for r in range(1, 256):
        N = 512 + r
        q_ok &= (cycle_fraction(8, N, -1, 7, 0) == 1.0) == (r <= 128)     # (0, h=0), v = -1
        q_ok &= cycle_fraction(8, N, -1, 6, 1) == 0.0                     # (1, h=1): bit 6 never constantly set
        q_ok &= cycle_fraction(8, N, -3, 5, 1) == 0.0                     # (1, h=2)
        q_ok &= cycle_fraction(8, N, -7, 4, 1) == 0.0                     # (1, h=3)
    checkbool("quadratic, msb: the all-zero sea consistent exactly for r <= 128; no frozen 1-cell on any cycle at any r = 1..255", q_ok)

    # ---------------------------------------------------------------- (e) widths below the word modulus
    emit("  (e) Section 4: a width below the word modulus against two of the same residue above it")
    for (N, s), ev in {(100, 1): 0.00, (100, 2): 0.00, (356, 1): 30.76, (356, 2): 23.90, (612, 1): 28.07, (612, 2): 29.55}.items():
        c0, n0 = start(N, s)
        a = evolve(c0, n0, N, 800)[2]['act']
        check(f"msb-linear tail activity at t=800, N={N} (N mod 256 = {N % 256}) seed{s}", a[-100:].mean(), ev, 0.15)

    # ---------------------------------------------------------------- (f) Table 1 beyond 800 steps
    emit("  (f) Table 1 at 2 x 10^4 steps (two-seed means, rank% / activity%), and the msb quadratic")
    emit("      activity at residue 128 followed to 6 x 10^4 steps")
    exp20k = {  # N: ((msb h), (lsb h), (msb h(1-h)), (lsb h(1-h)))
        512:  ((12.04, 47.62), (75.83, 48.56), (11.29, 75.17), (66.63, 66.54)),
        768:  ((12.74, 47.59), (83.86, 48.30), (12.12, 74.62), (73.75, 67.84)),
        1024: ((13.29, 47.56), (87.63, 48.33), (13.27, 74.76), (79.68, 67.17)),
        640:  ((3.37, 47.14), (45.46, 48.43), (50.55, 0.82), (41.38, 68.21)),
        896:  ((3.35, 47.26), (46.89, 48.47), (50.85, 0.47), (44.15, 66.59)),
        544:  ((23.73, 6.35), (12.80, 48.59), (27.44, 13.29), (14.97, 67.27)),
        800:  ((29.17, 7.56), (12.99, 48.36), (32.81, 14.49), (14.18, 67.34)),
        1056: ((31.79, 7.74), (13.13, 48.45), (34.47, 15.75), (12.04, 68.71))}
    for N, cells in exp20k.items():
        for (enc, g, name), (er, ea) in zip((('msb', lin, 'h'), ('lsb', lin, 'h'), ('msb', quad, 'h(1-h)'), ('lsb', quad, 'h(1-h)')), cells):
            R, A = [], []
            for s in (1, 2):
                c0, n0 = start(N, s)
                tr = evolve(c0, n0, N, 20000, msb=(enc == 'msb'), g=g, rank=True)[2]
                R.append(tr['rank'][-100:].mean() / 256 * 100); A.append(tr['act'][-100:].mean())
            check(f"dim {N} (mod {N % 256}) {enc} {name} rank% at t=20000", float(np.mean(R)), er, 0.15)
            check(f"dim {N} (mod {N % 256}) {enc} {name} act% at t=20000", float(np.mean(A)), ea, 0.15)
    exp128 = {(640, 1): (20.17, 5.07, 1.63, 0.94), (640, 2): (14.54, 4.02, 0.00, 0.00),
              (896, 1): (21.55, 5.77, 0.00, 0.00), (896, 2): (18.66, 2.88, 0.94, 0.81)}
    sea_end = []
    for (N, s), evs in exp128.items():
        c0, n0 = start(N, s)
        c1, n1, tr = evolve(c0, n0, N, 60000, g=quad)
        a = tr['act']
        for t, ev in zip((800, 5000, 20000, 60000), evs):
            check(f"msb quadratic, residue 128, N={N} seed{s}: activity% at t={t}", a[t - 100:t].mean(), ev, 0.15)
        if c1.sum() == 0 and n1.max() < 128: sea_end.append((N, s))
    checkbool("msb quadratic, residue 128: the runs that end in the all-zero sea (all cells 0, all words below 128) == (640, 2), (896, 1)",
              sea_end == [(640, 2), (896, 1)])

    # ---------------------------------------------------------------- (g)-(i) Section 5 in time and over seeds
    emit("  (g) residues 80 and 96 over eight seeds; (h) the fine scan; (i) the slow invasion at and below r_c")
    for N, lo, hi in ((592, 0.86, 3.31), (608, 1.06, 6.18)):
        A8, zero = [], 0
        for s in range(1, 9):
            c0, n0 = start(N, s)
            a = evolve(c0, n0, N, 2000)[2]['act']
            A8.append(a[700:800].mean()); zero += int(a[1900:2000].max() == 0)
        check(f"msb-linear N={N} (r={N % 256}), eight seeds, t=800: lowest tail activity", min(A8), lo, 0.15)
        check(f"msb-linear N={N} (r={N % 256}), eight seeds, t=800: highest tail activity", max(A8), hi, 0.15)
        checkbool(f"msb-linear N={N}: runs at zero activity over steps 1900-2000 == {1 if N == 592 else 0} of 8", zero == (1 if N == 592 else 0))
    exp_scan = {90: (0.00, 2.66, 0.00, 2.65), 92: (0.00, 2.48, 0.00, 1.84), 94: (0.00, 3.44, 0.00, 2.84), 95: (0.54, 2.75, 0.00, 3.25),
                96: (0.64, 4.18, 0.00, 4.54), 97: (12.17, 24.48, 18.72, 41.97), 98: (35.73, 39.18, 44.17, 44.96),
                100: (37.77, 40.71, 44.08, 44.95), 104: (42.90, 44.45, 44.62, 45.82)}
    largest97 = []
    for r, (l3, h3, l8, h8) in exp_scan.items():
        N = 512 + r; A3, A8 = [], []
        for s in range(1, 9):
            c0, n0 = start(N, s)
            c1, n1, tr = evolve(c0, n0, N, 3000, mark=100)
            A3.append(tr['act'][-100:].mean())
            if r == 97: largest97.append(largest_domain(tr['changed']))
            A8.append(evolve(c1, n1, N, 5000)[2]['act'][-100:].mean())
        check(f"fine scan r={r}, eight seeds, t=3000: lowest tail activity", min(A3), l3, 0.15)
        check(f"fine scan r={r}, eight seeds, t=3000: highest tail activity", max(A3), h3, 0.15)
        check(f"fine scan r={r}, eight seeds, t=8000: lowest tail activity", min(A8), l8, 0.15)
        check(f"fine scan r={r}, eight seeds, t=8000: highest tail activity", max(A8), h8, 0.15)
    checkbool("r = 97 at t = 3000: the largest domain of changing cells per seed == [75, 75, 89, 127, 143, 79, 75, 52]",
              largest97 == [75, 75, 89, 127, 143, 79, 75, 52])
    exp_long = {94: ((0, 0, 0, 0, 0), (7, 11, 11, 11, 11)), 95: ((0, 0, 0, 4, 5), (0, 4, 5, 5, 5)), 96: ((0, 0, 3, 8, 10), (0, 2, 2, 2, 2))}
    marks = (3000, 10000, 30000, 100000, 200000)
    for r, (above, zero) in exp_long.items():
        N = 512 + r; got_above = [0] * 5; got_zero = [0] * 5; sea = 0
        for s in range(1, 13):
            c0, n0 = start(N, s)
            c1, n1, tr = evolve(c0, n0, N, 200000)
            a = tr['act']
            for k, m in enumerate(marks):
                w = a[m - 100:m].mean()
                got_above[k] += int(w > 5.0); got_zero[k] += int(w == 0.0)
            sea += int(c1.sum() == 0 and n1.max() < r)
        checkbool(f"r={r}, twelve seeds: runs with tail activity above 5% at t = 3e3, 1e4, 3e4, 1e5, 2e5 == {above}", tuple(got_above) == above)
        checkbool(f"r={r}, twelve seeds: runs at zero activity at the same times == {zero}", tuple(got_zero) == zero)
        checkbool(f"r={r}: the runs at zero activity at t = 2e5 are in the all-zero sea (all cells 0, all words below r): {zero[-1]} of 12", sea == zero[-1])

    # ---------------------------------------------------------------- (j) excitability along single cycles
    emit("  (j) the excitabilities along the single cycles at r_c = 96")
    from fractions import Fraction
    def per_cycle(N, v, bit):
        T = [((x + v) % N) & 0xFF for x in range(256)]
        xs = list(range(256))
        for _ in range(600): xs = [T[x] for x in xs]
        out, seen = [], set()
        for s in sorted(set(xs)):
            if s in seen: continue
            cyc, y = [], s
            while y not in seen: seen.add(y); cyc.append(y); y = T[y]
            out.append(Fraction(sum((x >> bit) & 1 for x in cyc), len(cyc)))
        return sorted(out)
    checkbool("r = 96: e_2 along the three cycles of T_-3 == 5/16, 11/32, 11/32 (mean 1/3)",
              per_cycle(608, -3, 5) == [Fraction(5, 16), Fraction(11, 32), Fraction(11, 32)])
    checkbool("r = 96: e_1 along the two cycles of T_-2 == 1/3, 1/3", per_cycle(608, -2, 6) == [Fraction(1, 3), Fraction(1, 3)])

    # ---------------------------------------------------------------- (l) Section 6
    emit("  (l) Section 6: the frozen-phase rank over eight seeds; the fast linear cell passing 71 words")
    R20 = []
    for s in range(1, 9):
        c0, n0 = start(800, s)
        rk = evolve(c0, n0, 800, 20000, rank=True)[2]['rank']
        R20.append(rk[-100:].mean())
    check("msb-linear N=800, eight seeds, t=20000: lowest tail rank (words)", min(R20), 69.32, 0.15)
    check("msb-linear N=800, eight seeds, t=20000: highest tail rank (words)", max(R20), 78.23, 0.15)
    check("msb-linear N=800 seed1, t=20000: tail rank (words)", R20[0], 71.11, 0.15)
    first, tails = [], []
    for s in range(1, 9):
        c0, n0 = start(800, s)
        rk = evolve(c0, n0, 800, 5000, msb=False, rank=True)[2]['rank']
        first.append(int(np.flatnonzero(rk <= 71)[0]) + 1); tails.append(rk[-100:].mean())
    checkbool("lsb-linear N=800, seeds 1..8: first step with rank <= 71 == [274, 260, 268, 281, 273, 271, 274, 271]",
              first == [274, 260, 268, 281, 273, 271, 274, 271])
    check("lsb-linear N=800, eight seeds, t=5000: lowest tail rank (words)", min(tails), 33.09, 0.15)
    check("lsb-linear N=800, eight seeds, t=5000: highest tail rank (words)", max(tails), 33.29, 0.15)

    # ---------------------------------------------------------------- (m) Section 7
    emit("  (m) Section 7: the palindrome distance in time; the palindromic start; the lsb activity at every residue")
    exp_pd = {('msb', 'linear', 1): (1.737, 1.779, 1.808, 1.810), ('msb', 'linear', 2): (1.752, 1.760, 1.767, 1.764),
              ('lsb', 'linear', 1): (2.111, 2.134, 2.234, 2.230), ('lsb', 'linear', 2): (2.111, 2.125, 2.231, 2.234),
              ('msb', 'poly', 1): (2.073, 2.042, 2.025, 2.024), ('lsb', 'poly', 1): (2.031, 2.014, 2.018, 2.009)}
    for (enc, var, s), evs in exp_pd.items():
        c0, n0 = start(800, s)
        pd = evolve(c0, n0, 800, 5000, msb=(enc == 'msb'), g=(quad if var == 'poly' else lin), pal=True)[2]['pal']
        for (a_, b_), ev in zip(((50, 100), (100, 200), (400, 500), (4900, 5000)), evs):
            check(f"{enc} {var} seed{s} pal-dist over steps {a_}-{b_}", pd[a_:b_].mean(), ev, 0.006)
    def pal_start(seed):
        rng = np.random.default_rng(seed)
        n = PALS[rng.integers(0, 16, 800)].astype(np.int64)      # words drawn uniformly from the 16 palindromes
        c = rng.integers(0, 2, 800).astype(np.int64)             # then uniform bits
        return c, n
    exp_ps = {1: (73.50, 23.45, 8.31, 13.00, 48.42), 2: (75.375, 26.23, 10.36, 12.94, 48.22)}
    two = {'mr': [], 'ma': [], 'lr': [], 'la': []}
    for s, (e_np, e_mr, e_ma, e_lr, e_la) in exp_ps.items():
        c0, n0 = pal_start(s)
        c1, n1, _ = evolve(c0.copy(), n0.copy(), 800, 1)
        check(f"palindromic start seed{s}: words non-palindromic after one step (%)", 100 * float(np.mean(PALD[n1] > 0)), e_np, 0.01)
        cl, nl, _ = evolve(c0.copy(), n0.copy(), 800, 1, msb=False)
        c2m = evolve(c0.copy(), n0.copy(), 800, 2)[0]; c2l = evolve(c0.copy(), n0.copy(), 800, 2, msb=False)[0]
        checkbool(f"palindromic start seed{s}: after one step the two readings hold the same bits and the same integers "
                  "(so their stored words are mirror images wherever the word is no longer a palindrome); after two steps the bits differ",
                  bool(np.array_equal(c1, cl) and np.array_equal(n1, nl) and np.any(PALD[n1] > 0) and not np.array_equal(c2m, c2l)))
        tr = evolve(c0.copy(), n0.copy(), 800, 800, rank=True)[2]
        check(f"palindromic start seed{s}: msb tail rank%", tr['rank'][-100:].mean() / 2.56, e_mr, 0.15)
        check(f"palindromic start seed{s}: msb tail act%", tr['act'][-100:].mean(), e_ma, 0.15)
        two['mr'].append(tr['rank'][-100:].mean() / 2.56); two['ma'].append(tr['act'][-100:].mean())
        tr = evolve(c0.copy(), n0.copy(), 800, 800, msb=False, rank=True)[2]
        check(f"palindromic start seed{s}: lsb tail rank%", tr['rank'][-100:].mean() / 2.56, e_lr, 0.15)
        check(f"palindromic start seed{s}: lsb tail act%", tr['act'][-100:].mean(), e_la, 0.15)
        two['lr'].append(tr['rank'][-100:].mean() / 2.56); two['la'].append(tr['act'][-100:].mean())
    for key, name, ev in (('mr', 'msb tail rank%', 24.84), ('ma', 'msb tail act%', 9.33), ('lr', 'lsb tail rank%', 12.97), ('la', 'lsb tail act%', 48.32)):
        check(f"palindromic start, two-seed mean: {name}", float(np.mean(two[key])), ev, 0.05)
    RP, RG, AP, AG = [], [], [], []
    for s in range(1, 9):
        c0, n0 = pal_start(s)
        tr = evolve(c0, n0, 800, 800, rank=True)[2]
        RP.append(tr['rank'][-100:].mean() / 2.56); AP.append(tr['act'][-100:].mean())
        c0, n0 = start(800, s)
        tr = evolve(c0, n0, 800, 800, rank=True)[2]
        RG.append(tr['rank'][-100:].mean() / 2.56); AG.append(tr['act'][-100:].mean())
    check("msb tail rank%, eight seeds, palindromic start: mean", float(np.mean(RP)), 24.56, 0.05)
    check("msb tail rank%, eight seeds, palindromic start: sd", float(np.std(RP, ddof=1)), 1.48, 0.05)
    check("msb tail rank%, eight seeds, generic start: mean", float(np.mean(RG)), 29.29, 0.05)
    check("msb tail rank%, eight seeds, generic start: sd", float(np.std(RG, ddof=1)), 1.31, 0.05)
    check("msb tail act%, eight seeds, palindromic start: mean", float(np.mean(AP)), 8.82, 0.05)
    check("msb tail act%, eight seeds, palindromic start: sd", float(np.std(AP, ddof=1)), 1.21, 0.05)
    check("msb tail act%, eight seeds, generic start: mean", float(np.mean(AG)), 7.70, 0.05)
    check("msb tail act%, eight seeds, generic start: sd", float(np.std(AG, ddof=1)), 0.90, 0.05)
    L, Q = {}, {}
    for r in range(256):
        c0, n0 = start(512 + r, 1)
        L[r] = evolve(c0, n0, 512 + r, 3000, msb=False)[2]['act'][-100:].mean()
        c0, n0 = start(512 + r, 1)
        Q[r] = evolve(c0, n0, 512 + r, 3000, msb=False, g=quad)[2]['act'][-100:].mean()
    check("lsb linear, every residue (N = 512 + r, seed 1, t = 3000): lowest tail activity", min(L.values()), 46.25, 0.15)
    check("lsb linear, every residue: highest tail activity", max(L.values()), 55.51, 0.15)
    check("lsb linear, residues 25..255: lowest tail activity", min(L[r] for r in range(25, 256)), 47.27, 0.15)
    check("lsb linear, residues 25..255: highest tail activity", max(L[r] for r in range(25, 256)), 49.45, 0.15)
    check("lsb quadratic, r = 1: tail activity", Q[1], 30.32, 0.15)
    check("lsb quadratic, every residue except r = 1: lowest tail activity", min(Q[r] for r in range(256) if r != 1), 62.60, 0.15)
    check("lsb quadratic, every residue except r = 1: highest tail activity", max(Q[r] for r in range(256) if r != 1), 71.09, 0.15)

    # ---------------------------------------------------------------- (n) Section 9
    emit("  (n) Section 9: Rule 110 on 8 cells; toy widths; the resweep on all histories of Rule 110 on 4 cells")
    def rule110(x, nc):
        out = 0
        for j in range(nc):
            l, m_, r_ = (x >> ((j - 1) % nc)) & 1, (x >> j) & 1, (x >> ((j + 1) % nc)) & 1
            out |= ((110 >> (4 * l + 2 * m_ + r_)) & 1) << j
        return out
    F8 = np.array([rule110(x, 8) for x in range(256)]); y = np.arange(256)
    for _ in range(600): y = F8[y]
    cyc, seen = [], set()
    for s in sorted(set(y.tolist())):
        if s in seen: continue
        Lc, z = 0, s
        while z not in seen: seen.add(z); z = int(F8[z]); Lc += 1
        cyc.append(Lc)
    checkbool("Rule 110 on 8 cells: cycle lengths == [1, 2, 2, 8, 16, 16]", sorted(cyc) == [1, 2, 2, 8, 16, 16])
    counts = []
    for T in range(1, 17):
        z = np.arange(256)
        for _ in range(T): z = F8[z]
        counts.append((int((z == np.arange(256)).sum()), sum(Lc for Lc in cyc if T % Lc == 0)))
    checkbool("T-periodic histories: direct count == sum of cycle lengths dividing T, for T = 1..16", all(a_ == b_ for a_, b_ in counts))
    checkbool("the counts at T = 1, 2, 8, 16 == 1, 5, 13, 45", [counts[T - 1][0] for T in (1, 2, 8, 16)] == [1, 5, 13, 45])
    per_ok = zero_ok = True
    for dim in range(4, 101):
        for s in range(1, 6):
            c, n = start(dim, s); seen = {}; t = 0; key = (c.tobytes(), n.tobytes())
            while key not in seen:
                seen[key] = t
                h = c + np.roll(c, 1) + np.roll(c, -1)
                c = (n >> (7 - h)) & 1
                n = ((n + (2 * c - 1) * h - 1) % dim) & 0xFF
                t += 1; key = (c.tobytes(), n.tobytes())
            per_ok &= (t - seen[key] == dim); zero_ok &= (c.sum() == 0)
    checkbool("toy widths N = 4..100, seeds 1..5 (msb, linear): the cycle reached has period exactly N in all 485 runs", per_ok)
    checkbool("toy widths: on that cycle the configuration is all-zero and only the words move", bool(zero_ok))
    import itertools
    F4 = [rule110(x, 4) for x in range(16)]
    sweep = lambda hst: tuple(F4[hst[(t - 1) % 4]] for t in range(4))
    viol = lambda hst: sum(F4[hst[t]] != hst[(t + 1) % 4] for t in range(4))
    mono = ident = True; last = 0; positive = 0; falls = 0; cons0 = 0
    for hst in itertools.product(range(16), repeat=4):
        seq = [viol(hst)]; g_ = hst
        for _ in range(6):
            g_ = sweep(g_); seq.append(viol(g_))
        mono &= all(seq[k + 1] <= seq[k] for k in range(6))
        last = max(last, max([k for k in range(1, 7) if seq[k] != seq[k - 1]], default=0))
        positive += int(seq[-1] > 0); falls += int(seq[-1] < seq[0]); cons0 += int(seq[0] == 0)
        ident &= sweep(sweep(sweep(sweep(hst)))) == tuple(F4[F4[F4[F4[x]]]] for x in hst)
    img, transient = set(range(16)), 0
    while {F4[x] for x in img} != img: img = {F4[x] for x in img}; transient += 1
    checkbool("resweep G on the 65536 histories (Rule 110, 4 cells, T = 4): G^T = F^T slicewise", ident)
    checkbool("resweep: the violated-link count never increases and stops changing after at most two sweeps", mono and last == 2)
    checkbool("Rule 110 on 4 cells: the eventual image (5 states, F a bijection on it) is reached after two steps",
              transient == 2 and len(img) == 5 and len({F4[x] for x in img}) == 5)
    check("resweep: histories whose count falls (% of 65536)", 100 * falls / 65536, 45.116, 0.01)
    checkbool("resweep: consistent histories before the sweeps == 5, after them == 580", cons0 == 5 and 65536 - positive == 580)
    check("resweep: histories whose count stays positive (% of 65536)", 100 * positive / 65536, 99.115, 0.01)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    quick = '--quick' in args
    chosen = [a for a in args if a.startswith('S')] or ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11']
    seeds = (1,) if quick else (1, 2)
    emit(f"verify_p1  {datetime.date.today()}  numpy {np.__version__}  quick={quick}  sections={chosen}")
    for s in chosen:
        {'S1': S1, 'S2': S2, 'S3': lambda: S3(seeds), 'S4': lambda: S4(seeds),
         'S5': S5, 'S6': S6, 'S7': S7, 'S8': S8, 'S9': S9, 'S10': S10, 'S11': S11}[s]()
    emit(f"\nSUMMARY  PASS {TALLY['PASS']}  WARN {TALLY['WARN']}  FAIL {TALLY['FAIL']}")
    emit("Output also written to verify_p1_results.txt.")
    OUT.close()
