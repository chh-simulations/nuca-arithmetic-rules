#!/usr/bin/env python3
"""
verify_p2.py -- verification script for the paper

    P. Stoychev, "From automata to operator algebras: possible,
    impossible, and open".

Sections 1 to 10 were written independently of the author's code, from
the statements of the paper and standard definitions.  Sections 11 to
14, and the extensions of Sections 8, 11 and 12 (the four one-particle
copies and the algebra they generate, the reversible embedding of the
rule tables, the range of the excitability identity), were written
together with the statements they check, in the revisions of 7 and 13
November 2026.  The only reused code is the pair of step functions of
Wetterich's automaton taken verbatim from wetterich_mx3.py (Section 8
below, marked as such); the checks built on them are new.

Requirements: Python 3 and numpy only.  Deterministic (fixed seeds).
Runs in well under a minute on two cores.

Output format.  Every check prints exactly one line
    PASS <label>: <got> vs <expected>
or
    FAIL <label>: <got> vs <expected>
Lines starting with four spaces are informational (values printed for
the reader, or context for a referee).  The script ends with
    SUMMARY: <n> PASS, <m> FAIL
and writes the same output to verify_p2_results.txt in the working
directory.

The expected values are the paper's numbers.  A disagreement with the
paper is reported as FAIL and is never hidden or tuned away; the paper
is to be corrected, not the check.

Section titles name the section of the paper that each block regenerates.
"""
import sys
import time
import traceback
from fractions import Fraction
from itertools import permutations, product

import numpy as np

# ---------------------------------------------------------------------------
# Bookkeeping: every line printed is also collected for the results file.
# ---------------------------------------------------------------------------
LINES = []
COUNT = {"PASS": 0, "FAIL": 0}


def out(s=""):
    print(s)
    LINES.append(s)


def fmt(v):
    """Compact rendering of a value for the got/expected fields."""
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if v == 0.0 or 1e-3 <= abs(v) < 1e6:
            return "%.6g" % v
        return "%.3e" % v
    if isinstance(v, complex):
        return "(%.16g%+.3gj)" % (v.real, v.imag)
    if isinstance(v, np.ndarray):
        return fmt(v.tolist())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def check(label, got, expected, ok):
    """One PASS/FAIL line.  `ok` is decided by the caller so that the
    comparison rule is visible at the call site."""
    tag = "PASS" if ok else "FAIL"
    COUNT[tag] += 1
    out("%s %s: %s vs %s" % (tag, label, fmt(got), fmt(expected)))
    return bool(ok)


def info(s):
    out("    " + s)


def section(title):
    out("")
    out("=" * 78)
    out(title)
    out("=" * 78)


# ---------------------------------------------------------------------------
# Shared tools
# ---------------------------------------------------------------------------
def popcount(a, nbits):
    """Number of 1 bits among the low `nbits` bits of the integer array a."""
    a = np.asarray(a)
    c = np.zeros_like(a)
    for i in range(nbits):
        c = c + ((a >> i) & 1)
    return c


def bitrev(m, K):
    """Bit reversal R_K of K-bit integers (array or scalar)."""
    m = np.asarray(m)
    r = np.zeros_like(m)
    for i in range(K):
        r = r | (((m >> i) & 1) << (K - 1 - i))
    return r


def expm(A, terms=40):
    """Matrix exponential (numpy has none): scaling and squaring of the
    Taylor series.  Adequate here: 12 x 12 matrices of norm < 30."""
    A = np.asarray(A, dtype=complex)
    nrm = np.linalg.norm(A, 1)
    s = int(np.ceil(np.log2(nrm))) + 1 if nrm > 0.5 else 0
    B = A / (2.0 ** s)
    E = np.eye(A.shape[0], dtype=complex)
    term = E.copy()
    for k in range(1, terms):
        term = term @ B / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def gf2_rank(M):
    """Rank over GF(2) of a 0/1 integer matrix (Gaussian elimination)."""
    M = (np.array(M, dtype=np.int64) & 1).copy()
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def is_affine_gf2(table, nbits_in):
    """Is the map x -> table[x] on {0..2^nbits_in - 1} affine over GF(2),
    i.e. table[x] = A x XOR b (bitwise, any output width)?  Directly from
    the definition: b = f(0), column i of A is f(e_i) XOR b, and f must
    then equal b XOR (XOR of the columns selected by the bits of x)."""
    table = np.asarray(table, dtype=np.int64)
    xs = np.arange(1 << nbits_in)
    b = table[0]
    cols = [table[1 << i] ^ b for i in range(nbits_in)]
    pred = np.full_like(xs, b)
    for i in range(nbits_in):
        pred = pred ^ np.where((xs >> i) & 1, cols[i], 0)
    return bool(np.array_equal(pred, table))


# ---------------------------------------------------------------------------
# Rule 110 and the transfer operator (Sections 1, 3, 6)
# ---------------------------------------------------------------------------
RULE = 110


def eca_step(x, N, rule=RULE):
    """One step of the elementary cellular automaton with Wolfram number
    `rule` on a periodic ring of N cells, applied to every state in the
    integer array x (bit i of the integer = cell i).  Wolfram's convention:
    the neighbourhood (left, centre, right) = (l, c, r) is the 3-bit number
    4l + 2c + r and the new cell is that bit of the rule number."""
    x = np.asarray(x, dtype=np.int64)
    y = np.zeros_like(x)
    for i in range(N):
        l = (x >> ((i - 1) % N)) & 1
        c = (x >> i) & 1
        r = (x >> ((i + 1) % N)) & 1
        y |= ((rule >> (4 * l + 2 * c + r)) & 1) << i
    return y


def eventual_image(f):
    """E = intersection over k >= 1 of f^k(X), for f given as an array
    (f[x] = image of x).  Since f(X) >= f^2(X) >= ... the chain is
    followed until it stops shrinking.  Returns (E as sorted array,
    the smallest j with f^j(X) = E)."""
    n = len(f)
    cur = np.ones(n, dtype=bool)          # f^0(X) = X
    j = 0
    while True:
        nxt = np.zeros(n, dtype=bool)
        nxt[f[cur]] = True                # f(current set)
        if np.array_equal(nxt, cur):
            return np.flatnonzero(cur), j
        cur = nxt
        j += 1


def depth_to_core(f, E):
    """depth[x] = least k with f^k(x) in E (0 on E)."""
    n = len(f)
    inE = np.zeros(n, dtype=bool)
    inE[E] = True
    depth = np.full(n, -1, dtype=np.int64)
    depth[inE] = 0
    y = np.arange(n)
    k = 0
    while (depth < 0).any():
        y = f[y]
        k += 1
        newly = (depth < 0) & inE[y]
        depth[newly] = k
    return depth


def cycles_of_bijection(f, E):
    """Cycle decomposition of f restricted to E (f|E must be a bijection;
    every orbit is checked to close)."""
    seen = set()
    cycles = []
    for x in E.tolist():
        if x in seen:
            continue
        cyc = []
        y = x
        while y not in seen:
            seen.add(y)
            cyc.append(y)
            y = int(f[y])
        assert y == x, "orbit did not close: f|E is not a bijection"
        cycles.append(cyc)
    return cycles


def transfer_matrix(f):
    """S[f(x), x] = 1 (pushforward of point masses)."""
    n = len(f)
    S = np.zeros((n, n))
    S[f, np.arange(n)] = 1.0
    return S


def rule110_data(N):
    X = np.arange(1 << N)
    f = eca_step(X, N)
    E, stab = eventual_image(f)
    return X, f, E, stab


# ===========================================================================
# 1. Theorem 1 instantiated (Sec. 3.1)
# ===========================================================================
def section_1():
    section("1. Theorem 1 (Sec. 3.1): finite core of Rule 110 at N = 8, 10, 12")
    table = "".join(str((RULE >> b) & 1) for b in range(7, -1, -1))
    check("1 rule table of Wolfram number 110 for neighbourhoods 111..000",
          table, "01101110", table == "01101110")
    expected_pct = {8: 17.6, 10: 12.8, 12: 2.3}

    for N in (8, 10, 12):
        X, f, E, stab = rule110_data(N)
        nE = len(E)
        # (a) f|E is a bijection: f(E) = E and f injective on E.
        fE = set(f[E].tolist())
        bij = (fE == set(E.tolist())) and (len(f[E]) == len(np.unique(f[E])))
        check("1a N=%d f restricted to E is a bijection" % N, bij, True, bij)
        cycles = cycles_of_bijection(f, E)
        lengths = sorted(len(c) for c in cycles)
        depth = depth_to_core(f, E)
        k = int(depth.max())
        info("N=%d: |E| = %d, %d cycles, lengths %s; transient depth (nilpotency index) = %d, |X\\E| = %d"
             % (N, nE, len(cycles), lengths, k, (1 << N) - nE))
        check("1 N=%d f^j(X) stabilises at E exactly at j = transient depth" % N,
              stab, k, stab == k)

        # (c) core fraction, rounded to one decimal as in the paper.
        pct = 100.0 * nE / (1 << N)
        check("1c N=%d core fraction |E|/2^N (|E| = %d)" % (N, nE),
              "%.3f%%" % pct, "%.1f%%" % expected_pct[N],
              round(pct, 1) == expected_pct[N])

        # Structural claim of the proof, at the map level (exact, all N):
        # with E first S is block-triangular iff f(E) is contained in E,
        # and T^k = 0 iff every transient point enters E within k steps
        # (f(E) in E means T^k[y, x] = 1 iff f^k(x) = y with x, y outside E).
        inE = np.zeros(1 << N, dtype=bool)
        inE[E] = True
        f_maps_E_into_E = bool(inE[f[E]].all())
        check("1 N=%d f(E) subset of E (lower-left block of S vanishes, E first)" % N,
              f_maps_E_into_E, True, f_maps_E_into_E)
        # T^k = 0 and T^(k-1) != 0 via the partial map g = f on transients,
        # absorbing (-1) once E is reached.
        trans = np.flatnonzero(~inE)
        g = np.where(inE[f], -1, f)        # g[x] = f[x] if f[x] transient else -1
        y = trans.copy()
        alive_after = []
        for step in range(k):
            y = np.where(y >= 0, g[np.maximum(y, 0)], -1)
            alive_after.append(int((y >= 0).sum()))
        tk_zero = alive_after[-1] == 0
        tk1_nonzero = (alive_after[-2] > 0) if k >= 2 else True
        check("1 N=%d transient block nilpotent: T^k = 0 at k = %d <= |X\\E| = %d, T^(k-1) != 0" % (N, k, len(trans)),
              (tk_zero, tk1_nonzero, k <= len(trans)), (True, True, True),
              tk_zero and tk1_nonzero and k <= len(trans))

        if N <= 10:
            # Same claim with the dense matrix: reorder with E first,
            # S = [[P, B], [0, T]], P a permutation matrix, T^k = 0.
            S = transfer_matrix(f)
            order = np.concatenate([E, trans])
            S2 = S[np.ix_(order, order)]
            P = S2[:nE, :nE]
            T = S2[nE:, nE:]
            LL = S2[nE:, :nE]
            ll_zero = not LL.any()
            P_perm = bool((P.sum(0) == 1).all() and (P.sum(1) == 1).all())
            Tk = np.linalg.matrix_power(T, k)
            Tk1 = np.linalg.matrix_power(T, k - 1)
            dense_ok = ll_zero and P_perm and (not Tk.any()) and bool(Tk1.any())
            check("1 N=%d dense check: S block-triangular with E first, P a permutation, T^%d = 0, T^%d != 0" % (N, k, k - 1),
                  dense_ok, True, dense_ok)

        if N == 8:
            section_1b(f, E, lengths, k)


def section_1b(f, E, lengths, k):
    """(b) N = 8: the spectrum of the 256 x 256 transfer matrix."""
    nE = len(E)
    S = transfer_matrix(f)
    check("1b N=8 core size |E|", nE, 45, nE == 45)
    check("1b N=8 eigenvalue 0 has multiplicity |X\\E| exactly (nilpotency, T^%d = 0)" % k,
          256 - nE, 211, 256 - nE == 211)
    lam = np.linalg.eigvals(S)
    big = lam[np.abs(lam) > 0.5]
    small = lam[np.abs(lam) <= 0.5]
    check("1b N=8 number of computed eigenvalues on the unit circle (|lambda| > 1/2)",
          len(big), 45, len(big) == 45)
    # Expected multiset: the L-th roots of unity for every cycle of length L.
    roots = np.concatenate([np.exp(2j * np.pi * np.arange(L) / L) for L in lengths])
    dev = np.abs(big[:, None] - roots[None, :]).min(axis=1)
    maxdev = float(dev.max())
    iw = int(dev.argmax())
    info("max deviation %.4e at the computed eigenvalue %s (nearest root %s)"
         % (maxdev, fmt(complex(big[iw])), fmt(complex(roots[np.abs(big[iw] - roots).argmin()]))))
    # The paper (Sec. 3.1) states the bound 2e-15; the computed deviation is
    # 1.3e-15 on this machine and depends on the solver path (the complex
    # LAPACK path gives 2.2e-15 on the permutation block), which is why the
    # paper's bound is not the two-figure value.
    check("1b N=8 max deviation of on-circle eigenvalues from the cycle roots of unity (paper: 2e-15)",
          "%.4e" % maxdev, "<= 2e-15", maxdev <= 2e-15)
    # Multiset equality: assign each computed eigenvalue to the nearest
    # distinct root; the counts must reproduce the multiplicities.
    keys = sorted(set(Fraction(j, L) for L in lengths for j in range(L)))
    distinct = np.array([np.exp(2j * np.pi * float(q)) for q in keys])
    mult_exp = [sum(1 for L in lengths for j in range(L) if Fraction(j, L) == q) for q in keys]
    assign = np.abs(big[:, None] - distinct[None, :]).argmin(axis=1)
    mult_got = [int((assign == i).sum()) for i in range(len(keys))]
    info("distinct roots exp(2 pi i q), q = %s" % [str(q) for q in keys])
    check("1b N=8 multiplicities of the on-circle eigenvalues = multiplicities of the cycle roots",
          mult_got, mult_exp, mult_got == mult_exp)
    check("1b N=8 number of remaining eigenvalues", len(small), 211, len(small) == 211)
    maxsmall = float(np.abs(small).max()) if len(small) else 0.0
    info("remaining eigenvalues: max modulus %.3e, %d of them below 1e-12, %d below 1e-6"
         % (maxsmall, int((np.abs(small) < 1e-12).sum()), int((np.abs(small) < 1e-6).sum())))
    check("1b N=8 max modulus of the 211 remaining eigenvalues as the solver returns them (<= 1e-12; the fact is T^7 = 0)",
          maxsmall, "<= 1e-12", maxsmall <= 1e-12)
    # Diagnostics for the referee (not claims of the paper).
    # (i) The on-circle accuracy is that of the eigenvalue solver on the
    #     permutation block P alone (a normal matrix): the attached
    #     non-normal transient does not degrade it.
    order = np.concatenate([E, np.setdiff1d(np.arange(256), E)])
    P = S[np.ix_(order, order)][:nE, :nE]
    lamP = np.linalg.eigvals(P)
    devP = float(np.abs(lamP[:, None] - roots[None, :]).min(axis=1).max())
    check("1b diagnostic: eigenvalues of the permutation block P alone vs cycle roots (max deviation, machine level)",
          "%.4e" % devP, "<= 1e-14", devP <= 1e-14)
    # (ii) Why the transient eigenvalues come out exactly zero: in the state
    #     basis S has zero rows (Garden-of-Eden states), and LAPACK's
    #     balancing isolates them, then the states whose only preimages they
    #     were, and so on through the whole transient forest.  After a
    #     generic orthogonal change of basis the same 211 eigenvalues spread
    #     to about eps^(1/k), k = nilpotency index -- the usual behaviour of
    #     a non-normal matrix.  The exact statement is the nilpotency above.
    n_goe = int((S.sum(axis=1) == 0).sum())
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.standard_normal((256, 256)))
    lamQ = np.linalg.eigvals(Q @ S @ Q.T)
    smallQ = np.abs(lamQ[np.abs(lamQ) <= 0.5])
    info("diagnostic: %d Garden-of-Eden states (zero rows of S); after a random orthogonal similarity the"
         % n_goe)
    info("            %d transient eigenvalues spread to max modulus %.2e, compare eps^(1/%d) = %.2e"
         % (len(smallQ), smallQ.max(), k, np.finfo(float).eps ** (1.0 / k)))


# ===========================================================================
# 2. Theorem 2 (Sec. 3.2): clock and shift
# ===========================================================================
def section_2():
    section("2. Theorem 2 (Sec. 3.2): clock and shift generate M_n(C), n = 2..8")
    for n in range(2, 9):
        S = np.roll(np.eye(n), 1, axis=0)          # S e_k = e_{k+1 mod n}
        w = np.exp(2j * np.pi / n)
        C = np.diag(w ** np.arange(n))              # diag(1, w, ..., w^{n-1})
        weyl = float(np.abs(C @ S - w * S @ C).max())
        check("2 n=%d Weyl relation CS = wSC (max |CS - wSC|)" % n, weyl, "<= 1e-12", weyl <= 1e-12)
        keys = [(a, b) for a in range(n) for b in range(n)]
        mons = [np.linalg.matrix_power(C, a) @ np.linalg.matrix_power(S, b) for a, b in keys]
        coef = np.array([M.flatten() for M in mons])            # n^2 x n^2
        rank = int(np.linalg.matrix_rank(coef))
        check("2 n=%d rank of the n^2 monomials C^a S^b" % n, rank, n * n, rank == n * n)
        G = np.array([[np.trace(A.conj().T @ B) for B in mons] for A in mons])
        gdev = float(np.abs(G - n * np.eye(n * n)).max())
        check("2 n=%d Hilbert-Schmidt Gram matrix = n * delta_(a=c) delta_(b=d) (max deviation)" % n,
              gdev, "<= 1e-10", gdev <= 1e-10)


# ===========================================================================
# 3. Natural-extension remark (Sec. 3.3)
# ===========================================================================
def section_3_norm():
    # Sec. 3.3: the transfer operator S of Rule 110 at N = 8 is not a
    # contraction on l^2(X); its operator norm is the square root of the
    # largest fibre (number of preimages of a state).
    N = 8
    _, f, _, _ = rule110_data(N)
    S = np.asarray(transfer_matrix(f), dtype=float)
    nrm = float(np.linalg.norm(S, 2))
    fib = int(np.bincount(f, minlength=1 << N).max())
    check("3 Sec. 3.3: operator norm of S (Rule 110, N = 8) = sqrt(largest fibre), fibre = %d" % fib,
          "%.6f" % nrm, "%.6f" % np.sqrt(fib), abs(nrm - np.sqrt(fib)) < 1e-9)
    check("3 Sec. 3.3: largest fibre of Rule 110 at N = 8 (paper: sqrt 10)", fib, 10, fib == 10)


def section_3():
    section("3. Natural extension (Sec. 3.3): backward chains in Rule 110, N = 8")
    X, f, E, stab = rule110_data(8)
    n = len(f)
    # has_chain[x] = x admits a backward chain of length ell:
    # x_0 <- x_1 <- ... <- x_ell with f(x_{i+1}) = x_i.  Recursively,
    # x admits a chain of length ell+1 iff x = f(y) for some y admitting a
    # chain of length ell.
    has_chain = np.ones(n, dtype=bool)       # length 0: every state
    first_equal = None
    for ell in range(1, 21):
        nxt = np.zeros(n, dtype=bool)
        nxt[f[has_chain]] = True
        has_chain = nxt
        if first_equal is None and np.array_equal(np.flatnonzero(has_chain), E):
            first_equal = ell
    A20 = np.flatnonzero(has_chain)
    check("3 states admitting a backward chain of length 20 = the core E (|E| = %d)" % len(E),
          "%d states, equal to E: %s" % (len(A20), np.array_equal(A20, E)), "45 states, equal to E",
          np.array_equal(A20, E) and len(E) == 45)
    info("the chain-admitting set first coincides with E at length %s (transient depth)" % first_equal)
    inE = np.zeros(n, dtype=bool)
    inE[E] = True
    preimages_in_E = np.array([int(((f == x) & inE).sum()) for x in E])
    check("3 every core state has exactly one preimage inside E (min, max count)",
          (int(preimages_in_E.min()), int(preimages_in_E.max())), (1, 1),
          bool((preimages_in_E == 1).all()))
    total_pre = np.array([int((f == x).sum()) for x in E])
    info("preimages counted in all of X: core states have between %d and %d preimages (extra ones are transient)"
         % (total_pre.min(), total_pre.max()))


# ===========================================================================
# 4. The Walsh system and O_2 (Sec. 3.4), resolution 256
# ===========================================================================
GRID = 256
kk = np.arange(GRID)


def rademacher(j):
    """r_j(x_k) = r(2^j x_k mod 1) on the grid x_k = k/256, with r = +1 on
    [0, 1/2) and -1 on [1/2, 1).  2^j x_k mod 1 = ((2^j k) mod 256)/256."""
    num = (kk * (1 << j)) % GRID
    return np.where(num < GRID // 2, 1, -1)


def walsh(m):
    """Walsh-Paley function W_m = prod_j r_j^{m_j} sampled on the grid."""
    w = np.ones(GRID, dtype=np.int64)
    for j in range(8):
        if (m >> j) & 1:
            w = w * rademacher(j)
    return w


def S0(g):
    """(S_0 g)(x) = g(2x mod 1): x_k -> x_{2k mod 256}."""
    return g[(2 * kk) % GRID]


def S1(g):
    """(S_1 g)(x) = r(x) g(2x mod 1)."""
    return rademacher(0) * g[(2 * kk) % GRID]


def section_4():
    section("4. Walsh system and O_2 (Sec. 3.4), dyadic grid of 256 points")
    W = np.array([walsh(m) for m in range(GRID)])
    orth = np.array_equal(W @ W.T, GRID * np.eye(GRID, dtype=np.int64))
    check("4 sanity of the construction: the 256 sampled Walsh functions are orthogonal (W W^T = 256 I)",
          orth, True, orth)
    # (a) index action of S_0, S_1 for 2m+1 < 256
    ok0 = all(np.array_equal(S0(W[m]), W[2 * m]) for m in range(128))
    ok1 = all(np.array_equal(S1(W[m]), W[2 * m + 1]) for m in range(128))
    check("4a S_0 W_m = W_{2m} for all m with 2m+1 < 256 (m = 0..127)", ok0, True, ok0)
    check("4a S_1 W_m = W_{2m+1} for all m with 2m+1 < 256 (m = 0..127)", ok1, True, ok1)
    # (b) words of length 8 applied to the constant 1
    hits = []
    reading_ok = True
    for word in product((0, 1), repeat=8):
        g = np.ones(GRID, dtype=np.int64)
        for e in reversed(word):             # S_{e_1} S_{e_2} ... S_{e_8} 1: S_{e_8} acts first
            g = S0(g) if e == 0 else S1(g)
        match = np.flatnonzero((W == g).all(axis=1))
        if len(match) != 1:
            hits.append(-1)
            continue
        m = int(match[0])
        hits.append(m)
        # the first letter applied (e_8) is the top bit: m = sum e_i 2^(i-1)
        reading_ok &= (m == sum(e << (i) for i, e in enumerate(word)))
    hits = np.array(hits)
    all_walsh = bool((hits >= 0).all())
    check("4b all 256 words of length 8 in S_0, S_1 applied to 1 are Walsh functions", all_walsh, True, all_walsh)
    once = np.array_equal(np.sort(hits), np.arange(GRID))
    check("4b every W_m, m = 0..255, is obtained exactly once", once, True, once)
    check("4b word S_{e_1}...S_{e_8} 1 = W_m with m = sum_i e_i 2^(i-1) (first applied letter = top bit)",
          reading_ok, True, reading_ok)
    # (c) Cuntz relations at the index level, K = 2..12
    ok_c = True
    for K in range(2, 13):
        dom = np.arange(1 << (K - 1))
        im = [2 * dom, 2 * dom + 1]
        inj = all(len(np.unique(v)) == len(dom) for v in im)
        disjoint = len(np.intersect1d(im[0], im[1])) == 0
        cover = np.array_equal(np.sort(np.concatenate(im)), np.arange(1 << K))
        ok_c &= inj and disjoint and cover
    check("4c index maps m -> 2m + eps injective, disjoint images, union = {0..2^K-1}, K = 2..12", ok_c, True, ok_c)
    # (d) filter values on the two half-intervals
    ones = np.ones(GRID, dtype=np.int64)
    m0, m1 = S0(ones), S1(ones)
    halves = [slice(0, GRID // 2), slice(GRID // 2, GRID)]
    const = all(len(np.unique(mm[h])) == 1 for mm in (m0, m1) for h in halves)
    F = np.array([[int(mm[h][0]) for h in halves] for mm in (m0, m1)])
    check("4d filter values of (S_0, S_1) on [0,1/2), [1/2,1) (constant on each half: %s)" % const,
          F.tolist(), [[1, 1], [1, -1]], const and F.tolist() == [[1, 1], [1, -1]])
    Had = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    had_ok = np.allclose(F, np.sqrt(2) * Had) and np.allclose(Had @ Had.T, np.eye(2))
    check("4d filter block = sqrt(2) * Hadamard block (1/sqrt2)[[1,1],[1,-1]], the latter orthogonal",
          had_ok, True, had_ok)
    # Extra (operator level, not only index level): on the finite truncation
    # V_128 -> V_256 (functions constant on dyadic intervals of length 1/128
    # resp. 1/256, inner products (1/n) sum f g), S_0 and S_1 are isometries
    # with orthogonal ranges summing to V_256: the Cuntz relations of O_2.
    A0 = np.zeros((GRID, GRID // 2))
    A1 = np.zeros((GRID, GRID // 2))
    A0[kk, kk % (GRID // 2)] = 1.0
    A1[kk, kk % (GRID // 2)] = rademacher(0)
    # adjoints for the normalised inner products: A^* = (128/256) A^T
    adj = lambda A: A.T / 2.0
    cuntz = (np.allclose(adj(A0) @ A0, np.eye(128)) and np.allclose(adj(A1) @ A1, np.eye(128))
             and np.allclose(adj(A0) @ A1, 0) and np.allclose(A0 @ adj(A0) + A1 @ adj(A1), np.eye(256)))
    check("4 extra: operator-level Cuntz relations S_e^* S_e = 1, S_0^* S_1 = 0, sum S_e S_e^* = 1 on V_128 -> V_256",
          cuntz, True, cuntz)


# ===========================================================================
# 5. Proposition 3 (the reversal identity), K = 2..12
# ===========================================================================
def section_5():
    section("5. Proposition 3 (Sec. 3.4): bit reversal intertwines S_eps and P_eps, K = 2..12")
    ok_all = True
    ok_inv = True
    ok_P = True
    for K in range(2, 13):
        m = np.arange(1 << (K - 1))
        full = np.arange(1 << K)
        ok_inv &= np.array_equal(bitrev(bitrev(full, K), K), full)
        ims = []
        for eps in (0, 1):
            lhs = bitrev(2 * bitrev(m, K - 1) + eps, K)      # R_K o S_eps o R_{K-1}
            rhs = m + eps * (1 << (K - 1))                   # P_eps
            ok_all &= np.array_equal(lhs, rhs)
            ims.append(rhs)
        ok_P &= (len(np.intersect1d(ims[0], ims[1])) == 0
                 and np.array_equal(np.sort(np.concatenate(ims)), full))
    check("5 R_K is an involution on K-bit integers, K = 2..12", ok_inv, True, ok_inv)
    check("5 R_K(2 R_{K-1}(m) + eps) = m + eps 2^{K-1} for all (K-1)-bit m, eps in {0,1}, K = 2..12",
          ok_all, True, ok_all)
    check("5 P_eps: m -> m + eps 2^{K-1} have disjoint images covering {0..2^K-1}, K = 2..12", ok_P, True, ok_P)


# ===========================================================================
# 6. Proposition 4 (Sec. 3.5): time-periodic closure
# ===========================================================================
def section_6():
    section("6. Proposition 4 (Sec. 3.5): histories on a time circle, Rule 110, N = 8, T = 1..16")
    X, f, E, stab = rule110_data(8)
    S = transfer_matrix(f)
    lengths = [len(c) for c in cycles_of_bijection(f, E)]
    tr_vals, fix_vals, cyc_vals = [], [], []
    for T in range(1, 17):
        tr_vals.append(int(round(np.trace(np.linalg.matrix_power(S, T)))))   # (i)
        y = X.copy()
        for _ in range(T):
            y = f[y]
        fix_vals.append(int((y == X).sum()))                                 # (ii)
        cyc_vals.append(sum(L for L in lengths if T % L == 0))               # (iii)
    info("cycle lengths of f|E: %s" % sorted(lengths))
    info("T = 1..16, (i)   tr(S^T)            : %s" % tr_vals)
    info("T = 1..16, (ii)  #{x : f^T(x) = x}  : %s" % fix_vals)
    info("T = 1..16, (iii) sum_{L_c | T} L_c  : %s" % cyc_vals)
    agree = tr_vals == fix_vals == cyc_vals
    check("6 the three counts agree for all T = 1..16", agree, True, agree)
    check("6 tr(S^T), T = 1..16", tr_vals, cyc_vals, tr_vals == cyc_vals)


# ===========================================================================
# 7. Proposition 5 (Sec. 4.5): the branches of the logarithm do not compose, (L1, L2) = (3, 4)
# ===========================================================================
def shift_eigenbasis(L):
    """Eigenvectors of the cyclic shift S (S e_k = e_{k+1}) on C^L:
    v_a[k] = exp(-2 pi i a k/L)/sqrt(L) satisfies S v_a = exp(2 pi i a/L) v_a,
    eigenphase phi_a = 2 pi a/L in [0, 2 pi) -- the principal branch."""
    k = np.arange(L)
    return np.exp(-2j * np.pi * np.outer(k, np.arange(L)) / L) / np.sqrt(L)


def section_7():
    section("7. Proposition 5 (Sec. 4.5): principal-branch generators, (L1, L2) = (3, 4)")
    L1, L2 = 3, 4
    U1 = np.roll(np.eye(L1), 1, axis=0)
    U2 = np.roll(np.eye(L2), 1, axis=0)
    U = np.kron(U1, U2)
    V1, V2 = shift_eigenbasis(L1), shift_eigenbasis(L2)
    phi = 2 * np.pi * np.arange(L1) / L1
    psi = 2 * np.pi * np.arange(L2) / L2
    eig_ok = (np.allclose(U1 @ V1, V1 * np.exp(1j * phi)) and np.allclose(U2 @ V2, V2 * np.exp(1j * psi))
              and np.allclose(V1.conj().T @ V1, np.eye(L1)) and np.allclose(V2.conj().T @ V2, np.eye(L2)))
    check("7 Fourier modes are an orthonormal eigenbasis of U_1, U_2 with phases 2 pi a/L in [0, 2 pi)",
          eig_ok, True, eig_ok)
    # principal-branch generators of the factors and the additive candidate
    H1 = (V1 * phi) @ V1.conj().T
    H2 = (V2 * psi) @ V2.conj().T
    H_add = np.kron(H1, np.eye(L2)) + np.kron(np.eye(L1), H2)
    # joint eigenbasis and the principal-branch generator of U itself.
    # Exact bookkeeping: phi_a + psi_b = 2 pi (a L2 + b L1)/(L1 L2); the
    # principal phase reduces the numerator mod L1 L2; the state wraps iff
    # the numerator is >= L1 L2.
    W = np.kron(V1, V2)                                    # column a*L2 + b = v_a (x) u_b
    pairs = [(a, b) for a in range(L1) for b in range(L2)]
    num = np.array([a * L2 + b * L1 for a, b in pairs])
    wrapped = num >= L1 * L2
    theta_pr = 2 * np.pi * (num % (L1 * L2)) / (L1 * L2)
    H_pr = (W * theta_pr) @ W.conj().T
    joint_ok = np.allclose(U @ W, W * np.exp(1j * theta_pr))
    check("7 v_a (x) u_b are eigenvectors of U with eigenphase (phi_a + psi_b) mod 2 pi", joint_ok, True, joint_ok)
    nwrap = int(wrapped.sum())
    info("wrapped joint states (a, b): %s" % [pairs[i] for i in np.flatnonzero(wrapped)])
    check("7 number of joint eigenstates whose phase sum wraps", nwrap, 3, nwrap == 3)
    # The difference, in the joint eigenbasis: 2 pi on the wrapped states, 0 elsewhere.
    D = H_add - H_pr
    Dj = W.conj().T @ D @ W
    diag_exp = 2 * np.pi * wrapped
    diff_ok = np.allclose(Dj, np.diag(diag_exp), atol=1e-12)
    check("7 H_add - H_pr is diagonal in the joint eigenbasis with entries 2 pi * [wrapped]", diff_ok, True, diff_ok)
    ev = np.sort(np.linalg.eigvalsh(D))
    ev_exp = np.sort(diag_exp)
    check("7 eigenvalues of H_add - H_pr: three equal to 2 pi, nine equal to 0 (max deviation)",
          float(np.abs(ev - ev_exp).max()), "<= 1e-12", bool(np.abs(ev - ev_exp).max() <= 1e-12))
    check("7 H_pr(U) != H_pr(U_1)(x)1 + 1(x)H_pr(U_2) (max |difference| = 2 pi)",
          float(np.abs(D).max()) > 1e-6, True, float(np.abs(D).max()) > 1e-6)
    # Both exponentiate to U (exp computed by scaling-and-squaring Taylor, no eig)
    e_pr = float(np.abs(expm(1j * H_pr) - U).max())
    e_add = float(np.abs(expm(1j * H_add) - U).max())
    check("7 exp(i H_pr) = U to machine precision (max deviation)", e_pr, "<= 1e-12", e_pr <= 1e-12)
    check("7 exp(i H_add) = U to machine precision (max deviation)", e_add, "<= 1e-12", e_add <= 1e-12)
    # spectra
    sp_pr = np.sort(np.linalg.eigvalsh(H_pr))
    sp_exp = 2 * np.pi * np.arange(12) / 12
    check("7 spectrum of H_pr(U) = {2 pi k/12 : k = 0..11} in [0, 2 pi) (max deviation)",
          float(np.abs(sp_pr - sp_exp).max()), "<= 1e-12", bool(np.abs(sp_pr - sp_exp).max() <= 1e-12))
    sp_add = np.sort(np.linalg.eigvalsh(H_add))
    info("spectrum of H_add / (2 pi/12): %s (three values >= 12 are the wrapped ones)"
         % [int(round(v / (2 * np.pi / 12))) for v in sp_add])
    # Cross-check: H_pr from a numerical eigendecomposition of U, phases
    # taken in [0, 2 pi) with values within 1e-9 of 2 pi snapped to 0 (the
    # eigenvalue 1 of U sits on the branch cut; without the snap a rounding
    # error of 1e-17 in Im(lambda) would move its phase from 0 to 2 pi).
    lam, V = np.linalg.eig(U)
    ph = np.angle(lam) % (2 * np.pi)
    ph[ph > 2 * np.pi - 1e-9] = 0.0
    H_num = V @ np.diag(ph) @ np.linalg.inv(V)
    dnum = float(np.abs(H_num - H_pr).max())
    check("7 cross-check: H_pr from numpy eig of U (snapped principal phases) agrees (max deviation)",
          dnum, "<= 1e-10", dnum <= 1e-10)
    # Proof's inequality and non-emptiness of the wrapped set, 2 <= L1, L2 <= 12
    ineq_ok = True
    nonempty = True
    counts = {}
    for a_ in range(2, 13):
        for b_ in range(2, 13):
            # (L1-1)/L1 + (L2-1)/L2 >= 1  <=>  (L1-1) L2 + (L2-1) L1 >= L1 L2  (integers)
            ineq_ok &= (a_ - 1) * b_ + (b_ - 1) * a_ >= a_ * b_
            cnt = sum(1 for a in range(a_) for b in range(b_) if a * b_ + b * a_ >= a_ * b_)
            counts[(a_, b_)] = cnt
            nonempty &= cnt >= 1
    check("7 (L1-1)/L1 + (L2-1)/L2 >= 1 for all 2 <= L1, L2 <= 12", ineq_ok, True, ineq_ok)
    # The additive operator H_pr(U_1) (x) 1 + 1 (x) H_pr(U_2) is positive,
    # additive by construction, and generates U exactly, for every pair of
    # lengths: the proposition's content is the branch mismatch, not any
    # conflict between positivity and additivity.
    pos_ok = True
    gen_ok = True
    for a_ in range(2, 13):
        for b_ in range(2, 13):
            ph1 = 2 * np.pi * np.arange(a_) / a_
            ph2 = 2 * np.pi * np.arange(b_) / b_
            s_add = (ph1[:, None] + ph2[None, :]).ravel()          # spectrum of H_add
            pos_ok &= bool((s_add >= -1e-12).all())
            # exp(i H_add) and U share the joint eigenbasis; compare eigenvalues
            gen_ok &= bool(np.allclose(np.exp(1j * s_add), np.exp(1j * ((ph1[:, None] + ph2[None, :]) % (2 * np.pi))).ravel()))
    check("7 H_add = H_pr(U_1)(x)1 + 1(x)H_pr(U_2) is positive for all 2 <= L1, L2 <= 12", pos_ok, True, pos_ok)
    check("7 exp(i H_add) = U for all 2 <= L1, L2 <= 12 (on the joint eigenbasis)", gen_ok, True, gen_ok)
    # Branch obstruction: no integer shifts k_a, l_b make phi_a + psi_b +
    # 2 pi (k_a + l_b) equal to the principal phase (phi_a + psi_b) mod 2 pi
    # on every joint eigenstate.  Brute force at (3, 4) over shifts in
    # {-2..2}, and the proof's premise (row a = 0 and column b = 0 never
    # wrap) for all pairs.
    import itertools
    wrap34 = np.array([[1 if a * 4 + b * 3 >= 12 else 0 for b in range(4)] for a in range(3)])
    found = False
    for ks in itertools.product(range(-2, 3), repeat=3):
        for ls in itertools.product(range(-2, 3), repeat=4):
            if all(ks[a] + ls[b] == -wrap34[a, b] for a in range(3) for b in range(4)):
                found = True
    check("7 no choice of integer branch shifts for the two factors gives H_pr(U) at (3, 4) (shifts in -2..2)",
          found, False, not found)
    rowcol_ok = all(all(a * b_ + 0 * a_ < a_ * b_ for a in range(a_)) and all(0 * b_ + b * a_ < a_ * b_ for b in range(b_))
                    for a_ in range(2, 13) for b_ in range(2, 13))
    check("7 the row a = 0 and the column b = 0 never wrap, all 2 <= L1, L2 <= 12 (premise of the branch argument)",
          rowcol_ok, True, rowcol_ok)
    check("7 wrapped set non-empty for all 2 <= L1, L2 <= 12 (min count)", min(counts.values()), ">= 1", nonempty)
    check("7 wrapped count at (L1, L2) = (3, 4) from the general formula", counts[(3, 4)], 3, counts[(3, 4)] == 3)
    info("wrapped counts, rows L1 = 2..12, columns L2 = 2..12:")
    for a_ in range(2, 13):
        info("L1=%2d: %s" % (a_, [counts[(a_, b_)] for b_ in range(2, 13)]))


# ===========================================================================
# 8. Wetterich's automaton (Sec. 5), three sites, 4096 states
# ===========================================================================
# --- reproduced verbatim from wetterich_mx3.py (the step definitions) -----
Mx = 3
NB = 4 * Mx
NS = 1 << NB


def get(s, x, g):
    return (s >> (4 * x + g)) & 1


def setb(s, x, g, v):
    m = 1 << (4 * x + g)
    return (s | m) if v else (s & ~m)


def step_free(s, mx=Mx):
    out_ = 0
    for x in range(mx):
        for g in (0, 1):
            if get(s, x, g):
                out_ = setb(out_, (x + 1) % mx, g, 1)
        for g in (2, 3):
            if get(s, x, g):
                out_ = setb(out_, (x - 1) % mx, g, 1)
    return out_


def step_int(s, mx=Mx):
    out_ = s
    for x in range(mx):
        occ = tuple(get(s, x, g) for g in range(4))
        if occ == (1, 0, 0, 1):
            new = (0, 1, 1, 0)
        elif occ == (0, 1, 1, 0):
            new = (1, 0, 0, 1)
        elif occ == (1, 0, 1, 0):
            new = (0, 1, 0, 1)
        elif occ == (0, 1, 0, 1):
            new = (1, 0, 1, 0)
        else:
            continue
        for g in range(4):
            out_ = setb(out_, x, g, new[g])
    return out_
# --- end of reproduced code -------------------------------------------------


def section_8():
    section("8. Wetterich's automaton (Sec. 5): three sites, %d states" % NS)
    check("8 number of states at three sites (four bits per site)", NS, 4096, NS == 4096)
    states = np.arange(NS)
    F = np.array([step_free(s) for s in range(NS)])
    I_ = np.array([step_int(s) for s in range(NS)])
    S = I_[F]                                   # combined step: free, then interaction
    is_perm = lambda v: np.array_equal(np.sort(v), states)
    check("8 free step is a permutation", is_perm(F), True, is_perm(F))
    check("8 interaction step is a permutation", is_perm(I_), True, is_perm(I_))
    inv = np.array_equal(I_[I_], states)
    check("8 interaction step is an involution", inv, True, inv)
    check("8 combined step S = S_int o S_free is a permutation", is_perm(S), True, is_perm(S))
    NR = np.array([sum(get(s, x, g) for x in range(Mx) for g in (0, 1)) for s in range(NS)])
    NL = np.array([sum(get(s, x, g) for x in range(Mx) for g in (2, 3)) for s in range(NS)])
    cr = np.array_equal(NR[S], NR)
    cl = np.array_equal(NL[S], NL)
    check("8 number of right-movers conserved", cr, True, cr)
    check("8 number of left-movers conserved", cl, True, cl)
    # wave function q = sqrt(p) transported by the permutation matrix
    rng = np.random.default_rng(0)
    q = rng.normal(size=NS)
    q /= np.linalg.norm(q)
    Smat = transfer_matrix(S)
    q2 = Smat @ q
    nrm = float(np.linalg.norm(q2))
    check("8 norm of a unit vector preserved by the step (|S q|)", nrm, 1.0, abs(nrm - 1.0) <= 1e-12)
    transported = bool(np.allclose(q2[S] ** 2, q ** 2, atol=1e-15))
    check("8 p = q^2 transported to the image configuration: (Sq)[S(s)]^2 = q[s]^2 for all s",
          transported, True, transported)
    n_op = np.diag(np.array([float(get(s, 0, 0)) for s in range(NS)]))
    comm = Smat @ n_op - n_op @ Smat
    cmax = float(np.abs(comm).max())
    check("8 max |[S, n_R1(0)]|", cmax, 1.0, cmax == 1.0)
    # one-particle sector: one red right-mover, nothing else
    one = [1 << (4 * x + 0) for x in range(Mx)]
    invariant = all(int(S[c]) in one for c in one)
    sub = np.array([[1.0 if S[c] == r else 0.0 for c in one] for r in one])
    shift3 = np.roll(np.eye(Mx), 1, axis=0)
    check("8 one-particle sector is invariant under S", invariant, True, invariant)
    check("8 one-particle sector of S = 3 x 3 cyclic shift", np.array_equal(sub, shift3), True, np.array_equal(sub, shift3))
    w = np.exp(2j * np.pi / Mx)
    Ck = np.diag(w ** np.arange(Mx))
    span = np.array([(np.linalg.matrix_power(Ck, a) @ np.linalg.matrix_power(sub, b)).flatten()
                     for a in range(Mx) for b in range(Mx)])
    dim = int(np.linalg.matrix_rank(span))
    check("8 clock and shift span M_3(C): dimension of span{C^a S^b}", dim, 9, dim == 9)
    weyl = bool(np.allclose(Ck @ sub, w * sub @ Ck))
    check("8 CS = wSC in the one-particle sector", weyl, True, weyl)
    # all four copies: right-movers shift forward (CS = w SC), left-movers backward (CS = w^-1 SC); each pair spans M_3(C)
    ok4 = True
    dims = []
    for g in range(4):
        st = [1 << (4 * x + g) for x in range(Mx)]
        inv_g = all(int(S[c]) in st for c in st)
        sub_g = np.array([[1.0 if S[c] == r_ else 0.0 for c in st] for r_ in st])
        fwd = np.array_equal(sub_g, shift3)
        bwd = np.array_equal(sub_g, shift3.T)
        ok4 &= inv_g and ((fwd and g in (0, 1)) or (bwd and g in (2, 3)))
        ww = w if g in (0, 1) else 1 / w
        ok4 &= bool(np.allclose(Ck @ sub_g, ww * sub_g @ Ck))
        span_g = np.array([(np.linalg.matrix_power(Ck, a) @ np.linalg.matrix_power(sub_g, b)).flatten()
                           for a in range(Mx) for b in range(Mx)])
        dims.append(int(np.linalg.matrix_rank(span_g)))
    check("8 the four one-particle copies: forward shift with CS = wSC (right-movers), backward with CS = w^-1 SC (left-movers)", ok4, True, ok4)
    check("8 clock and shift span M_3(C) in each of the four copies", dims, [9, 9, 9, 9], dims == [9, 9, 9, 9])
    # the algebra generated on the whole one-particle sector (12 states) by the step and the twelve occupation numbers
    st12 = [1 << (4 * x + g) for g in range(4) for x in range(Mx)]
    idx = {c: i for i, c in enumerate(st12)}
    S12 = np.zeros((12, 12))
    for c in st12:
        S12[idx[int(S[c])], idx[c]] = 1.0
    gens = [S12] + [np.diag([1.0 if i == j else 0.0 for i in range(12)]) for j in range(12)]
    basis = [np.eye(12)]
    rank = 1
    while True:
        new = basis + [a @ g_ for a in basis for g_ in gens]
        M_ = np.array([m.flatten() for m in new])
        u_, sv, vt = np.linalg.svd(M_, full_matrices=False)
        r_new = int((sv > 1e-9).sum())
        basis = [vt[i].reshape(12, 12) for i in range(r_new)]
        if r_new == rank:
            break
        rank = r_new
    check("8 algebra generated on the one-particle sector by the step and the occupation numbers: dimension (four copies of M_3, not M_12)", rank, 36, rank == 36)


# ===========================================================================
# 9. The Clifford line (Sec. 6)
# ===========================================================================
def pauli_strings(n):
    """All 4^n Pauli strings X^a Z^b on n qubits as 2^n x 2^n matrices,
    (X^a Z^b)[y, x] = (-1)^{b.x} if y = x XOR a, else 0 (qubit i <-> bit i
    of the basis index).  Returns the key list [(a, b)] and the array."""
    D = 1 << n
    xs = np.arange(D)
    keys = [(a, b) for a in range(D) for b in range(D)]
    P = np.zeros((len(keys), D, D), dtype=complex)
    for idx, (a, b) in enumerate(keys):
        P[idx, xs ^ a, xs] = (-1.0) ** popcount(b & xs, n)
    return keys, P


def perm_conjugates_paulis_structural(pi, n):
    """Pauli test for a permutation pi of {0..2^n-1} using the exact form
    of the conjugates (no 2^n x 2^n matrices): U X_i U^T is the permutation
    matrix of x -> pi(pi^{-1}(x) XOR e_i), which is a phase times a Pauli
    string iff it is XOR by a constant (the only permutation matrices among
    the phase*Pauli strings are the X^a); U Z_i U^T is diagonal with entries
    (-1)^{(pi^{-1}(x))_i}, a phase times a Pauli string iff the sign vector
    is +-(-1)^{b.x} for one b, i.e. iff its Walsh transform (the HS
    decomposition over the diagonal strings Z^b) has one nonzero entry."""
    D = 1 << n
    xs = np.arange(D)
    pi = np.asarray(pi)
    inv = np.empty(D, dtype=np.int64)
    inv[pi] = xs
    Hw = (-1.0) ** popcount(xs[:, None] & xs[None, :], n)      # Walsh kernel
    for i in range(n):
        conjX = pi[inv ^ (1 << i)]
        if len(np.unique(conjX ^ xs)) != 1:
            return False
        sign = (-1.0) ** ((inv >> i) & 1)
        if int((np.abs(Hw @ sign) > 1e-9).sum()) != 1:
            return False
    return True


def section_9():
    section("9. The Clifford line: permutations of the computational basis")
    n = 3
    D = 1 << n
    xs = np.arange(D)
    keys, P = pauli_strings(n)
    kidx = {k: i for i, k in enumerate(keys)}
    # sanity of the Pauli strings: HS-orthogonal, unitary, X_i Z_i = -Z_i X_i
    PB = P.reshape(len(keys), D * D)
    gram = PB.conj() @ PB.T
    sane = (np.allclose(gram, D * np.eye(len(keys)))
            and all(np.allclose(M @ M.conj().T, np.eye(D)) for M in P)
            and all(np.allclose(P[kidx[(1 << i, 0)]] @ P[kidx[(0, 1 << i)]],
                                -P[kidx[(0, 1 << i)]] @ P[kidx[(1 << i, 0)]]) for i in range(n)))
    check("9 sanity: the 64 Pauli strings on 3 qubits are HS-orthogonal, unitary, X_i Z_i = -Z_i X_i",
          sane, True, sane)

    # (a) all 40320 permutations: honest Pauli test, vectorised over the
    # permutations.  U[p] is the permutation matrix (U[p][pi(x), x] = 1),
    # M = U G U^T for each of the 2n generators G in {X_i, Z_i}, and M is a
    # phase times a Pauli string iff its HS expansion in the 64 strings has
    # exactly one nonzero coefficient, that coefficient being in {1,-1,i,-i}.
    perms = np.array(list(permutations(range(D))), dtype=np.int64)
    NP = len(perms)
    check("9a number of permutations of 8 elements", NP, 40320, NP == 40320)
    U = np.zeros((NP, D, D))
    U[np.arange(NP)[:, None], perms, xs[None, :]] = 1.0
    gens = [kidx[(1 << i, 0)] for i in range(n)] + [kidx[(0, 1 << i)] for i in range(n)]
    phases = np.array([1, -1, 1j, -1j])
    is_cl = np.ones(NP, dtype=bool)
    for g in gens:
        M = U @ P[g] @ U.transpose(0, 2, 1)
        coef = M.reshape(NP, D * D) @ PB.conj().T / D       # tr(P^dagger M)/D for all 64 strings
        nz = np.abs(coef) > 1e-9
        single = nz.sum(axis=1) == 1
        c = coef[np.arange(NP), nz.argmax(axis=1)]          # the (first) nonzero coefficient
        phase_ok = np.abs(c[:, None] - phases[None, :]).min(axis=1) < 1e-9
        is_cl &= single & phase_ok
    n_cl = int(is_cl.sum())
    check("9a permutations conjugating every X_i, Z_i to a phase times a Pauli string", n_cl, 1344, n_cl == 1344)
    # affine bijections x -> Ax + b over GF(2)
    b = perms[:, 0]
    cols = [perms[:, 1 << i] ^ b for i in range(n)]
    pred = np.empty_like(perms)
    for x in range(D):
        v = b.copy()
        for i in range(n):
            if (x >> i) & 1:
                v = v ^ cols[i]
        pred[:, x] = v
    is_af = (pred == perms).all(axis=1)
    n_af = int(is_af.sum())
    check("9a affine bijections x -> Ax + b among the 40320 permutations", n_af, 1344, n_af == 1344)
    same = bool((is_cl == is_af).all())
    check("9a the Clifford permutations are exactly the affine ones (sets coincide)", same, True, same)
    # A invertible for every affine one: the 8 XOR-combinations of the columns are distinct
    inv_ok = True
    for p in np.flatnonzero(is_af):
        A = np.array([[(cols[i][p] >> r) & 1 for i in range(n)] for r in range(n)])
        inv_ok &= gf2_rank(A) == n
    check("9a the linear part A is invertible over GF(2) for every affine permutation", inv_ok, True, inv_ok)
    n_gl = sum(1 for bits in product((0, 1), repeat=n * n)
               if gf2_rank(np.array(bits).reshape(n, n)) == n)
    check("9a |AGL(3,2)| = 2^3 * |GL(3,2)| with |GL(3,2)| counted by enumeration (%d)" % n_gl,
          D * n_gl, 1344, D * n_gl == 1344)
    # literal cross-check on a sample: compare M with all 256 phase*Pauli matrices
    rng = np.random.default_rng(11)
    sample = [int(i) for i in rng.choice(NP, 150, replace=False)]
    sample += [int(i) for i in np.flatnonzero(is_af)[rng.choice(n_af, 50, replace=False)]]
    cand = (P[:, None, :, :] * phases[None, :, None, None]).reshape(-1, D, D)
    lit_ok = True
    for p in sample:
        res = True
        for g in gens:
            M = U[p] @ P[g] @ U[p].T
            res &= bool(np.abs(M[None] - cand).max(axis=(1, 2)).min() < 1e-12)
        lit_ok &= (res == bool(is_cl[p]))
    check("9a literal comparison with all 256 phase*Pauli matrices agrees with the HS test on %d sampled permutations" % len(sample),
          lit_ok, True, lit_ok)
    # the structural test (used on 8 bits below) agrees with the honest one on all 40320
    struct_ok = all(perm_conjugates_paulis_structural(perms[p], n) == bool(is_cl[p]) for p in range(NP))
    check("9a structural Pauli test (conjugates in closed form) agrees with the matrix test on all 40320",
          struct_ok, True, struct_ok)

    # (b) eight-bit maps
    x8 = np.arange(256)
    xor_all = all(is_affine_gf2(x8 ^ c, 8) for c in range(256))
    check("9b x -> x XOR c is affine for every c in 0..255", xor_all, True, xor_all)
    inc = (x8 + 1) & 0xFF
    dec = (x8 - 1) & 0xFF
    check("9b x -> x + 1 mod 256 is affine", is_affine_gf2(inc, 8), False, not is_affine_gf2(inc, 8))
    check("9b x -> x - 1 mod 256 is affine", is_affine_gf2(dec, 8), False, not is_affine_gf2(dec, 8))
    inc2 = (np.arange(4) + 1) % 4             # x + 1 restricted to the low two bits, as a map on 2 bits
    inc3 = (np.arange(8) + 1) % 8             # restricted to the low three bits, as a map on 3 bits
    check("9b x -> x + 1 on the low two bits (map on 2 bits) is affine", is_affine_gf2(inc2, 2), True, is_affine_gf2(inc2, 2))
    check("9b x -> x + 1 on the low three bits (map on 3 bits) is affine", is_affine_gf2(inc3, 3), False, not is_affine_gf2(inc3, 3))
    # the same two statements through the honest Pauli test (n = 2, 3 qubits)
    small_ok = True
    for nb, table, expect in ((2, inc2, True), (3, inc3, False)):
        Db = 1 << nb
        kb, Pb = pauli_strings(nb)
        kb_idx = {k: i for i, k in enumerate(kb)}
        Ub = np.zeros((Db, Db))
        Ub[table, np.arange(Db)] = 1.0
        PBb = Pb.reshape(len(kb), Db * Db)
        res = True
        for i in range(nb):
            for G in (Pb[kb_idx[(1 << i, 0)]], Pb[kb_idx[(0, 1 << i)]]):
                M = Ub @ G @ Ub.T
                coef = PBb.conj() @ M.flatten() / Db
                nz = np.abs(coef) > 1e-9
                res &= int(nz.sum()) == 1 and bool(np.abs(coef[nz][0] - phases).min() < 1e-9)
        small_ok &= (res == expect)
    check("9b Pauli test on 2 and 3 qubits confirms: +1 on 2 bits Clifford, +1 on 3 bits not", small_ok, True, small_ok)

    # (c) the incrementer as a cascade of multi-controlled NOTs
    def ckx(k):
        """C^k X: flip bit k if bits 0..k-1 are all 1 (k = 0: X on bit 0)."""
        def f(x):
            x = np.asarray(x)
            low_all_one = (x & ((1 << k) - 1)) == ((1 << k) - 1)
            return x ^ (low_all_one.astype(np.int64) << k)
        return f
    y = x8.copy()
    for k in range(7, -1, -1):                 # top bit first, X_0 last
        y = ckx(k)(y)
    cascade_ok = np.array_equal(y, inc)
    check("9c C^7X(0..6->7) first, ..., CNOT(0->1), X_0 last equals x -> x + 1 mod 256", cascade_ok, True, cascade_ok)
    aff = [is_affine_gf2(ckx(k)(x8), 8) for k in range(8)]
    check("9c affinity of the pieces C^k X, k = 0..7", aff, [True, True] + [False] * 6, aff == [True, True] + [False] * 6)
    n_non = sum(1 for a in aff if not a)
    check("9c number of non-affine (non-Clifford) carries", n_non, 6, n_non == 6)
    # Pauli test in closed form on the 8-bit permutations (validated on 3 bits above)
    struct = [perm_conjugates_paulis_structural(ckx(k)(x8), 8) for k in range(8)]
    struct_xor = perm_conjugates_paulis_structural(x8 ^ 0b10110, 8)
    struct_inc = perm_conjugates_paulis_structural(inc, 8)
    struct_dec = perm_conjugates_paulis_structural(dec, 8)
    agree = (struct == aff) and struct_xor and (not struct_inc) and (not struct_dec)
    check("9c Pauli test (closed-form conjugates) on 8 bits agrees with the affine test for the pieces, XOR, +1, -1",
          agree, True, agree)


# ===========================================================================
# 10. Other regenerable statements
# ===========================================================================
def section_10():
    section("10. Other regenerable statements")
    # Sec. 2: "positional operations (bit lookups) are reading-independent
    # while arithmetic operations are reading-dependent" for eight-bit words
    # exchanged by the bit-reversal involution R.  Formalisation used here:
    # R commutes with XOR and relabels bit lookups (bit p of x = bit 7-p of
    # R(x)) for all words, whereas R(x + 1) = R(x) + 1 fails for most x.
    x8 = np.arange(256)
    R = bitrev(x8, 8)
    inv_ok = np.array_equal(R[R], x8)
    xor_ok = bool((R[x8[:, None] ^ x8[None, :]] == (R[:, None] ^ R[None, :])).all())
    lookup_ok = all(np.array_equal((x8 >> p) & 1, (R >> (7 - p)) & 1) for p in range(8))
    n_plus = int((R[(x8 + 1) & 0xFF] == ((R + 1) & 0xFF)).sum())
    check("10 Sec. 2: R_8 is an involution", inv_ok, True, inv_ok)
    check("10 Sec. 2: R commutes with XOR and relabels bit lookups p -> 7-p (reading-independent), all 256 words",
          xor_ok and lookup_ok, True, xor_ok and lookup_ok)
    check("10 Sec. 2: R(x+1) = R(x)+1 holds for only %d of 256 words (reading-dependent)" % n_plus,
          n_plus < 256, True, n_plus < 256)
    # Sec. 4.2, Montina: 2*2^n - 2 is the real dimension of the pure-state
    # space CP^{2^n - 1} (an identity, recorded for the reader; not a check
    # of the theorem).
    ident = all(2 * 2 ** n - 2 == 2 * (2 ** n - 1) for n in range(1, 11))
    check("10 Sec. 4.2: 2*2^n - 2 = dim_R CP^(2^n - 1) = 2(2^n - 1), n = 1..10 (identity only)", ident, True, ident)



# ===========================================================================
# 11. The affine boundaries of the lifted automaton (Sec. 6)
# ===========================================================================
def section_11():
    section("11. The affine boundaries of the lifted automaton (Sec. 6)")
    # (a) Elementary cellular automata: of the 256 rule tables (3 inputs,
    #     1 output), exactly 16 are affine over GF(2); Rule 110 is not.
    aff_rules = [rule for rule in range(256)
                 if is_affine_gf2([(rule >> x) & 1 for x in range(8)], 3)]
    check("11 affine elementary rule tables among the 256", len(aff_rules), 16, len(aff_rules) == 16)
    info("affine rules: %s" % aff_rules)
    check("11 Rule 110 is affine", 110 in aff_rules, False, 110 not in aff_rules)
    # (a') A rule table g is not a bijection; its lift is the reversible
    #     embedding (x, y) -> (x, y XOR g(x)) on 3 + 1 bits (x in bits 0..2,
    #     y in bit 3), a permutation of the 16 basis states.  Honest Pauli
    #     test on four qubits (every generator X_i, Z_i conjugated to a
    #     phase times one of the 256 Pauli strings) for all 256 rules: the
    #     embedding is Clifford exactly for the 16 affine rules.
    k4, P4 = pauli_strings(4)
    k4_idx = {k: i for i, k in enumerate(k4)}
    PB4 = P4.reshape(len(k4), 16 * 16)
    phases = np.array([1, -1, 1j, -1j])
    z4 = np.arange(16)
    cl_rules, emb_affine_ok, emb_bij_ok = [], True, True
    for rule in range(256):
        g = np.array([(rule >> x) & 1 for x in range(8)])
        emb = (z4 & 7) | ((((z4 >> 3) & 1) ^ g[z4 & 7]) << 3)
        emb_bij_ok &= sorted(emb.tolist()) == list(range(16))
        emb_affine_ok &= (is_affine_gf2(emb, 4) == (rule in aff_rules))
        U4 = np.zeros((16, 16))
        U4[emb, z4] = 1.0
        res = True
        for i in range(4):
            for G in (P4[k4_idx[(1 << i, 0)]], P4[k4_idx[(0, 1 << i)]]):
                M = U4 @ G @ U4.T
                coef = PB4.conj() @ M.flatten() / 16
                nz = np.abs(coef) > 1e-9
                res &= int(nz.sum()) == 1 and bool(np.abs(coef[nz][0] - phases).min() < 1e-9)
        if res:
            cl_rules.append(rule)
    check("11 the reversible embedding (x, y) -> (x, y XOR g(x)) of each of the 256 rule tables is a bijection of 4 bits, affine iff g is",
          emb_bij_ok and emb_affine_ok, True, emb_bij_ok and emb_affine_ok)
    check("11 Pauli test on four qubits: the embeddings that are Clifford are exactly those of the 16 affine rules",
          cl_rules, aff_rules, cl_rules == aff_rules)
    # (b) The cell rule of a word: the lookup reads bit B-1-h of the word
    #     for neighbourhood sum h = 0..3, so a fixed word runs the
    #     totalistic rule with outputs (o_0, o_1, o_2, o_3) = its top four
    #     bits.  Of the 16 totalistic rules, 4 are affine: the constants and
    #     parity with its complement (top bits 0000, 1111, 0101, 1010).
    aff_pat = []
    for pat in range(16):
        o = [(pat >> (3 - h)) & 1 for h in range(4)]      # o[h] = bit (B-1-h)
        table = [o[popcount(np.array([x]), 3)[0]] for x in range(8)]
        if is_affine_gf2(table, 3):
            aff_pat.append("%04d" % int(bin(pat)[2:]))
    check("11 affine totalistic cell rules (top four bits of the word)", sorted(aff_pat),
          ["0000", "0101", "1010", "1111"], sorted(aff_pat) == ["0000", "0101", "1010", "1111"])
    # (c) Translations n -> n + d mod 2^b: affine on the low b bits iff
    #     b <= v + 2, v the index of the lowest set bit of |d|, for the six
    #     values d = (2c'-1)h - 1 that occur (h = 0..3, c' = 0, 1; d = 0 is
    #     the identity).
    first = {}
    for d in (-4, -3, -2, -1, 1, 2):
        v = (abs(d) & -abs(d)).bit_length() - 1
        res = [is_affine_gf2([(x + d) % (1 << b) for x in range(1 << b)], b) for b in range(1, 9)]
        first[d] = next(b for b, ok in zip(range(1, 9), res) if not ok)
        check("11 translation by d = %d: affine on the low b bits iff b <= v + 2 = %d (first non-affine b)" % (d, v + 2),
              first[d], v + 3, first[d] == v + 3)
    check("11 d = +-1: first non-affine carry is the one into the third bit", (first[1], first[-1]), (3, 3),
          (first[1], first[-1]) == (3, 3))
    check("11 d = +-2: into the fourth bit", (first[2], first[-2]), (4, 4), (first[2], first[-2]) == (4, 4))
    check("11 d = -4: into the fifth bit", first[-4], 5, first[-4] == 5)
    # (d) The full rule-layer step of the companion paper's automaton at
    #     B = 8: n' = (n + d) & 255 if n + d >= 0, else (n + d + r) & 255
    #     (the reduction modulo N = k 2^B + r, k >= 1, in its residue form),
    #     for every (h, c') and several residues r: non-affine except the
    #     identity case h = 1, c' = 1 (d = 0).
    B = 8
    mask = (1 << B) - 1
    ok_all = True
    for r in (80, 95, 96, 112):
        for h in range(4):
            for c in (0, 1):
                d = (2 * c - 1) * h - 1
                table = [((n + d) & mask) if n + d >= 0 else ((n + d + r) & mask) for n in range(1 << B)]
                aff = is_affine_gf2(table, B)
                ok_all &= (aff == (d == 0))
    check("11 full rule-layer step at B = 8 (r = 80, 95, 96, 112): affine iff (h, c') = (1, 1)", ok_all, True, ok_all)
    # (e) The lookup with the neighbourhood sum lifted: (h, n) -> n_{B-1-h},
    #     h in 2 bits, n in 8 bits, is not affine over GF(2).
    table = []
    for z in range(1 << 10):
        h = z & 3
        n = z >> 2
        table.append((n >> (B - 1 - h)) & 1)
    mux_aff = is_affine_gf2(table, 10)
    check("11 the multiplexer (h, n) -> n_{B-1-h} on 2 + 8 bits is affine", mux_aff, False, not mux_aff)


# ===========================================================================
# 12. The companion automaton's word map and the residue (Secs. 3.1, 3.4, 6)
# ===========================================================================
def section_12():
    section("12. The companion automaton's word map and the residue (Secs. 3.1, 3.4, 6)")
    import math
    B = 8
    M = 1 << B
    mask = M - 1

    def wordmap(d, r):
        # n' = (n + d) & mask if n + d >= 0 else (n + d + r) & mask
        # (the reduction modulo N = k 2^B + r, k >= 1, in residue form)
        return np.array([((n + d) & mask) if n + d >= 0 else ((n + d + r) & mask) for n in range(M)])

    def eventual_image_set(f):
        X = set(range(len(f)))
        k = 0
        while True:
            Y = {int(f[x]) for x in X}
            if Y == X:
                return X, k
            X = Y
            k += 1

    def cycle_lengths(f, E):
        seen = set()
        L = []
        for x in sorted(E):
            if x in seen:
                continue
            y = x
            n = 0
            while True:
                seen.add(y)
                y = int(f[y])
                n += 1
                if y == x:
                    break
            L.append(n)
        return sorted(set(L))

    # (a) Sec. 3.1: for d < 0 the word map is dissipative with core [0, r),
    #     nilpotency index ceil((2^B - r)/|d|), image of size 2^B - |d|
    #     (two words to one), and the core rotation has cycles of length
    #     r / gcd(|d|, r); for d > 0 it is a rotation of Z_{2^B}.
    ok_core = ok_nil = ok_img = ok_cyc = ok_rot = True
    for r in (32, 96):
        for d in (-1, -2, -3, -4):
            f = wordmap(d, r)
            E, k = eventual_image_set(f)
            ok_core &= (E == set(range(r)))
            ok_nil &= (k == math.ceil((M - r) / abs(d)))
            ok_img &= (len(set(f.tolist())) == M - abs(d))
            ok_cyc &= (cycle_lengths(f, E) == [r // math.gcd(abs(d), r)])
        for d in (1, 2):
            f = wordmap(d, r)
            ok_rot &= (len(set(f.tolist())) == M) and (cycle_lengths(f, set(range(M))) == [M // math.gcd(d, M)])
    check("12 d < 0: core of the word map = [0, r), r = 32, 96, d = -1..-4 (B = 8)", ok_core, True, ok_core)
    check("12 d < 0: nilpotency index of the transient = ceil((2^B - r)/|d|)", ok_nil, True, ok_nil)
    check("12 d < 0: the word map is not injective (image size 2^B - |d|)", ok_img, True, ok_img)
    check("12 d < 0: cycles of the core rotation have length r/gcd(|d|, r)", ok_cyc, True, ok_cyc)
    check("12 d > 0: the word map is the rotation of Z_{2^B} by d (bijective, cycles 2^B/gcd(d, 2^B))", ok_rot, True, ok_rot)
    # (b) Sec. 6: the only affine rotations of Z_256 are by multiples of 64.
    aff_rot = [d for d in range(M) if is_affine_gf2([(x + d) & mask for x in range(M)], B)]
    check("12 affine rotations of Z_256 (multiples of 2^(B-2))", aff_rot, [0, 64, 128, 192], aff_rot == [0, 64, 128, 192])
    # (c) Sec. 6: the top character chi_{2^(B-1)} is constant (+1) on [0, r)
    #     for every r <= 2^(B-1); the excitability e_1 = (r - 2^(B-2))/r is
    #     the density on [0, r) of the words with chi_{2^(B-2)} = -1.
    top_ok = all(all(((n >> (B - 1)) & 1) == 0 for n in range(r)) for r in range(1, (1 << (B - 1)) + 1))
    check("12 chi_{2^(B-1)} = +1 on [0, r) for every r <= 2^(B-1)", top_ok, True, top_ok)
    e1_ok = all(abs(sum(1 for n in range(r) if (n >> (B - 2)) & 1) / r - (r - 2 ** (B - 2)) / r) < 1e-12
                for r in range(2 ** (B - 2), 2 ** (B - 1) + 1))
    check("12 e_1 = (r - 2^(B-2))/r = density of chi_{2^(B-2)} = -1 on [0, r), all 2^(B-2) <= r <= 2^(B-1)", e1_ok, True, e1_ok)
    # outside that range the identity fails: below 2^(B-2) the density is 0 and the formula negative; above 2^(B-1) the bit clears again
    e1_out = [r for r in range(1, 1 << B) if abs(sum(1 for n in range(r) if (n >> (B - 2)) & 1) / r - (r - 2 ** (B - 2)) / r) < 1e-12]
    check("12 the residues 1 <= r < 2^B at which the identity holds are exactly 2^(B-2) .. 2^(B-1)", [e1_out[0], e1_out[-1], len(e1_out)],
          [64, 128, 65], [e1_out[0], e1_out[-1], len(e1_out)] == [64, 128, 65])
    # (d) Sec. 3.4: the lookup, bit B-1-h of the word, equals
    #     (1 - chi_{2^(B-1-h)}(n))/2, chi_m(n) = (-1)^{m.n}, for h = 0..3.
    look_ok = all(((n >> (B - 1 - h)) & 1) == (1 - (-1) ** bin((1 << (B - 1 - h)) & n).count("1")) // 2
                  for n in range(M) for h in range(4))
    check("12 lookup: bit_{B-1-h}(n) = (1 - chi_{2^(B-1-h)}(n))/2, all 256 words, h = 0..3", look_ok, True, look_ok)


# ===========================================================================
# 13. The two readings of the companion automaton are two dynamics
#     (Secs. 2 and 3.4): exact counts of periodic states
# ===========================================================================
def nuca_weights(N, B, T, msb):
    """w[a, b, c] = number of words n0 of a cell whose own state history is b
    and whose neighbours' histories are a and c (T-periodic binary histories,
    coded as T-bit integers) such that the lookup reproduces b at every step
    and the word returns to n0 after T steps.  Rule of Sec. 2; msb: the lookup
    reads the bit of significance B-1-h, lsb: of significance h."""
    H = 1 << T
    M = 1 << B
    mask = M - 1
    hist = np.array([[(x >> t) & 1 for t in range(T)] for x in range(H)], dtype=np.int64)
    a = hist[:, None, None, :]
    b = hist[None, :, None, :]
    c = hist[None, None, :, :]
    h = a + b + c
    bnext = np.roll(hist, -1, axis=1)[None, :, None, :] + 0 * h
    w = np.zeros((H, H, H), dtype=np.int64)
    for n0 in range(M):
        n = np.full((H, H, H), n0, dtype=np.int64)
        ok = np.ones((H, H, H), dtype=bool)
        for t in range(T):
            ht = h[..., t]
            pos = (B - 1 - ht) if msb else ht
            bit = (n >> pos) & 1
            ok &= bit == bnext[..., t]
            n = ((n + (2 * bit - 1) * ht - 1) % N) & mask
        ok &= n == n0
        w += ok
    return w


def periodic_states(N, B, T, msb):
    """Exact number of states (c, n) of the ring of N cells with f^T = identity
    on them: the trace of the N-th power of the transfer matrix
    M[(a, b), (b, c)] = w[a, b, c] over pairs of adjacent histories."""
    w = nuca_weights(N, B, T, msb)
    H = w.shape[0]
    Mx_ = np.zeros((H * H, H * H), dtype=object)
    for a_ in range(H):
        for b_ in range(H):
            for c_ in range(H):
                Mx_[a_ * H + b_, b_ * H + c_] = int(w[a_, b_, c_])
    P = None
    base = Mx_
    e = N
    while e:
        if e & 1:
            P = base if P is None else P.dot(base)
        e >>= 1
        if e:
            base = base.dot(base)
    return int(sum(P[i, i] for i in range(H * H)))


def nuca_global_map(N, B, msb):
    """The global map of the ring on all (2 * 2^B)^N states, by enumeration."""
    M = 1 << B
    S_ = (2 * M) ** N
    x = np.arange(S_, dtype=np.int64)
    codes = np.empty((N, S_), dtype=np.int64)
    for j in range(N):
        codes[j] = x % (2 * M)
        x //= (2 * M)
    c = codes // M
    n = codes % M
    h = np.roll(c, 1, axis=0) + c + np.roll(c, -1, axis=0)
    pos = (B - 1 - h) if msb else h
    c2 = (n >> pos) & 1
    n2 = ((n + (2 * c2 - 1) * h - 1) % N) & (M - 1)
    out_ = np.zeros(S_, dtype=np.int64)
    for j in reversed(range(N)):
        out_ = out_ * (2 * M) + (c2[j] * M + n2[j])
    return out_


def frozen_states(N, w0, w1, gap):
    """Number of states with a frozen configuration of the kind of [P1, Prop. 2]:
    binary rings of N cells with at least `gap` zeros between consecutive ones,
    w0 words per zero and w1 per one (exact, by a transfer matrix on the number
    of zeros since the last one)."""
    m = gap + 1
    Tm = [[0] * m for _ in range(m)]
    for z in range(m):
        Tm[min(z + 1, gap)][z] += w0
        if z == gap:
            Tm[0][z] += w1
    P = [[int(i == j) for j in range(m)] for i in range(m)]
    for _ in range(N):
        P = [[sum(P[i][k] * Tm[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
    return sum(P[i][i] for i in range(m))


def section_13():
    section("13. The two readings are two dynamics (Secs. 2, 3.4): exact counts of periodic states")
    # (a) the transfer-matrix count against enumeration of all states, rings of 3 and 4 cells, B = 4
    agree = True
    cases = 0
    for N in (3, 4):
        for msb in (True, False):
            f = nuca_global_map(N, 4, msb)
            ident = np.arange(len(f))
            g = ident.copy()
            for T in range(1, 5):
                g = f[g]
                agree &= int((g == ident).sum()) == periodic_states(N, 4, T, msb)
                cases += 1
    check("13 transfer-matrix count of the states with f^T = id equals enumeration (B = 4; N = 3, 4; T = 1..4; both readings)",
          (agree, cases), (True, 16), agree and cases == 16)
    # (b) a ring inside the regime of the papers: B = 4, N = 2^B + 4 = 20, T = r = 4
    B, N, T = 4, 20, 4
    n_msb = periodic_states(N, B, T, True)
    n_lsb = periodic_states(N, B, T, False)
    check("13 B = 4, N = 20: states of period dividing 4, reading of Sec. 2 (bit B-1-h)", n_msb, 4 ** 20 * 38609, n_msb == 4 ** 20 * 38609)
    check("13 B = 4, N = 20: states of period dividing 4, other reading (bit h)", n_lsb, 2 ** 51 + 2560, n_lsb == 2 ** 51 + 2560)
    check("13 the two readings have different numbers of periodic states (not conjugate)", n_msb != n_lsb, True, n_msb != n_lsb)
    # (c) the first number is the number of frozen states of [P1, Prop. 2] at r = 4, B = 4:
    #     ones at mutual distance >= 3 (two zeros between), 8 static words per one (bit B-2 set), 4 sea words per zero
    fz = frozen_states(N, 4, 8, 2)
    check("13 B = 4, N = 20: frozen states of [P1, Prop. 2] (ones at distance >= 3, 8 words per one, 4 per zero)", fz, n_msb, fz == n_msb)
    #     and such states do return after four steps with their cells unchanged throughout, so that the
    #     equality of the two counts makes the two sets equal: 2000 states drawn at random
    rng = np.random.default_rng(4)
    ret, drawn = 0, 0
    static8 = np.array([w for w in range(1 << B) if (w >> (B - 2)) & 1])
    while drawn < 2000:
        c0 = (rng.random(N) < 0.3).astype(np.int64)
        ones = np.flatnonzero(c0)
        if any(c0[(i + 1) % N] or c0[(i + 2) % N] for i in ones):
            continue                                     # two ones closer than three cells
        n0 = np.where(c0 == 1, static8[rng.integers(0, 8, N)], rng.integers(0, 4, N))
        c_, n_ = c0.copy(), n0.copy()
        frozen = True
        for _ in range(T):
            h = np.roll(c_, 1) + c_ + np.roll(c_, -1)
            c2 = (n_ >> (B - 1 - h)) & 1
            n_ = ((n_ + (2 * c2 - 1) * h - 1) % N) & ((1 << B) - 1)
            frozen &= bool(np.array_equal(c2, c0))
            c_ = c2
        ret += int(frozen and np.array_equal(n_, n0))
        drawn += 1
    check("13 states of that kind drawn at random keep their cells and return after 4 steps (of 2000)", ret, 2000, ret == 2000)
    # (d) the bit reversal R, applied to the words of every cell, does not carry one reading into the other:
    #     R o M_msb o R against M_lsb on a random state of a ring of 800 cells, B = 8
    B, N = 8, 800
    rng = np.random.default_rng(1)
    c = rng.integers(0, 2, N)
    n = rng.integers(0, 1 << B, N)
    rev = np.array([bitrev(m, B) for m in range(1 << B)])

    def one(c_, n_, msb):
        h = np.roll(c_, 1) + c_ + np.roll(c_, -1)
        c2 = (n_ >> ((B - 1 - h) if msb else h)) & 1
        return c2, ((n_ + (2 * c2 - 1) * h - 1) % N) & ((1 << B) - 1)
    c_a, n_a = one(c, rev[n], True)
    n_a = rev[n_a]                                   # R M_msb R
    c_b, n_b = one(c, n, False)                      # M_lsb
    same_c = bool(np.array_equal(c_a, c_b))
    differ = int((n_a != n_b).sum())
    check("13 R M_msb R and M_lsb agree on the new cell states (the lookup is carried along)", same_c, True, same_c)
    check("13 R M_msb R and M_lsb differ on the new words (cells of 800 that differ > 0)", differ > 0, True, differ > 0)
    info("cells whose new word differs between R M_msb R and M_lsb: %d of %d" % (differ, N))


# ===========================================================================
# 14. Equivalence classes of states with a common future and the core
#     (Sec. 4.5, the remark on [tH99])
# ===========================================================================
def section_14():
    section("14. States with the same eventual future: the classes are in bijection with the core (Sec. 4.5)")
    rng = np.random.default_rng(7)
    ok = True
    nmaps = 0
    for _ in range(300):
        n = int(rng.integers(2, 41))
        f = rng.integers(0, n, n)
        E, _stab = eventual_image(f)
        g = np.arange(n)
        for _k in range(n):                          # f^n(x): two states are equivalent iff these agree
            g = f[g]
        classes = {}
        for x in range(n):
            classes.setdefault(int(g[x]), []).append(x)
        inE = set(E.tolist())
        ok &= len(classes) == len(E)
        ok &= all(sum(1 for x in cl if x in inE) == 1 for cl in classes.values())
        nmaps += 1
    check("14 on %d random maps: number of classes = |E|, and each class contains exactly one core state" % nmaps, ok, True, ok)


# ===========================================================================
def main():
    t0 = time.time()
    out("verify_p2.py -- the paper's computational claims regenerated (numpy %s)" % np.__version__)
    for sec in (section_1, section_2, section_3, section_3_norm, section_4, section_5,
                section_6, section_7, section_8, section_9, section_10, section_11, section_12,
                section_13, section_14):
        try:
            sec()
        except Exception:      # a crash is a finding too; report and go on
            COUNT["FAIL"] += 1
            out("FAIL %s raised an exception: %s vs (no exception)" % (sec.__name__, traceback.format_exc().strip().splitlines()[-1]))
            for line in traceback.format_exc().strip().splitlines():
                info(line)
    out("")
    out("SUMMARY: %d PASS, %d FAIL" % (COUNT["PASS"], COUNT["FAIL"]))
    info("elapsed %.1f s" % (time.time() - t0))
    with open("verify_p2_results.txt", "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    return COUNT["FAIL"]


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
