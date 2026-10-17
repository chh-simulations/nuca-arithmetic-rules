#!/usr/bin/env python3
"""Verification suite of the transition paper.  Regenerates the numbers
and compares them with the recorded values in expected_pre.py.

    python3 verify_pre.py quick     (the runner gate, the exact bit table,
                                     the islands, the B = 8 fronts, the
                                     short bulk and quench runs; ~30 min)
    python3 verify_pre.py full      (everything in the paper; hours)

Every check prints PASS or FAIL with the value and the recorded value;
nothing is truncated.  Results go to verify_pre_results.txt.
"""
import sys, time, os
from fractions import Fraction
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import nuca_fast as nf
import front as FR
import threshold as TH
import quench as QU
from expected_pre import EXPECTED

MODE = sys.argv[1] if len(sys.argv) > 1 else "quick"
OUT = open("verify_pre_results.txt", "w")
npass = nfail = 0


def emit(line):
    print(line, flush=True); OUT.write(line + "\n"); OUT.flush()


def check(name, value, expected, tol=0.0, rel=False):
    global npass, nfail
    if isinstance(expected, (list, tuple, np.ndarray)):
        ok = len(value) == len(expected) and all(abs(a - b) <= (tol * abs(b) if rel else tol) for a, b in zip(value, expected))
        emit(f"{'PASS' if ok else 'FAIL'} {name}: {np.array2string(np.asarray(value, dtype=float), precision=6)} vs {np.array2string(np.asarray(expected, dtype=float), precision=6)}")
    else:
        ok = abs(value - expected) <= (tol * abs(expected) if rel else tol)
        emit(f"{'PASS' if ok else 'FAIL'} {name}: {value} vs {expected}")
    if ok: npass += 1
    else: nfail += 1


def checkbool(name, ok):
    global npass, nfail
    emit(f"{'PASS' if ok else 'FAIL'} {name}")
    if ok: npass += 1
    else: nfail += 1


# ---------------------------------------------------------------- S1 runner
emit("S1  the compiled runner against the canonical loop (gate_fast.py)")
import subprocess
res = subprocess.run([sys.executable, "gate_fast.py"], capture_output=True, text=True)
lines = res.stdout.splitlines()
g_pass = sum(1 for l in lines if l.startswith("PASS")); g_fail = sum(1 for l in lines if l.startswith("FAIL"))
checkbool(f"gate_fast.py: {g_pass} PASS, {g_fail} FAIL", g_fail == 0 and g_pass >= 110)

# ---------------------------------------------------------------- S2 exact bit table
emit("S2  the readable bits of an ignited word, exact, B = 6..14")
ok_all = True
for B in range(6, 15):
    base = 1 << B; q = base // 4
    for r in range(q + 1, base // 2 + 1):
        words = range(q, r)
        has3 = any((w >> (B - 3)) & 1 for w in words)
        has4 = any((w >> (B - 4)) & 1 for w in words)
        both = any(((w >> (B - 3)) & 1) and ((w >> (B - 4)) & 1) for w in words)
        u = Fraction(r, 1 << (B - 4))
        ok_all &= (has3 == (u > 6)) and (has4 == (u > 5)) and (both == (u > 7))
        e1 = Fraction(r - q, r)
        ok_all &= (e1 > Fraction(1, 3)) == (u > 6) and (e1 > Fraction(1, 5)) == (u > 5) and (e1 > Fraction(3, 7)) == (u > 7)
checkbool("bit B-3 present iff u > 6 (e_1 > 1/3); bit B-4 iff u > 5 (e_1 > 1/5); both iff u > 7 (e_1 > 3/7)", ok_all)
checkbool("e_1(r_c) = 1/3 exactly, B = 6..14", all(Fraction(3 * (1 << (B - 3)) - (1 << (B - 2)), 3 * (1 << (B - 3))) == Fraction(1, 3) for B in range(6, 15)))
# the trap: words just below 2^(B-1) have bits B-2, B-3, B-4 set and B-1 clear; just above: B-1 set, the next three clear
ok_trap = True
for B in range(6, 15):
    top = 1 << (B - 1)
    for d in range(1, min(8, 1 << (B - 4)) + 1):
        w = top - d; ok_trap &= all((w >> (B - k)) & 1 for k in (2, 3, 4)) and not (w >> (B - 1)) & 1
        w = top + d - 1; ok_trap &= (w >> (B - 1)) & 1 == 1 and not any((w >> (B - k)) & 1 for k in (2, 3, 4))
checkbool("the trap at 2^(B-1): below it bits B-2..B-4 set and B-1 clear, above it the reverse (min(8, 2^(B-4)) words each side)", ok_trap)

# ---------------------------------------------------------------- S3 islands
emit("S3  islands at B = 10, r = 376")
B = 10; base = 1 << B; r = 376; N = 2 * base + r
c, n = nf.ic_random(N, B, 1)
nf.run(c, n, r, B, 200000, stop_dead=False)
isl = nf.islands(c, n, r, B, steps=4000)
check("number of islands (N = 2424, 2e5 steps, seed 1)", len(isl), 13)
check("island widths (changed cells within 4000 steps)", [b - a + 1 for a, b in isl], [18, 18, 13, 14, 14, 15, 12, 16, 26, 14, 12, 15, 14])
cores = []; band_lo = band_hi = 0; off = []
for a, b in isl:
    seg = n[a:b + 1]; core = seg >= base // 2 - 32; cores.append(int(core.sum()))
    for w in seg[core]:
        d1 = abs(int(w) - 512); d2 = abs(int(w) - 768)
        if d1 <= d2: band_lo += 1
        else: band_hi += 1
        off.append(min(d1, d2))
check("core sizes", sorted(cores), [5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 9, 10])
check("core words in the band at 512 / at 768", [band_lo, band_hi], [61, 26])
check("largest offset of a core word from its band centre", max(off), 20)
# transplant of the first island into three fresh seas: 2e6 steps, same statistics
a, b = isl[0]; a -= 8; b += 9
stats = []
for trial in range(3):
    rng = np.random.default_rng(100 + trial)
    cs = np.zeros(N, dtype=np.int64); ns = rng.integers(0, r, N).astype(np.int64)
    x0 = N // 2; cs[x0:x0 + (b - a)] = c[a:b]; ns[x0:x0 + (b - a)] = n[a:b]
    nch, n1, lo, hi, td = nf.run(cs, ns, r, B, 2_000_000, stop_dead=True)
    stats.append((td, round(float(nch[-100000:].mean()), 2), round(float(n1[-100000:].mean()), 2), int(hi[lo >= 0].max() - x0), int(lo[lo >= 0].min() - x0)))
check("transplant: alive at 2e6 in three seas (tdead = 0)", [s[0] for s in stats], [0, 0, 0])
check("transplant: cells changing per step, late", [s[1] for s in stats], [2.53, 2.53, 2.53], 0.01)
check("transplant: cells on per step, late", [s[2] for s in stats], [4.16, 4.16, 4.16], 0.01)
check("transplant: extreme offsets of changed cells (max right, min left), sea 100", stats[0][3:], (30, 4))
# sensitivity
rng = np.random.default_rng(100); cs = np.zeros(N, dtype=np.int64); ns = rng.integers(0, r, N).astype(np.int64)
x0 = N // 2; cs[x0:x0 + (b - a)] = c[a:b]; ns[x0:x0 + (b - a)] = n[a:b]
cA, nA = cs.copy(), ns.copy(); cB, nB = cs.copy(), ns.copy(); mid = x0 + (b - a) // 2; nB[mid] = (nB[mid] + 1) & (base - 1)
dist = []
for t in range(1, 3001):
    nf.run(cA, nA, r, B, 1, stop_dead=False); nf.run(cB, nB, r, B, 1, stop_dead=False)
    if t in (10, 100, 1000, 3000): dist.append((int((cA != cB).sum()), int((nA != nB).sum())))
check("sensitivity: (cells, words) differing at t = 10, 100, 1000, 3000 after a +1 word perturbation",
      [x for p in dist for x in p], [0, 1, 1, 13, 3, 17, 8, 18])

emit("S3b island cores at B = 9, 11 (r = r_c - 8*2^(B-10), 200*2^B steps)")
exp_cores = {9: (8, [5, 5, 5, 5, 5, 6, 7, 9], 38, 9), 11: (49, None, 247, 99)}
for Bx in ((9, 11) if MODE == "quick" else (9, 11, 12)):
    basex = 1 << Bx; rx = 3 * basex // 8 - 8 * max(1, basex // 1024); Nx = 4 * basex + rx
    cx, nx = nf.ic_random(Nx, Bx, 1); nf.run(cx, nx, rx, Bx, 200 * basex, stop_dead=False)
    islx = nf.islands(cx, nx, rx, Bx, steps=4000)
    cores = []; lo_ = hi_ = 0
    for a_, b_ in islx:
        seg = nx[a_:b_ + 1]; core = seg >= basex // 2 - 32; cores.append(int(core.sum()))
        for w in seg[core]:
            if abs(int(w) - basex // 2) <= abs(int(w) - 3 * basex // 4): lo_ += 1
            else: hi_ += 1
    if Bx in exp_cores:
        e = exp_cores[Bx]
        check(f"B={Bx} r={rx}: number of islands", len(islx), e[0])
        if e[1] is not None: check(f"B={Bx} r={rx}: core sizes", sorted(cores), e[1])
        else: check(f"B={Bx} r={rx}: mode of the core size", int(np.bincount(cores).argmax()), 5)
        check(f"B={Bx} r={rx}: core words in the lower / upper band", [lo_, hi_], [e[2], e[3]])
    else:
        check(f"B={Bx} r={rx}: number of islands", len(islx), 80)
        check(f"B={Bx} r={rx}: mode of the core size", int(np.bincount(cores).argmax()), 5)
        check(f"B={Bx} r={rx}: core words in the lower / upper band", [lo_, hi_], [401, 185])

# ---------------------------------------------------------------- S4 fronts
emit("S4  front velocities (front.py geometry, two seeds, four fronts)")
plan = {8: ("quick", 500000, 250), 9: ("quick", 1000000, 500), 10: ("full", 2000000, 1000), 11: ("full", 4000000, 1000), 12: ("full", 8000000, 1000)}
for Bx, (level, T, every) in plan.items():
    if level == "full" and MODE != "full": continue
    if Bx not in EXPECTED['fronts']: continue
    basex = 1 << Bx; k = 24 if Bx <= 10 else 8
    for rx, vexp in sorted(EXPECTED['fronts'][Bx].items()):
        if MODE == "quick" and Bx == 9 and rx not in (190, 191, 192, 193, 194, 195, 256): continue
        Nx = k * basex + rx; ax, bx = Nx // 4, 3 * Nx // 4
        t0 = time.time(); vs = []
        for seed in (1, 2):
            t, lo, hi, nch, td = FR.one((Bx, rx, k, seed, T, every))
            vs += FR.velocity(t, lo, hi, Nx, ax, bx)
        check(f"B={Bx} r={rx} (r_c{rx - 3 * basex // 8:+d}) four front velocities ({time.time() - t0:.0f}s)", vs, vexp, 1e-6, rel=True)

# ---------------------------------------------------------------- S5 bulk runs: collapse, plateau, growth, domain density
emit("S5  random start at B = 10: collapse to islands, creep at 383-384, invasion at 385; domain density")
exp_bulk = {382: (1.2, 1.8, 0), 383: (3.0, 25.0, 1), 384: (5.0, 40.0, 1), 385: (44.0, 47.0, 1)}
T5 = 2_000_000 if MODE == "full" else 400_000
for rx in (382, 383, 384, 385):
    Nx = 10 * 1024 + rx; cx, nx = nf.ic_random(Nx, 10, 1)
    nch, n1, lo, hi, td = nf.run(cx, nx, rx, 10, T5, stop_dead=False)
    tail = 100 * float(nch[-10000:].mean()) / Nx
    islx = nf.islands(cx, nx, rx, 10, steps=2000); size = sum(b_ - a_ + 1 for a_, b_ in islx)
    nch2, n12, _, _, _ = nf.run(cx, nx, rx, 10, 2000, stop_dead=False)
    dens = 100 * float(nch2.mean()) / size
    rec = {382: (1.802, 1.72, 1.7, 1.3), 383: (21.3, 44.9, 50.1, 4.0), 384: (34.57, 46.1, 77.6, 6.5), 385: (46.01, 45.9, 100.0, 46.0)}
    if MODE == "full":
        check(f"r={rx}: tail activity at 2e6 (% of N)", tail, rec[rx][0], 0.3)
        check(f"r={rx}: activity inside the domains (%)", dens, rec[rx][1], 1.0)
        check(f"r={rx}: fraction of the ring covered by domains (%)", 100 * size / Nx, rec[rx][2], 2.0)
    else:
        lo_, hi_, grow = exp_bulk[rx]
        checkbool(f"r={rx}: tail activity at 4e5 = {tail:.2f}% in [{lo_}, {hi_}]; domains {len(islx)}, inside density {dens:.1f}%",
                  lo_ <= tail <= hi_ and (dens > 35 if grow else True))

# ---------------------------------------------------------------- S5b domain densities at r_c - 1 and r_c + 1
emit("S5b domain density at r_c - 1 and bulk density at r_c + 1 (domain_density.py)")
import domain_density as DD
exp_dd = {(8, 95): (2, 0.9818, 46.94, 72.23),
          (8, 97): (1, 1.0, 46.7, 72.42),
          (9, 191): (12, 0.5286, 44.76, 68.89),
          (9, 193): (1, 1.0, 45.63, 70.56),
          (10, 383): (34, 0.5011, 44.91, 69.2),
          (10, 385): (1, 1.0, 46.01, 71.11),
          (11, 767): (49, 0.4033, 44.66, 69.03),
          (11, 769): (1, 1.0, 45.98, 71.15),
          (12, 1535): (54, 0.3966, 44.89, 69.4),
          (12, 1537): (1, 1.0, 46.22, 71.41)}
plan_dd = {8: (8, 1000000), 9: (8, 1000000), 10: (10, 2000000), 11: (8, 4000000), 12: (4, 8000000)}
for (Bx, rx), e in sorted(exp_dd.items()):
    if MODE == "quick" and Bx > 9: continue
    k, T = plan_dd[Bx]
    rec = DD.measure(Bx, rx, k, T, 1)
    check(f"B={Bx} r={rx} (r_c{rx - 3 * (1 << (Bx - 3)):+d}): domains, covered fraction, activity inside (%), ones inside (%)",
          [rec['domains'], rec['covered'], rec['density'], rec['ones']], list(e), 1e-9)

# ---------------------------------------------------------------- S6 quench
emit("S6  the developed active phase below threshold (quench.py)")
for (Bx, rx, seed), e in sorted(EXPECTED['quench'].items()):
    if MODE == "quick" and (Bx != 8 or rx not in (80, 94, 96)): continue
    basex = 1 << Bx; k = 10 if Bx == 10 else 32
    c0, n0 = nf.ic_random(k * basex + basex // 2, Bx, seed); nf.run(c0, n0, basex // 2, Bx, 100000 if Bx == 10 else 50000, stop_dead=False)
    Nx = k * basex + rx; cx, nx = c0[:Nx].copy(), n0[:Nx].copy()
    T = e['T']; cps = [int(T * f) for f in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0)]; t = 0; cover = []; dom = []
    for cp in cps:
        nf.run(cx, nx, rx, Bx, cp - t - 4000, stop_dead=False)
        islx = nf.islands(cx, nx, rx, Bx, steps=4000); cover.append(round(sum(b_ - a_ + 1 for a_, b_ in islx) / Nx, 5)); dom.append(len(islx)); t = cp
    check(f"B={Bx} r={rx} seed {seed}: covered fraction at the six checkpoints", cover, e['cover'], 1e-9)
    check(f"B={Bx} r={rx} seed {seed}: number of active domains at the six checkpoints", dom, e['domains'])

# ---------------------------------------------------------------- S7 bulk thresholds at other B (full only)
if MODE == "full":
    emit("S7  bulk runs around the threshold at B = 8, 9, 11, 12 (threshold.py)")
    for (Bx, rx, seed), e in sorted(EXPECTED['threshold'].items()):
        basex = 1 << Bx; Nx = e['N']; T = e['T']; tmid = e['tmid']
        cx, nx = nf.ic_random(Nx, Bx, seed)
        nch1, _, _, _, _ = nf.run(cx, nx, rx, Bx, tmid, stop_dead=False); im = nf.islands(cx, nx, rx, Bx, steps=4000)
        nch2, _, _, _, _ = nf.run(cx, nx, rx, Bx, T - tmid - 4000, stop_dead=False); ie = nf.islands(cx, nx, rx, Bx, steps=4000)
        got = [round(100 * float(nch1[-10000:].mean()) / Nx, 4), round(100 * float(nch2[-10000:].mean()) / Nx, 4), len(im), len(ie),
               max((b_ - a_ + 1 for a_, b_ in im), default=0), max((b_ - a_ + 1 for a_, b_ in ie), default=0)]
        check(f"B={Bx} r={rx} seed {seed}: activity at t_mid and end, islands at t_mid and end, max widths",
              got, [e['act_mid'], e['act_end'], e['islands_mid'], e['islands_end'], e['maxw_mid'], e['maxw_end']], 1e-6, rel=True)

# ---------------------------------------------------------------- S8 reduced edge model
emit("S8  the reduced edge model (edge_model.py), B = 10, p = 0.5, seed 1, 4e6 steps")
import edge_model as EM
exp_edge = {376: 1, 380: 16, 382: 60, 383: 180, 384: 283, 385: 3033, 386: 5786, 400: 9872}
for rx, e in exp_edge.items():
    s_, _, _ = EM.run_edge(rx, 10, 48, 4_000_000, 0.5, 1)
    check(f"edge model r={rx}: successes", s_, e)

emit(f"\nSUMMARY {'full' if MODE == 'full' else 'quick'}: {npass} PASS, {nfail} FAIL")
OUT.close()
