#!/usr/bin/env python3
"""Every number quoted in the paper, from the stored results. Prints a
structured report; `python3 paper_numbers.py` runs everything,
`python3 paper_numbers.py json` also writes paper_numbers.json.
Nothing is truncated."""
import glob, json, os, sys
import numpy as np

OUT = {}


def vtau(B, r):
    e1 = (r - (1 << (B - 2))) / r
    return 1.0 / ((1 - e1) * (1 << (B - 3)) + 0.5)


def e1_of(B, r):
    return (r - (1 << (B - 2))) / r


# ---------------------------------------------------------------- fronts
def load_random_block():
    """front_B*.npz: v{r} = four front velocities (two seeds x two fronts)."""
    V = {}
    for B in (8, 9, 10, 11, 12):
        d = np.load(f'front_B{B}.npz', allow_pickle=True)
        for k in d.files:
            if k.startswith('v'):
                v = d[k].ravel().astype(float); v = v[~np.isnan(v)]
                V[(B, int(k[1:]))] = v
    return V


def load_developed():
    """front_dev_B8.jsonl and front_dev_long_B*.jsonl: per-front net velocities (T/10 -> T)."""
    D = {}
    for fn in sorted(glob.glob('front_dev_B*.jsonl')) + sorted(glob.glob('front_dev_long_B*.jsonl')):
        for l in open(fn):
            x = json.loads(l); B, r, T = x['B'], x['r'], x['T']
            vs = []
            quarter = x['N'] // 4 if 'N' in x else (x['k'] * (1 << B) + r) // 4
            for run in x['runs']:
                if 'hist' not in run:          # front_dev.py (4e6 steps): only the net velocities are stored
                    vs += [v for v in run.get('v_net', []) if v == v]; continue
                h = run['hist']
                for col in (1, 2):   # left, right front
                    pts = np.array([(e[0], e[col]) for e in h if e[col] is not None and e[0] >= T / 10], dtype=float)
                    if len(pts) >= 4 and pts[:, 1].max() < quarter - 64:   # a front that reached the antipode is not a measurement
                        vs.append(float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0]))
            if vs:
                key = (B, r)
                if key not in D or x.get('k', 16) > D[key]['k'] or (x.get('k', 16) == D[key]['k'] and T > D[key]['T']):
                    D[key] = dict(T=T, k=x.get('k', 16), v=np.array(vs, dtype=float), fronts=len(vs), file=fn)
    return D


def table_velocity():
    V = load_random_block(); D = load_developed()
    rows = {}
    print("TABLE II: front velocity per step.  'dev': developed-phase fronts (front_dev*, least-squares slope of each front's position over t >= T/10, mean +- sem over fronts, T in steps);")
    print("          'rb': random-block fronts (front.py, four fronts, mean and half the spread).  Scaled: v 2^(B-3).")
    for B in (8, 9, 10, 11, 12):
        rc = 3 * (1 << (B - 3))
        for dr in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 8):
            r = rc + dr
            if (B, r) in D:
                v = D[(B, r)]['v']; m = v.mean(); s = v.std(ddof=1) / np.sqrt(len(v))
                rows[(B, dr)] = dict(v=m, err=s, kind='dev', n=len(v), T=D[(B, r)]['T'], lo=v.min(), hi=v.max())
            elif (B, r) in V:
                v = V[(B, r)]; m = v.mean()
                rows[(B, dr)] = dict(v=m, err=(v.max() - v.min()) / 2, kind='rb', n=len(v), lo=v.min(), hi=v.max())
        for lab, r in (('u7', 7 * (1 << (B - 4))), ('u8', 1 << (B - 1))):
            if (B, r) in V:
                v = V[(B, r)]; rows[(B, lab)] = dict(v=v.mean(), err=(v.max() - v.min()) / 2, kind='rb', n=len(v), lo=v.min(), hi=v.max(), r=r)
    for key in sorted(rows, key=lambda k: (k[0], (k[1] if isinstance(k[1], int) else (100 if k[1] == 'u7' else 101)))):
        B, dr = key; x = rows[key]; sc = (1 << (B - 3))
        r = x.get('r', 3 * (1 << (B - 3)) + dr if isinstance(dr, int) else None)
        print(f"  B={B:2d} dr={str(dr):>3} r={r:5d} e1={e1_of(B, r):.4f}: v = {x['v']:.3e} +- {x['err']:.1e} ({x['kind']}, {x['n']} fronts{', T=%.0e' % x['T'] if 'T' in x else ''}) [min {x['lo']:.2e}, max {x['hi']:.2e}]  scaled {(('%.4f' % (x['v']*sc)) if x['v']*sc >= 0.01 else ('%.2e' % (x['v']*sc)))}  v/v_tau {x['v']/vtau(B, r):.3f}")
    OUT['velocity_rows'] = {f"{k[0]},{k[1]}": {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in x.items()} for k, x in rows.items()}
    # ratios and collapse
    print("\nRATIOS per B (random-block values at dr >= 1, developed at dr <= 0 where available):")
    for B in (8, 9, 10, 11, 12):
        g = lambda dr: rows[(B, dr)]['v']
        print(f"  B={B}: v(+1)/v(0) = {g(1)/g(0):.1f}, v(0)/v(-1) = {g(0)/g(-1):.2f}, v(+2)/v(+1) = {g(2)/g(1):.2f}" + (f", v(-1)/v(-2) = {g(-1)/g(-2):.1f}" if (B, -2) in rows else '') + (f", v(-2)/v(-3) = {g(-2)/g(-3):.2f}, v(-3)/v(-4) = {g(-3)/g(-4):.2f}" if (B, -3) in rows and (B, -4) in rows else ''))
    print("\nCOLLAPSE: scaled velocity v 2^(B-3) across B at fixed dr (min, max, max/min):")
    coll = {}
    for dr in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 8, 'u7', 'u8'):
        vals = [rows[(B, dr)]['v'] * (1 << (B - 3)) for B in (8, 9, 10, 11, 12) if (B, dr) in rows]
        Bs = [B for B in (8, 9, 10, 11, 12) if (B, dr) in rows]
        coll[str(dr)] = dict(B=Bs, scaled=vals)
        print(f"  dr={str(dr):>3}: " + ", ".join(f"B{B}: {v:.4f}" if v >= 0.01 else f"B{B}: {v:.2e}" for B, v in zip(Bs, vals)) + f"   max/min {max(vals)/min(vals):.2f}")
    OUT['collapse'] = coll
    print("\nv/v_tau with v_tau = 1/((1-e_1) 2^(B-3) + 1/2):")
    for dr in (1, 2, 3, 4, 8, 'u7', 'u8'):
        print(f"  dr={str(dr):>3}: " + ", ".join(f"B{B}: {rows[(B, dr)]['v']/vtau(B, rows[(B, dr)].get('r', 3*(1<<(B-3))+dr) if isinstance(dr, int) else rows[(B, dr)]['r']):.3f}" for B in (8, 9, 10, 11, 12) if (B, dr) in rows))
    # kink at u = 7
    print("\nKINK at u = 7 (random-block fronts): v at the residues around 7 2^(B-4):")
    for B in (8, 9, 10):
        u7 = 7 * (1 << (B - 4))
        ks = sorted(r for (b, r) in V if b == B and u7 - 8 * (1 << max(0, B - 10)) <= r <= u7 + 8 * (1 << max(0, B - 10)))
        print(f"  B={B}: " + ", ".join(f"r={r}: {V[(B, r)].mean():.3e}" for r in ks))
        if (B, u7) in V and (B, u7 + 1) in V:
            print(f"        step {u7} -> {u7+1}: {100*(V[(B, u7+1)].mean()/V[(B, u7)].mean()-1):+.1f}%")
        elif (B, u7 - 1) in V and (B, u7 + 1) in V:
            print(f"        {u7-1} -> {u7+1}: {100*(V[(B, u7+1)].mean()/V[(B, u7-1)].mean()-1):+.1f}%")
    return rows


# ---------------------------------------------------------------- creep chains
def creep_chains(rows):
    print("\nCREEP CHAINS (developed-phase fronts; scaled v 2^(B-3)):")
    D = load_developed(); V = load_random_block()
    chains = {}
    for name, pts in (("e1=0.3263 (r = 95, 190, 380)", [(8, 95), (9, 190), (10, 380)]),
                      ("e1=0.3298 (r = 191, 382, 764)", [(9, 191), (10, 382), (11, 764)]),
                      ("e1=0.3316 (r = 383, 766)", [(10, 383), (11, 766)]),
                      ("e1=1/3 (dr=0)", [(B, 3 * (1 << (B - 3))) for B in (8, 9, 10, 11, 12)]),
                      ("e1=1/3+2^-B (dr=+1)", [(B, 3 * (1 << (B - 3)) + 1) for B in (8, 9, 10, 11, 12)])):
        vals = []
        for B, r in pts:
            if (B, r) in D:
                v = D[(B, r)]['v']; vals.append((B, r, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), 'dev', len(v)))
            elif (B, r) in V:
                v = V[(B, r)]; vals.append((B, r, v.mean(), (v.max() - v.min()) / 2, 'rb', len(v)))
        chains[name] = [dict(B=B, r=r, v=m, err=s, kind=k, fronts=n, scaled=m * (1 << (B - 3))) for B, r, m, s, k, n in vals]
        print(f"  {name}: " + "  ->  ".join(f"B{B} r{r}: {m*(1<<(B-3)):.2e} +- {s*(1<<(B-3)):.1e} ({k})" for B, r, m, s, k, n in vals))
        if len(vals) > 1:
            print("      ratios: " + ", ".join(f"{vals[i+1][2]*(1<<(vals[i+1][0]-3))/(vals[i][2]*(1<<(vals[i][0]-3))):.3f}" for i in range(len(vals) - 1)))
    OUT['chains'] = chains


# ---------------------------------------------------------------- floor
def floor_table():
    print("\nPINNING FLOOR (floor_exact.jsonl):")
    res = {}
    for l in open('floor_exact.jsonl'):
        x = json.loads(l); B = x['B']; fl = x['floor']; rows = x['rows']
        below = [y for y in rows if y['r'] <= fl]; above = [y for y in rows if y['r'] > fl]
        exc = [(y['r'], y['max_word']) for y in below if not (y['max_word'] == y['r'] - 1)]
        ok_above = all(y['max_word'] == (1 << (B - 1)) + 1 for y in above)
        res[B] = dict(floor=fl, n_below=len(below), n_above=len(above), exceptions=exc, above_all_reach=ok_above)
        print(f"  B={B:2d}: floor {fl}; {len(below)} residues at or below it: reachable words exactly [0, r) except {exc} (the two residues without an ignition band); "
              f"{len(above)} residues above it: every one reaches exactly 2^(B-1)+1 = {(1<<(B-1))+1}: {ok_above}")
    OUT['floor'] = res


# ---------------------------------------------------------------- islands
def islands():
    print("\nISLAND CORES (island_cores.jsonl):")
    recs = [json.loads(l) for l in open('island_cores.jsonl')] if os.path.exists('island_cores.jsonl') else []
    for x in recs:
        print(f"  B={x['B']} seed={x['seed']} r={x['r']} N={x['N']}: {x['islands']} islands ({x['density_pct']}%), cores {x['core_sizes']}, mode {x['mode']}, bands {x['band_lo']}/{x['band_hi']} ({100*x['band_lo']/max(1,x['core_words']):.0f}% lower), |offset| max {x['offset_max']} mean {x['offset_mean']}, widths max {max(x['widths']) if x['widths'] else 0}")
    OUT['island_cores'] = recs
    print("\nISLAND LIFETIMES (island_life_B*.jsonl):")
    life = {}
    for fn in sorted(glob.glob('island_life_B*.jsonl')):
        recs = [json.loads(l) for l in open(fn)]
        for r in sorted(set(x['r'] for x in recs)):
            sub = [x for x in recs if x['r'] == r]; B = sub[0]['B']
            T = sub[0].get('T') or {8: 10_000_000, 10: 4_000_000}[B]   # run lengths of the stored campaigns (island_life.py did not record T)
            dead = sorted(x['tdead'] for x in sub if x['tdead'] > 0)
            exposure = sum(x['tdead'] if x['tdead'] > 0 else T for x in sub)
            grown = sum(1 for x in sub if (x.get('max_ext') or 0) >= 50)
            life[f"{B},{r}"] = dict(n=len(sub), dead=len(dead), lifetimes=dead, T=T, exposure=exposure, rate=len(dead) / exposure, grown50=grown)
            print(f"  B={B} r={r} (dr{r-3*(1<<(B-3)):+d}): {len(sub)} islands, {len(dead)} dead by {T:.0e} (lifetimes {dead}), exposure {exposure:.2e} island-steps, rate {len(dead)/exposure:.1e}; grown beyond 50 cells: {grown}")
    OUT['lifetimes'] = life


# ---------------------------------------------------------------- bulk runs
def bulk():
    print("\nBULK RUNS (threshold_B*.jsonl): activity in windows; islands at t_mid and at the end")
    out = {}
    for fn in sorted(glob.glob('threshold_B*.jsonl')):
        for rec in (json.loads(l) for l in open(fn)):
            e = np.array(rec['edges']); mid = np.sqrt(e[:-1] * e[1:]); rho = np.array(rec['rho'], dtype=float)
            B, r, seed = rec['B'], rec['r'], rec['seed']
            # activity at 1e2, 6e3, 1e5, 1e6, end; minimum after 1e3
            def at(t):
                i = int(np.argmin(np.abs(mid - t))); return float(rho[i])
            later = rho[mid > 1e3]; imin = int(np.nanargmin(later)); tmin = float(mid[mid > 1e3][imin])
            key = f"{B},{r},{seed}"
            out[key] = dict(N=rec['N'], T=rec['T'], act_1e2=at(1e2), act_6e3=at(6e3), act_1e5=at(1e5), act_end=float(rho[~np.isnan(rho)][-1]),
                            act_min=float(np.nanmin(later)), t_min=tmin, islands_mid=rec['islands_mid'], islands_end=rec['islands_end'],
                            wmax_mid=max(rec['widths_mid']) if rec['widths_mid'] else 0, wmax_end=max(rec['widths_end']) if rec['widths_end'] else 0,
                            act_mid=rec['act_mid'], act_end_rec=rec['act_end'])
            x = out[key]
            print(f"  B={B} r={r} (dr{r-rec['rc']:+d}) seed {seed} N={rec['N']} T={rec['T']:.1e}: act 1e2 {x['act_1e2']:.1f}%, 6e3 {x['act_6e3']:.2f}%, min {x['act_min']:.2f}% at {x['t_min']:.0e}, 1e5 {x['act_1e5']:.2f}%, end {x['act_end']:.2f}%; "
                  f"islands {x['islands_mid']} (w<={x['wmax_mid']}) -> {x['islands_end']} (w<={x['wmax_end']})")
    OUT['bulk'] = out


# ---------------------------------------------------------------- quench
def quench():
    print("\nQUENCH (quench_B*.jsonl): coverage and domains at the checkpoints; hole nucleation")
    out = {}
    for fn in sorted(glob.glob('quench_B*.jsonl')):
        for rec in (json.loads(l) for l in open(fn)):
            B, r, seed = rec['B'], rec['r'], rec['seed']
            cps = [(c['t'], c['domains'], 100 * c['cover']) for c in rec['checkpoints']]
            N = rec['N']
            key = f"{B},{r},{seed}"
            # holes = domains (ring: number of gaps equals number of domains); nucleation rate from the rise of the count
            t0, d0 = cps[0][0], cps[0][1]; t1, d1 = cps[-1][0], cps[-1][1]
            nuc = (d1 - d0) / ((t1 - t0) * N) if t1 > t0 else float('nan')
            out[key] = dict(N=N, T=rec['T'], cps=cps, nucleation_per_cell_step=nuc, v_retreat=rec.get('v_retreat'))
            print(f"  B={B} r={r} (dr{r-3*(1<<(B-3)):+d}) seed {seed} N={N}: " + ", ".join(f"{c[2]:.1f}% ({c[1]} dom.) at {c[0]:.0e}" for c in cps) + f"; holes {d0} -> {d1}: {nuc:.1e} per cell-step; v_retreat {rec.get('v_retreat'):.1e}")
    OUT['quench'] = out
    if glob.glob('quench_long_B*.jsonl'):
        print("\nQUENCH LONG (quench_long_B*.jsonl):")
        for fn in sorted(glob.glob('quench_long_B*.jsonl')):
            for rec in (json.loads(l) for l in open(fn)):
                h = rec['hist']
                print(f"  B={rec['B']} r={rec['r']} N={rec['N']} T={rec['T']:.0e}: " + ", ".join(f"{x[2]:.2f}% ({int(x[1])}) at {x[0]:.0e}" for x in h[::max(1, len(h)//8)]) + f"; final {h[-1][2]:.2f}% ({int(h[-1][1])} domains)")


# ---------------------------------------------------------------- phase words
def phase():
    if os.path.exists('phase_words.jsonl'):
        print("\nACTIVE PHASE (phase_words.jsonl):")
        recs = [json.loads(l) for l in open('phase_words.jsonl')]
        for x in recs:
            print(f"  B={x['B']} r={x['r']}: changing {x['changing_pct']}%, on {x['on_pct']}%, within 32 of 2^(B-1): {x['near_half_pct']}%, of 3 2^(B-2): {x['near_three_quarters_pct']}%")
        OUT['phase'] = recs


def natural():
    if os.path.exists('front_natural_B8.jsonl'):
        print("\nNATURAL SEA (front_natural_B8.jsonl):")
        for rec in (json.loads(l) for l in open('front_natural_B8.jsonl')):
            print(f"  B={rec['B']} r={rec['r']} T={rec['T']:.0e}: natural {rec['v_natural_mean']:.3e} +- {rec['v_natural_sem']:.1e}, uniform {rec['v_uniform_mean']:.3e} +- {rec['v_uniform_sem']:.1e}; sea histograms {[x['sea_hist'] for x in rec['runs']][:1]}")


def seeds():
    print("\nSEED SURVIVAL (long_survival*.jsonl): alive at T and growing")
    for fn in sorted(glob.glob('long_survival_B*.jsonl')):
        R = [json.loads(l) for l in open(fn)]
        T = R[0]['T']; alive = [x for x in R if x['tdead'] == 0]
        print(f"  {fn}: {len(R)} seeds, T={T:.0e}: alive {len(alive)}; latest death {max(x['tdead'] for x in R)}; extents at T {sorted(x['hist'][-1][1] for x in alive)[:10]}{'...' if len(alive)>10 else ''}")


if __name__ == '__main__':
    rows = table_velocity(); creep_chains(rows); floor_table(); islands(); bulk(); quench(); phase(); natural(); seeds()
    if 'json' in sys.argv:
        json.dump(OUT, open('paper_numbers.json', 'w'), indent=1, default=float)
        print("\nwritten paper_numbers.json")
