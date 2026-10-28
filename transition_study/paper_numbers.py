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
    return 1.0 / ((1 - e1) * ((1 << (B - 3)) + 0.5))   # Eq. (4): mean wait (1 - e1)(2^(B-3) + 1/2)


def e1_of(B, r):
    return (r - (1 << (B - 2))) / r


# ---------------------------------------------------------------- fronts
def load_random_block():
    """front_B*.npz: v{r} = four front velocities (two seeds x two fronts)."""
    V = {}
    for B in (8, 9, 10, 11, 12, 13):
        if not os.path.exists(f'front_B{B}.npz'): continue
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
    for B in (8, 9, 10, 11, 12, 13):
        rc = 3 * (1 << (B - 3))
        for dr in ([-16, -14, -12, -10] if B == 8 else []) + [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8]:
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
    for (B, dr), x in rows.items():
        r = x.get('r', 3 * (1 << (B - 3)) + dr if isinstance(dr, int) else None)
        x['r'] = r; x['e1'] = e1_of(B, r); x['scaled'] = x['v'] * (1 << (B - 3)); x['v_over_vtau'] = x['v'] / vtau(B, r)
    OUT['velocity_rows'] = {f"{k[0]},{k[1]}": {kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in x.items()} for k, x in rows.items()}
    # the two preparations where they overlap (dr = -1, 0, +1)
    print("\nTWO PREPARATIONS at dr = -1, 0, +1: random-block (four fronts: mean, min, max) against developed-phase (eight fronts: mean, min, max), ratio of means")
    tp = {}
    for dr in (-1, 0, 1):
        for B in (8, 9, 10, 11, 12):
            r = 3 * (1 << (B - 3)) + dr
            if (B, r) in V and (B, r) in D:
                v = V[(B, r)]; d = D[(B, r)]['v']
                tp[f"{B},{dr}"] = dict(rb_mean=float(v.mean()), rb_min=float(v.min()), rb_max=float(v.max()), dev_mean=float(d.mean()), dev_min=float(d.min()), dev_max=float(d.max()), ratio=float(v.mean() / d.mean()),
                                       ranges_overlap=bool(v.min() <= d.max() and d.min() <= v.max()))
                x = tp[f"{B},{dr}"]
                print(f"  dr={dr:+d} B={B}: rb {x['rb_mean']:.3e} [{x['rb_min']:.2e}, {x['rb_max']:.2e}], dev {x['dev_mean']:.3e} [{x['dev_min']:.2e}, {x['dev_max']:.2e}], ratio {x['ratio']:.2f}, ranges overlap {x['ranges_overlap']}")
    OUT['two_preparations'] = tp
    # ratios and collapse
    print("\nRATIOS per B (developed-phase values at dr <= 1 where available, random-block above):")
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
    print("\nv/v_tau with v_tau = 1/((1-e_1)(2^(B-3) + 1/2)):")
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


# ---------------------------------------------------------------- fixed excitability above threshold
def fixed_e1():
    """v/v_tau at the residues that share an excitability across word lengths (r doubles with B); random-block fronts,
    dr >= 2 only (dr = 1 is the anomalous neighbourhood of the threshold and is listed separately)."""
    from fractions import Fraction
    V = load_random_block(); D = load_developed()
    print("\nFIXED EXCITABILITY above threshold (random-block fronts, dr >= 2): v/v_tau at the residues r = r0 2^(B-B0)")
    groups = {}
    for (B, r), v in V.items():
        dr = r - 3 * (1 << (B - 3))
        if dr < 2: continue
        e = Fraction(r - (1 << (B - 2)), r)
        groups.setdefault(e, []).append((B, r, dr, float(v.mean()) / vtau(B, r), float((v.max() - v.min()) / 2) / vtau(B, r)))
    out = {}
    for e in sorted(groups):
        g = sorted(groups[e])
        if len(g) < 2: continue
        key = f"{float(e):.4f}"
        out[key] = dict(e1=float(e), dist=float(e) - 1 / 3, points=[dict(B=B, r=r, dr=dr, G=G, spread=s) for B, r, dr, G, s in g],
                        G_mean=float(np.mean([x[3] for x in g])), G_min=min(x[3] for x in g), G_max=max(x[3] for x in g))
        print(f"  e1={float(e):.4f} (e1-1/3={float(e)-1/3:+.4f}): " + ", ".join(f"B{B} r{r} (dr{dr:+d}): {G:.3f}+-{s:.3f}" for B, r, dr, G, s in g) + f"   mean {out[key]['G_mean']:.3f}")
    print("  dr=+1 (developed-phase fronts), v/v_tau: " + ", ".join(f"B{B}: {D[(B, 3*(1<<(B-3))+1)]['v'].mean()/vtau(B, 3*(1<<(B-3))+1):.3f}" for B in (8, 9, 10, 11, 12) if (B, 3*(1<<(B-3))+1) in D))
    print("  closest points above threshold (dr=+2), v/v_tau and e1-1/3: " + ", ".join(f"B{B}: {V[(B, 3*(1<<(B-3))+2)].mean()/vtau(B, 3*(1<<(B-3))+2):.3f} at {e1_of(B, 3*(1<<(B-3))+2)-1/3:.4f}" for B in (8, 9, 10, 11, 12) if (B, 3*(1<<(B-3))+2) in V))
    OUT['fixed_e1'] = out


# ---------------------------------------------------------------- creep chains
def creep_chains(rows):
    print("\nCREEP CHAINS (developed-phase fronts; scaled v 2^(B-3)):")
    D = load_developed(); V = load_random_block()
    chains = {}
    for name, pts in (("e1=0.3263 (r = 95, 190, 380, 760)", [(8, 95), (9, 190), (10, 380), (11, 760)]),
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
            # time from which the activity stays within 5% of its final value (end of the first such window), for the runs that reach the bulk phase
            fin = out[key]['act_end']; ok = np.array([bool(np.all(rho[i:] >= 0.95 * fin)) for i in range(len(rho))])
            out[key]['t_95'] = float(e[1:][int(np.argmax(ok))]) if ok.any() and fin > 40 else None
            x = out[key]
            print(f"  B={B} r={r} (dr{r-rec['rc']:+d}) seed {seed} N={rec['N']} T={rec['T']:.1e}: act 1e2 {x['act_1e2']:.1f}%, 6e3 {x['act_6e3']:.2f}%, min {x['act_min']:.2f}% at {x['t_min']:.0e}, 1e5 {x['act_1e5']:.2f}%, end {x['act_end']:.2f}%; "
                  f"islands {x['islands_mid']} (w<={x['wmax_mid']}) -> {x['islands_end']} (w<={x['wmax_end']})" + (f"; within 5% of final from t={x['t_95']:.1e}" if x['t_95'] else ''))
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
        print("\nQUENCH LONG (quench_long_B*.jsonl): minimum coverage and when, coverage and holes at the end, holes after the relaxation (t >= 2e7)")
        ql = {}
        for fn in sorted(glob.glob('quench_long_B*.jsonl')):
            for rec in (json.loads(l) for l in open(fn)):
                h = rec['hist']; cov = np.array([x[2] for x in h]); dom = np.array([int(x[1]) for x in h]); tt = np.array([x[0] for x in h])
                j = int(cov.argmin()); late = tt >= 2e7
                back = tt[(tt > tt[j]) & (cov >= 99.95)]
                i7 = int(np.searchsorted(tt, 1e7))
                d = dict(N=rec['N'], T=rec['T'], min_cover=float(cov[j]), t_min=float(tt[j]), cover_1e7=float(cov[i7]), holes_1e7=int(dom[i7]), final_cover=float(cov[-1]), final_holes=int(dom[-1]),
                         holes_late=[int(dom[late].min()), int(dom[late].max())], cover_late=[float(cov[late].min()), float(cov[late].max())],
                         t_back_to_100=float(back[0]) if back.size else None)
                ql[f"{rec['B']},{rec['r']},{rec['seed']}"] = d
                print(f"  B={rec['B']} r={rec['r']} (dr{rec['r']-3*(1<<(rec['B']-3)):+d}, e1={e1_of(rec['B'], rec['r']):.3f}) N={rec['N']} T={rec['T']:.0e}: min {d['min_cover']:.2f}% at {d['t_min']:.0e}; at 1e7 {d['cover_1e7']:.2f}% ({d['holes_1e7']} holes); "
                      f"back to >= 99.95% at {d['t_back_to_100']}; final {d['final_cover']:.2f}% with {d['final_holes']} holes; t >= 2e7: {d['cover_late'][0]:.2f}-{d['cover_late'][1]:.2f}%, {d['holes_late'][0]}-{d['holes_late'][1]} holes")
        OUT['quench_long'] = ql
    if os.path.exists('hole_words.jsonl'):
        print("\nHOLES of the quenched phase at T (hole_words.jsonl): lengths, words of the hole cells, lower bounds on openings and closings from the stored hole counts")
        hw = {}
        for rec in (json.loads(l) for l in open('hole_words.jsonl')):
            t = rec['totals']; per = rec['chunks'] * rec['chunk'] / 1e7
            hw[f"{rec['B']},{rec['r']},{rec['seed']}"] = dict(N=rec['N'], t=rec['t'], holes=t['holes'], lengths=rec['lengths'], cells=t['cells'], sea_range=t['sea_range'], lower_band=t['lower_band'], upper_band=t['upper_band'], other=t['other'], on=t['on'],
                                                              openings_per_1e7=rec['count_increases'] / per, closings_per_1e7=rec['count_decreases'] / per)
            print(f"  B={rec['B']} r={rec['r']} (e1={e1_of(rec['B'], rec['r']):.3f}) t={rec['t']:.0e} N={rec['N']}: {t['holes']} holes of {min(rec['lengths']) if rec['lengths'] else 0}-{max(rec['lengths']) if rec['lengths'] else 0} cells ({t['cells']} cells, {100*t['cells']/rec['N']:.2f}% of N); "
                  f"words in [0,r): {t['sea_range']}, in the bands: {t['lower_band']+t['upper_band']}, other: {t['other']}, on: {t['on']}; hole count up {rec['count_increases']} / down {rec['count_decreases']} in {rec['chunks']} chunks = {rec['count_increases']/per:.1f} openings and {rec['count_decreases']/per:.1f} closings per 1e7 steps (lower bounds)")
        OUT['holes'] = hw


# ---------------------------------------------------------------- domain density, the floor models, the reduced edge model
def domain_density():
    if os.path.exists('domain_density.jsonl'):
        print("\nDOMAIN DENSITY (domain_density.jsonl): activity inside the growing domains at r_c - 1 and in the bulk at r_c + 1")
        out = {}
        for x in (json.loads(l) for l in open('domain_density.jsonl')):
            out[f"{x['B']},{x['r']}"] = dict(N=x['N'], T=x['T'], domains=x['domains'], covered=x['covered'], density=x['density'], ones=x['ones'])
            print(f"  B={x['B']} r={x['r']} (dr{x['r']-3*(1<<(x['B']-3)):+d}) T={x['T']:.0e}: {x['domains']} domains covering {100*x['covered']:.1f}%; cells changing per step inside {x['density']}%, on {x['ones']}%")
        OUT['domain_density'] = out


def floor_models():
    if os.path.exists('floor_models.jsonl'):
        print("\nFLOOR MODELS (floor_models.jsonl): two and three exact sea cells with free inputs at the ends; largest reachable word at floor-1, floor, floor+1")
        out = []
        for x in (json.loads(l) for l in open('floor_models.jsonl')):
            out.append(dict(cells=x['cells'], B=x['B'], r=x['r'], floor=x['floor'], max_word=x['max_word'], states=x['states']))
            print(f"  {x['cells']} cells, B={x['B']} r={x['r']} (floor{x['r']-x['floor']:+d}): largest reachable word {x['max_word']} ({'= r-1, sea range only' if x['max_word'] == x['r'] - 1 else '= 2^(B-1)+1' if x['max_word'] == (1 << (x['B'] - 1)) + 1 else ''}), {x['states']} reachable states")
        OUT['floor_models'] = out


def edge_model():
    if not os.path.exists('edge_model.jsonl'): return
    print("\nREDUCED EDGE MODEL (edge_model.jsonl): successes (sea words reaching the pinned band) per run, 47 sea cells beside a random edge")
    R = [json.loads(l) for l in open('edge_model.jsonl')]
    out = {}
    tab = {(x['p'], x['r']): x for x in R if x['kind'] == 'table'}
    for p in (0.5, 0.7):
        rows = sorted((r, x) for (pp, r), x in tab.items() if pp == p)
        s = {r: float(np.mean(x['successes'])) for r, x in rows}
        print(f"  B=10, p={p}, T=4e6, seeds 1-3, mean successes: " + ", ".join(f"r={r}: {s[r]:.1f}" for r, x in rows))
        f = dict(v0_over_vm1=s[384] / s[383], closure_vm1_over_vm2=s[383] / s[382], jump_v1_over_v0=s[385] / s[384], v2_over_v1=s[386] / s[385],
                 below=[s[382] / s[381], s[381] / s[380], s[380] / s[378], s[378] / s[376]])
        out[f"table_p{p}"] = dict(successes={str(r): x['successes'] for r, x in rows}, factors=f)
        print(f"      factors: dr=0 / dr=-1: {f['v0_over_vm1']:.2f}; closure dr=-1 / dr=-2: {f['closure_vm1_over_vm2']:.2f}; jump dr=+1 / dr=0: {f['jump_v1_over_v0']:.1f}; dr=+2 / dr=+1: {f['v2_over_v1']:.2f}; "
              f"ratios 382/381, 381/380, 380/378, 378/376: " + ", ".join(f"{v:.2f}" for v in f['below']))
    sc = [x for x in R if x['kind'] == 'scaling']
    print("  scaling (p=0.5, seeds 1-2, T = 4e6 2^(B-10)): successes per clock period (successes r / T) at fixed dr:")
    per = {}
    for x in sc:
        per[(x['B'], x['dr'] if abs(x['dr']) <= 2 else x['dr'] // max(1, (1 << x['B']) // 1024))] = float(np.mean(x['successes'])) * x['r'] / x['T']
    for dr in (-2, -1, 0, 1, 2, 4, 8, 16, 32, 64):
        print(f"      dr={dr:+d}: " + ", ".join(f"B{B}: {per[(B, dr)]:.3f}" for B in (9, 10, 11, 12) if (B, dr) in per) + (f"   (dr scaled with 2^(B-10) for |dr| > 2)" if abs(dr) > 2 else ''))
    out['scaling_per_clock'] = {f"{B},{dr}": v for (B, dr), v in per.items()}
    OUT['edge_model'] = out


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
        out = {}
        for rec in (json.loads(l) for l in open('front_natural_B8.jsonl')):
            dead = [x for x in rec['runs'] if x['natural_sea_dead']]
            H = np.array([x['sea_hist'] for x in dead], dtype=float); pooled = H.sum(axis=0); exp = pooled.sum() / 8; chi2 = float(((pooled - exp) ** 2 / exp).sum())
            nat = np.array([x['natural']['v'] for x in dead]).ravel(); uni = np.array([x['uniform']['v'] for x in rec['runs']]).ravel()
            out[str(rec['r'])] = dict(T=rec['T'], N=rec['N'], seeds_dead=[x['seed'] for x in dead], natural_mean=float(nat.mean()), natural_sem=float(nat.std(ddof=1) / np.sqrt(nat.size)), natural_n=int(nat.size),
                                       uniform_mean=float(uni.mean()), uniform_sem=float(uni.std(ddof=1) / np.sqrt(uni.size)), uniform_n=int(uni.size), pooled_hist=pooled.astype(int).tolist(), chi2_7dof=chi2)
            print(f"  B={rec['B']} r={rec['r']} T={rec['T']:.0e} N={rec['N']}: dead seas in seeds {out[str(rec['r'])]['seeds_dead']}; natural {nat.mean():.3e} +- {out[str(rec['r'])]['natural_sem']:.1e} ({nat.size} fronts), "
                  f"uniform {uni.mean():.3e} +- {out[str(rec['r'])]['uniform_sem']:.1e} ({uni.size} fronts); pooled phase histogram (eighths of [0, r_sea)) {pooled.astype(int).tolist()}, chi2 {chi2:.1f} (7 dof)")
        OUT['natural'] = out


def word_fronts(rows):
    """word_front.jsonl: the word-defined front at T against the changed-cell front at T and the slope-based displacement."""
    if not os.path.exists('word_front.jsonl'): return
    print("\nWORD FRONT at T (word_front.jsonl): mean over eight fronts of the displacement of the outermost cell with a word outside [0, r); the changed-cell front at T; the halo; slope x T from Table II")
    out = {}
    for rec in (json.loads(l) for l in open('word_front.jsonl')):
        B, r = rec['B'], rec['r']; dr = r - 3 * (1 << (B - 3))
        wf = np.array([run['word_front'] for run in rec['runs']], dtype=float).ravel()
        cf = np.array([run['changed_front'] for run in rec['runs']], dtype=float).ravel()
        T = rec['runs'][0]['t']; row = rows.get((B, dr))
        d = dict(T=T, fronts=int(wf.size), word_mean=float(wf.mean()), word_sem=float(wf.std(ddof=1) / np.sqrt(wf.size)), changed_mean=float(cf.mean()), changed_sem=float(cf.std(ddof=1) / np.sqrt(cf.size)),
                 halo_mean=float((cf - wf).mean()), halo_min=float((cf - wf).min()), halo_max=float((cf - wf).max()),
                 slope_x_T=float(row['v'] * T) if row else None, slope_x_T_err=float(row['err'] * T) if row else None, v_word=float(wf.mean() / T), v_word_sem=float(wf.std(ddof=1) / np.sqrt(wf.size) / T))
        out[f"{B},{r}"] = d
        d['word_fronts'] = [run['word_front'] for run in rec['runs']]; d['changed_fronts'] = [run['changed_front'] for run in rec['runs']]
        print(f"  B={B} r={r} (dr{dr:+d}, e1={e1_of(B, r):.4f}) T={T:.0e}: word front {d['word_mean']:.1f} +- {d['word_sem']:.1f} cells (v_word {d['v_word']:.2e} +- {d['v_word_sem']:.1e}); changed-cell front {d['changed_mean']:.1f} +- {d['changed_sem']:.1f}; halo {d['halo_mean']:.1f} [{d['halo_min']:.0f}, {d['halo_max']:.0f}]"
              + (f"; slope x T {d['slope_x_T']:.1f} +- {d['slope_x_T_err']:.1f}" if row else f"; per front: words {d['word_fronts']}, changed cells {d['changed_fronts']} (no Table II row: control run)"))
    OUT['word_front'] = out


def seeds():
    print("\nSEED SURVIVAL (long_survival*.jsonl): alive at T, latest death, extents at T, growth of the extent (least squares over t >= T/10, per front = half the slope)")
    out = {}
    for fn in sorted(glob.glob('long_survival_B*.jsonl')):
        R = [json.loads(l) for l in open(fn)]
        T = R[0]['T']; alive = [x for x in R if x['tdead'] == 0]
        ext = sorted(x['hist'][-1][1] for x in alive)
        rates = []
        for x in alive:
            H = np.array([(h[0], h[1]) for h in x['hist'] if h[0] >= T / 10], dtype=float)
            if len(H) >= 4: rates.append(float(np.polyfit(H[:, 0], H[:, 1], 1)[0]) / 2)
        d = dict(n=len(R), T=T, alive=len(alive), frac_alive=len(alive) / len(R), latest_death=max(x['tdead'] for x in R), ext_min=min(ext) if ext else None, ext_max=max(ext) if ext else None,
                 rate_per_front_min=min(rates) if rates else None, rate_per_front_max=max(rates) if rates else None, rate_per_front_mean=float(np.mean(rates)) if rates else None)
        out[fn] = d
        print(f"  {fn}: {d['n']} seeds, T={T:.0e}: alive {d['alive']} ({d['frac_alive']:.3f}); latest death {d['latest_death']:.2e}; extents at T {d['ext_min']}-{d['ext_max']}; "
              f"growth per front {d['rate_per_front_min']:.2e}-{d['rate_per_front_max']:.2e} (mean {d['rate_per_front_mean']:.2e})" if alive else f"  {fn}: {d['n']} seeds: none alive")
    # survival curves and interval hazards (deaths per island-step) of the seed populations and of the transplanted islands
    print("\nSURVIVAL: P(alive at t) and the hazard (deaths per island-step) in each decade, seeds (long_survival) and transplants (island_life, time since transplant)")
    def hazards(td, T, bounds):
        td = np.array(td, dtype=float); res = []
        for t1, t2 in bounds:
            if t1 >= T: res.append(None); continue
            t2 = min(t2, T)
            alive1 = (td == 0) | (td > t1); deaths = int(((td > t1) & (td <= t2)).sum())
            ends = np.where(td == 0, T, td)[alive1]; expo = float(np.sum(np.minimum(ends, t2) - t1))
            res.append(dict(t1=t1, t2=t2, alive_at_start=int(alive1.sum()), deaths=deaths, exposure=expo, hazard=deaths / expo if expo else None))
        return res
    bounds = [(1e3, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7), (1e7, 1e8)]; times = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    sv = {}
    for fn in sorted(glob.glob('long_survival_B*.jsonl')):
        R = [json.loads(l) for l in open(fn)]; T = R[0]['T']; td = np.array([x['tdead'] for x in R], dtype=float)
        P = [float(((td == 0) | (td > t)).mean()) if t <= T else None for t in times]
        sv[fn] = dict(n=len(R), T=T, P_alive=P, hazards=hazards(td, T, bounds))
        print(f"  {fn}: n={len(R)}, P(alive) at 1e3..1e8: " + ", ".join('-' if v is None else f"{v:.3f}" for v in P) + "; hazard per decade: " +
              ", ".join('-' if h is None else f"{h['deaths']}/{h['exposure']:.1e}={h['hazard']:.1e}" for h in sv[fn]['hazards']))
    for fn in sorted(glob.glob('island_life_B*.jsonl')):
        R = [json.loads(l) for l in open(fn)]; Bx = R[0]['B']
        rs = sorted(set(x['r'] for x in R)); Tt = 1e7 if Bx == 8 else 4e6
        for sel, name in [(rs, 'all r')] + [([rx], f"r={rx}") for rx in rs]:
            td = np.array([x['tdead'] for x in R if x['r'] in sel], dtype=float)
            hz = hazards(td, Tt, bounds); key = f"{fn}:{name}"
            sv[key] = dict(n=int(td.size), T=Tt, hazards=hz, deaths=int((td > 0).sum()), rate_overall=float((td > 0).sum() / np.sum(np.where(td == 0, Tt, td))))
            print(f"  {fn} {name}: n={td.size}, deaths {sv[key]['deaths']}, overall {sv[key]['rate_overall']:.1e}; hazard per decade since transplant: " + ", ".join('-' if h is None else f"{h['deaths']}/{h['exposure']:.1e}={h['hazard']:.1e}" for h in hz))
    OUT['survival'] = sv
    print("\nSPREAD (spread_B*_w32_a.npz): fraction alive at 1e3, 1e5 and T; extent at 1e5 and T of the runs alive at T")
    for fn in sorted(glob.glob('spread_B*_w32_a.npz')):
        z = np.load(fn); t = z['times']; al = z['alive']; ex = z['ext']; last = al[:, -1] == 1
        i3, i5 = int(np.searchsorted(t, 1000)), int(np.searchsorted(t, 100000))
        B_, r_ = (int(s[1:]) for s in fn.split('_')[1:3])
        d = dict(B=B_, r=r_, runs=int(al.shape[0]), T=int(t[-1]), alive_1e3=float(al[:, i3].mean()), alive_1e5=float(al[:, i5].mean()), alive_T=float(last.mean()),
                 ext_1e5_of_survivors=[int(ex[last, i5].min()), int(ex[last, i5].max())] if last.any() else None, ext_T_of_survivors=[int(ex[last, -1].min()), int(ex[last, -1].max())] if last.any() else None)
        out[fn] = d
        print(f"  B={B_} r={r_} (dr{r_-3*(1<<(B_-3)):+d}): {d['runs']} runs, T={d['T']:.0e}: alive {d['alive_1e3']:.3f} at 1e3, {d['alive_1e5']:.3f} at 1e5, {d['alive_T']:.3f} at T; survivors' extent {d['ext_1e5_of_survivors']} at 1e5, {d['ext_T_of_survivors']} at T")
    OUT['seeds'] = out


# ---------------------------------------------------------------- paired blocks: the same prepared blocks at two residues
def paired_blocks(pairs=(((10, 379), (10, 378)),)):
    """front_dev_long records: per-seed sums of the two front slopes (t >= T/10) at two residues run from the same prepared blocks."""
    import glob as _g
    print("\nPAIRED BLOCKS (same prepared blocks, same seeds, two residues): per-seed sum of the two front slopes, t >= T/10")
    out = {}
    for (B1, r1), (B2, r2) in pairs:
        vals = {}
        for (B, r) in ((B1, r1), (B2, r2)):
            for fn in sorted(_g.glob(f'front_dev_long_B{B}.jsonl')):
                for l in open(fn):
                    x = json.loads(l)
                    if x['r'] != r: continue
                    T = x['T']
                    for run in x['runs']:
                        h = run['hist']; sl = 0.0
                        for col in (1, 2):
                            pts = np.array([(e[0], e[col]) for e in h if e[col] is not None and e[0] >= T / 10], dtype=float)
                            sl += float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0])
                        vals.setdefault((B, r), {})[run['seed']] = sl
        a, b = vals.get((B1, r1), {}), vals.get((B2, r2), {})
        seeds = sorted(set(a) & set(b))
        ratios = [b[sd] / a[sd] for sd in seeds]
        out[f"{B1},{r1} vs {B2},{r2}"] = dict(seeds=seeds, sums_a=[a[sd] for sd in seeds], sums_b=[b[sd] for sd in seeds], ratios=ratios)
        print(f"  B={B1} r={r1} (dr{r1-3*(1<<(B1-3)):+d}) vs r={r2} (dr{r2-3*(1<<(B2-3)):+d}): seeds {seeds}: sums " +
              ", ".join(f"{a[sd]:.2e} -> {b[sd]:.2e} (x{b[sd]/a[sd]:.1f})" for sd in seeds) + f"; ratios {min(ratios):.1f} to {max(ratios):.1f}")
    OUT['paired_blocks'] = out


# ---------------------------------------------------------------- alternation statistics of the developed phase
def alternation():
    """alternation.jsonl: P(a bulk cell changed state at each of the last m steps) against p^m."""
    if not os.path.exists('alternation.jsonl'): return
    print("\nALTERNATION IN THE DEVELOPED PHASE (alternation.jsonl): P(changed at each of the last m steps) vs random p^m")
    out = []
    for rec in (json.loads(l) for l in open('alternation.jsonl')):
        out.append(rec)
        print(f"  B={rec['B']} r={rec['r']} N={rec['N']} T={rec['T']}: on {rec['on']:.3f}, change rate {rec['change_rate']:.3f}; " +
              ", ".join(f"m={x['m']}: {x['prob']:.3g} ({x['random']:.3g})" for x in rec['alternation']))
    OUT['alternation'] = out


if __name__ == '__main__':
    rows = table_velocity(); fixed_e1(); creep_chains(rows); paired_blocks(); word_fronts(rows); floor_table(); floor_models(); edge_model(); islands(); bulk(); domain_density(); quench(); phase(); natural(); seeds(); alternation()
    if 'json' in sys.argv:
        json.dump(OUT, open('paper_numbers.json', 'w'), indent=1, default=float)
        print("\nwritten paper_numbers.json")
