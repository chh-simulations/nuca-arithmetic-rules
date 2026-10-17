#!/usr/bin/env python3
"""Tables from the stored results: front velocities (front_B*.npz),
threshold runs (threshold_B*.jsonl), creep (creep_B*.jsonl), island
lifetimes (island_life_B*.jsonl).  Prints everything; truncates nothing."""
import glob, json, os
import numpy as np


def bound(B, r):
    e1 = (r - (1 << (B - 2))) / r
    return 1.0 / ((1 - e1) * (1 << (B - 3)) + 1)


def fronts():
    print("FRONT VELOCITIES (cells/step; mean over the four fronts of two seeds, range in brackets); bound = 1/((1-e_1) 2^(B-3) + 1)")
    print("  B     r    dr   u=r/2^(B-4)    e_1        v         [min, max]        bound     v/bound   hit")
    for fn in sorted(glob.glob("front_B*.npz")):
        B = int(fn[7:-4].split('b')[0]) if 'b' in fn[7:-4] else int(fn[7:-4])
        d = np.load(fn, allow_pickle=True)
        rc = 3 * (1 << (B - 3))
        for key in sorted((k for k in d.files if k.startswith('v')), key=lambda k: int(k[1:])):
            r = int(key[1:]); vs = d[key].ravel(); vs = vs[~np.isnan(vs)]
            e1 = (r - (1 << (B - 2))) / r
            # did the fronts hit the antipode? (final hi near N/4 from b)
            runs = d['r' + key[1:]]; N = 24 * (1 << B) + r
            hit = any(int(run[2][-1]) >= 3 * N // 4 + N // 4 - 64 for run in runs)
            print(f"  {B:2d}  {r:5d}  {r-rc:+4d}   {r/(1<<(B-4)):8.4f}   {e1:.5f}   {vs.mean():.3e}  [{vs.min():.2e}, {vs.max():.2e}]  {bound(B, r):.3e}   {vs.mean()/bound(B, r):.3f}   {'*' if hit else ''}")


def thresholds():
    for fn in sorted(glob.glob("threshold_B*.jsonl")):
        recs = [json.loads(l) for l in open(fn)]
        print(f"\nTHRESHOLD RUNS {fn}: activity (% of N) in log windows; islands (count, max width) at t_mid and at the end")
        for rec in recs:
            e = np.array(rec['edges']); mid = np.sqrt(e[:-1] * e[1:]); rho = rec['rho']
            sel = list(range(0, len(mid), 6)) + [len(mid) - 1]
            print(f"  B={rec['B']} r={rec['r']} (r_c{rec['r']-rec['rc']:+d}) seed {rec['seed']} N={rec['N']} T={rec['T']}: "
                  f"islands {rec['islands_mid']} (w<= {max(rec['widths_mid']) if rec['widths_mid'] else 0}) at {rec['tmid']} -> "
                  f"{rec['islands_end']} (w<= {max(rec['widths_end']) if rec['widths_end'] else 0}); act {rec['act_mid']:.2f}% -> {rec['act_end']:.2f}%")
            print("      " + "  ".join(f"{mid[i]:.0e}:{rho[i]:.2f}" for i in sel if rho[i] == rho[i]))


def creep():
    for fn in sorted(glob.glob("creep_B*.jsonl")):
        recs = [json.loads(l) for l in open(fn)]
        print(f"\nCREEP {fn}: n1 slope per island per front (v_est), islands mid -> end")
        for rec in recs:
            print(f"  B={rec['B']} r={rec['r']} (r_c{rec['r']-rec['rc']:+d}) seed {rec['seed']}: islands {rec['islands_mid']} -> {rec['islands_end']}; "
                  f"n1 {rec['n1'][0]:.0f} -> {rec['n1'][-1]:.0f}; v_est {rec['v_est']:.2e}; widths_end {rec['widths_end'][-5:]}")


def lifetimes():
    for fn in sorted(glob.glob("island_life_B*.jsonl")):
        recs = [json.loads(l) for l in open(fn)]
        print(f"\nISLAND LIFETIMES {fn}")
        rs = sorted(set(x['r'] for x in recs))
        for r in rs:
            sub = [x for x in recs if x['r'] == r]
            dead = [x['tdead'] for x in sub if x['tdead'] > 0]
            T = max(x['tdead'] for x in sub) if dead else 0
            print(f"  B={sub[0]['B']} r={r}: {len(dead)}/{len(sub)} dead; lifetimes {sorted(dead)}; max extent of survivors {[x['max_ext'] for x in sub if x['tdead']==0]}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fronts(); thresholds(); creep(); lifetimes()
