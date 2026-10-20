#!/usr/bin/env python3
"""Rebuilds missing lines of front_dev_long_B{B}.jsonl from the per-seed
files in records/ (written one by one as the runs complete; the summary
line is written last and can be missing after an interrupted run)."""
import glob, json, os, re
import numpy as np

groups = {}
for g in glob.glob('records/frontdev_B*_r*_k*_T*_s*.json'):
    m = re.match(r'records/frontdev_B(\d+)_r(\d+)_k(\d+)_T(\d+)_s(\d+)\.json', g)
    B, r, k, T, s = (int(x) for x in m.groups())
    groups.setdefault((B, r, k, T), {})[s] = json.load(open(g))
for (B, r, k, T), runs in sorted(groups.items()):
    fn = f'front_dev_long_B{B}.jsonl'
    have = [json.loads(l) for l in open(fn)] if os.path.exists(fn) else []
    if any(x['r'] == r and x['k'] == k and x['T'] == T for x in have):
        continue
    if sorted(runs) != [1, 2, 3, 4]:
        print(f'B={B} r={r} k={k} T={T}: incomplete ({sorted(runs)}), skipped'); continue
    res = [runs[s] for s in (1, 2, 3, 4)]
    rc = 3 * (1 << (B - 3)); e1 = (r - (1 << (B - 2))) / r
    vf = np.array([x.get('v_fit', [np.nan, np.nan]) for x in res], dtype=float).ravel()
    vn = np.array([x.get('v_net', [np.nan, np.nan]) for x in res], dtype=float).ravel()
    chunk = res[0]['hist'][0][0]
    rec = dict(B=B, r=r, rc=rc, e1=round(e1, 5), k=k, N=k * (1 << B) + r, t_prep=500000 if B >= 11 else 200000, T=T, chunk=chunk, nseeds=4, runs=res,
               v_fit_mean=float(np.nanmean(vf)), v_fit_sem=float(np.nanstd(vf) / np.sqrt(np.sum(~np.isnan(vf)))),
               v_net_mean=float(np.nanmean(vn)), v_net_sem=float(np.nanstd(vn) / np.sqrt(np.sum(~np.isnan(vn)))), secs=None, rebuilt=True)
    with open(fn, 'a') as out:
        out.write(json.dumps(rec) + '\n'); out.flush(); os.fsync(out.fileno())
    print(f'B={B} r={r} k={k} T={T}: rebuilt from records/')
