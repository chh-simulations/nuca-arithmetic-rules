#!/usr/bin/env python3
"""Writes expected_pre.py, the table of recorded values that verify_pre.py
checks against, from the stored results of the measurement drivers
(front_B*.npz, threshold_B*.jsonl, quench_B*.jsonl, creep_B*.jsonl,
island_life_B*.jsonl).  Run once after the measurement campaign; the
suite then regenerates every value and compares."""
import glob, json, os, pprint
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
exp = {}

fronts = {}
for fn in sorted(glob.glob("front_B*.npz")):
    B = int(fn[7:-4]); d = np.load(fn, allow_pickle=True)
    fronts[B] = {int(k[1:]): [float(x) for x in d[k].ravel()] for k in d.files if k.startswith('v')}
exp['fronts'] = fronts

thr = {}
for fn in sorted(glob.glob("threshold_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        thr[(rec['B'], rec['r'], rec['seed'])] = dict(N=rec['N'], T=rec['T'], tmid=rec['tmid'], act_mid=rec['act_mid'], act_end=rec['act_end'],
                                                     islands_mid=rec['islands_mid'], islands_end=rec['islands_end'],
                                                     maxw_mid=max(rec['widths_mid']) if rec['widths_mid'] else 0,
                                                     maxw_end=max(rec['widths_end']) if rec['widths_end'] else 0)
exp['threshold'] = thr

qu = {}
for fn in sorted(glob.glob("quench_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        qu[(rec['B'], rec['r'], rec['seed'])] = dict(N=rec['N'], T=rec['T'], act_prep=rec['act_prep'],
                                                    cover=[x['cover'] for x in rec['checkpoints']], domains=[x['domains'] for x in rec['checkpoints']],
                                                    v_retreat=round(rec['v_retreat'], 10))
exp['quench'] = qu

cr = {}
for fn in sorted(glob.glob("creep_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        cr[(rec['B'], rec['r'], rec['seed'])] = dict(N=rec['N'], T=rec['T'], tmid=rec['tmid'], islands_mid=rec['islands_mid'], islands_end=rec['islands_end'],
                                                    n1_first=round(rec['n1'][0], 3), n1_last=round(rec['n1'][-1], 3), v_est=round(rec['v_est'], 10))
exp['creep'] = cr

il = {}
for fn in sorted(glob.glob("island_life_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        il[(rec['B'], rec['r'], rec['island'])] = dict(tdead=rec['tdead'], max_ext=rec['max_ext'])
exp['island_life'] = il

# developed-phase fronts (front_dev_long.py, k = 4): per (B, r) the run length and the least-squares slope of every front over t >= T/10
fd = {}
for fn in sorted(glob.glob("front_dev_long_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        T = rec['T']; slopes = []; quarter = rec['N'] // 4
        for run in rec['runs']:
            for col in (1, 2):
                pts = np.array([(e[0], e[col]) for e in run['hist'] if e[col] is not None and e[0] >= T / 10], dtype=float)
                slopes.append(round(float(np.polyfit(pts[:, 0], pts[:, 1], 1)[0]), 12) if len(pts) >= 4 and pts[:, 1].max() < quarter - 64 else None)
        key = (rec['B'], rec['r'])
        if key not in fd or rec['k'] > fd[key]['k'] or (rec['k'] == fd[key]['k'] and T > fd[key]['T']):
            fd[key] = dict(T=T, k=rec['k'], t_prep=rec['t_prep'], chunk=rec['chunk'], slopes=slopes)
exp['front_dev'] = fd

# the pinning floor (floor_exact.py): per B the largest reachable word at every residue
fl = {}
for rec in (json.loads(l) for l in open("floor_exact.jsonl")):
    fl[rec['B']] = {x['r']: x['max_word'] for x in rec['rows']}
exp['floor'] = fl
if os.path.exists("floor_models.jsonl"):
    exp['floor_models'] = [dict(cells=x['cells'], B=x['B'], r=x['r'], floor=x['floor'], max_word=x['max_word']) for x in (json.loads(l) for l in open("floor_models.jsonl"))]

# island cores (island_cores.py) and the active phase (phase_words.py)
exp['island_cores'] = {(x['B'], x['seed']): dict(islands=x['islands'], core_sizes=x['core_sizes'], band_lo=x['band_lo'], band_hi=x['band_hi'], offset_max=x['offset_max'])
                       for x in (json.loads(l) for l in open("island_cores.jsonl"))}
exp['phase_words'] = {(x['B'], x['r']): dict(changing_pct=x['changing_pct'], on_pct=x['on_pct'], near_half_pct=x['near_half_pct'], near_three_quarters_pct=x['near_three_quarters_pct'])
                      for x in (json.loads(l) for l in open("phase_words.jsonl"))}

# long seed-survival runs: the analysis of the stored records
ls = {}
for fn in sorted(glob.glob("long_survival_B*.jsonl")):
    R = [json.loads(l) for l in open(fn)]
    alive = [x for x in R if x['tdead'] == 0]
    ls[fn] = dict(n=len(R), T=R[0]['T'], alive=len(alive), latest_death=max(x['tdead'] for x in R), extents=sorted(x['hist'][-1][1] for x in alive))
exp['long_survival'] = ls

# the natural sea (front_natural.py) and the long quench (quench_long.py), when present
if os.path.exists("front_natural_B8.jsonl"):
    exp['natural'] = {(x['B'], x['r']): dict(T=x['T'], natural=[run['natural']['v'] for run in x['runs'] if 'natural' in run], uniform=[run['uniform']['v'] for run in x['runs']])
                      for x in (json.loads(l) for l in open("front_natural_B8.jsonl"))}
ql = {}
for fn in sorted(glob.glob("quench_long_B*.jsonl")):
    for rec in (json.loads(l) for l in open(fn)):
        ql[(rec['B'], rec['r'], rec['seed'])] = dict(T=rec['T'], N=rec['N'], final_cover=rec['hist'][-1][2], final_domains=rec['hist'][-1][1],
                                                    cover=[round(x[2], 4) for x in rec['hist'][::10]])
exp['quench_long'] = ql

if os.path.exists("hole_words.jsonl"):
    exp['holes'] = {(x['B'], x['r'], x['k'], x['seed']): dict(t=x['t'], lengths=x['lengths'], totals=x['totals'], up=x['count_increases'], down=x['count_decreases'])
                    for x in (json.loads(l) for l in open("hole_words.jsonl"))}

if os.path.exists("edge_model.jsonl"):
    exp['edge'] = [dict(kind=x['kind'], B=x['B'], r=x['r'], L=x['L'], T=x['T'], p=x['p'], seeds=x['seeds'], successes=x['successes']) for x in (json.loads(l) for l in open("edge_model.jsonl"))]

if os.path.exists("word_front.jsonl"):
    exp['word_front'] = {(x['B'], x['r'], x['k']): dict(word_fronts=[run['word_front'] for run in x['runs']], changed_fronts=[run['changed_front'] for run in x['runs']], t=[run['t'] for run in x['runs']])
                         for x in (json.loads(l) for l in open("word_front.jsonl"))}

sp = {}
for fn in sorted(glob.glob("spread_B*_w32_a.npz")):
    z = np.load(fn); t = z['times']; al = z['alive']; ex = z['ext']; last = al[:, -1] == 1
    i3, i5 = int(np.searchsorted(t, 1000)), int(np.searchsorted(t, 100000))
    sp[fn] = dict(runs=int(al.shape[0]), T=int(t[-1]), alive_1e3=int(al[:, i3].sum()), alive_1e5=int(al[:, i5].sum()), alive_T=int(last.sum()),
                  ext_T=sorted(int(v) for v in ex[last, -1]), tdead_max=int(z['tdead'].max()))
exp['spread'] = sp

with open("expected_pre.py", "w") as f:
    f.write("# Recorded values of the measurement campaign (generated by make_expected.py); verify_pre.py checks against them.\n")
    f.write("EXPECTED = " + pprint.pformat(exp, width=120) + "\n")
print("expected_pre.py written:", {k: len(v) for k, v in exp.items()})
