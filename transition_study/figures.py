#!/usr/bin/env python3
"""Figures of the PRE paper from the stored results (front_B*.npz,
threshold_B*.jsonl, quench_B*.jsonl, creep_B*.jsonl) and one short
regeneration (the island strip).  Writes figP_*.pdf."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nuca_fast as nf

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8})


def load_fronts():
    out = {}
    for fn in sorted(glob.glob("front_B*.npz")):
        B = int(fn[7:-4]); d = np.load(fn, allow_pickle=True)
        rows = []
        for key in d.files:
            if key.startswith('v'):
                r = int(key[1:]); vs = d[key].ravel(); vs = vs[~np.isnan(vs)]
                rows.append((r, vs.mean(), vs.min(), vs.max()))
        out[B] = np.array(sorted(rows))
    return out


def fig_velocity(fronts):
    """(a) scaled velocity against dr with uncertainties (paper_numbers.json: developed-phase
    fronts at dr <= 0, random-block fronts above); (b) the creep chains at fixed excitability
    against B; (c) v / v_tau against e_1 (random-block fronts)."""
    J = json.load(open('paper_numbers.json'))
    rows = J['velocity_rows']; chains = J['chains']
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.2, 2.7))
    markers = {8: 'o', 9: 's', 10: '^', 11: 'v', 12: 'D'}
    colors = {8: '#1b1b1b', 9: '#4477aa', 10: '#cc6677', 11: '#228833', 12: '#aa3377'}
    for B in (8, 9, 10, 11, 12):
        pts = []
        for key, x in rows.items():
            b, dr = key.split(','); b = int(b)
            if b != B or not dr.lstrip('+-').isdigit(): continue
            dr = int(dr); sc = (1 << (B - 3))
            pts.append((dr, x['v'] * sc, x['err'] * sc))
        pts.sort()
        d = np.array([p[0] for p in pts]); v = np.array([p[1] for p in pts]); e = np.array([p[2] for p in pts])
        m = v > 0
        ax1.errorbar(d[m], v[m], yerr=np.minimum(e[m], 0.999 * v[m]), fmt=markers[B] + '-', color=colors[B], ms=3.5, lw=0.8, capsize=1.5, label=f"B = {B}")
    ax1.set_yscale('log'); ax1.set_xlabel(r"$dr = r - r_c$"); ax1.set_ylabel(r"$v\, 2^{B-3}$")
    ax1.set_xlim(-4.5, 8.5); ax1.axvline(0.5, color='0.6', lw=0.6, ls='--'); ax1.legend(frameon=False, loc='lower right')
    # (b) chains
    styles = {"e1=0.3263 (r = 95, 190, 380)": ('o-', '#4477aa', r"$e_1 = 0.3263$"), "e1=0.3298 (r = 191, 382, 764)": ('s-', '#cc6677', r"$e_1 = 0.3298$"),
              "e1=0.3316 (r = 383, 766)": ('^-', '#228833', r"$e_1 = 0.3316$"), "e1=1/3 (dr=0)": ('v--', '0.4', r"$e_1 = 1/3$"),
              "e1=1/3+2^-B (dr=+1)": ('D--', '0.1', r"$dr = +1$")}
    for name, (fmt, col, lab) in styles.items():
        pts = chains[name]
        Bs = [p['B'] for p in pts]; sc = [p['scaled'] for p in pts]; er = [p['err'] * (1 << (p['B'] - 3)) for p in pts]
        ax2.errorbar(Bs, sc, yerr=np.minimum(er, 0.999 * np.array(sc)), fmt=fmt, color=col, ms=3.5, lw=0.8, capsize=1.5, label=lab)
    ax2.set_yscale('log'); ax2.set_xlabel(r"$B$"); ax2.set_ylabel(r"$v\, 2^{B-3}$"); ax2.set_xticks([8, 9, 10, 11, 12]); ax2.legend(frameon=False, fontsize=6.5, loc='lower left')
    # (c) v / v_tau against e_1 (random-block fronts)
    for B, rr in sorted(fronts.items()):
        e1 = (rr[:, 0] - (1 << (B - 2))) / rr[:, 0]
        vt = 1.0 / ((1 - e1) * (1 << (B - 3)) + 0.5)
        m = rr[:, 1] > 0
        ax3.plot(e1[m], (rr[:, 1] / vt)[m], markers[B] + '-', color=colors[B], ms=3, lw=0.7, label=f"B = {B}")
    for x in (1 / 3, 3 / 7, 1 / 2):
        ax3.axvline(x, color='0.6', lw=0.6, ls='--')
    ax3.set_xlabel(r"$e_1$"); ax3.set_ylabel(r"$v / v_\tau$"); ax3.set_ylim(0, 1.1)
    for ax, lab in ((ax1, '(a)'), (ax2, '(b)'), (ax3, '(c)')):
        ax.text(0.03, 0.96, lab, transform=ax.transAxes, va='top')
    fig.tight_layout(); fig.savefig("figP_velocity.pdf"); plt.close(fig)


def fig_island():
    B = 10; base = 1 << B; r = 376; N = 2 * base + r
    c, n = nf.ic_random(N, B, 1)
    nf.run(c, n, r, B, 200000, stop_dead=False)
    isl = nf.islands(c, n, r, B, steps=4000)
    a, b = isl[0]; a0 = a - 2; b0 = b + 3
    rows = []; words = []
    for t in range(120):
        nf.run(c, n, r, B, 1, stop_dead=False)
        rows.append(c[a0:b0].copy()); words.append(n[a0:b0].copy())
    rows = np.array(rows); words = np.array(words)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
    ax1.imshow(rows.T, cmap='Greys', aspect='auto', interpolation='nearest')
    ax1.set_xlabel("step"); ax1.set_ylabel("cell"); ax1.set_title("island, B = 10, r = 376", fontsize=9)
    for j in range(words.shape[1]):
        w = words[:, j]
        if w.max() > r: ax2.plot(w, lw=0.8, label=f"cell {j + a0 - a}")
    for y, lab in ((base // 2, r"$2^{B-1}$"), (3 * base // 4, r"$3\cdot 2^{B-2}$")):
        ax2.axhline(y, color='0.5', lw=0.6, ls='--'); ax2.text(2, y + 6, lab, fontsize=8, color='0.3')
    ax2.set_xlabel("step"); ax2.set_ylabel("word"); ax2.set_ylim(480, 800)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, va='top', color='k'); ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, va='top')
    fig.tight_layout(); fig.savefig("figP_island.pdf"); plt.close(fig)


def fig_activity():
    """activity against time: random block (threshold runs) and quench (quench runs), B = 10"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    recs = [json.loads(l) for fn in glob.glob("threshold_B10.jsonl") for l in open(fn)]
    for rec in sorted(recs, key=lambda x: x['r']):
        if rec['seed'] != 1 or rec['r'] not in (368, 376, 382, 383, 384, 385, 386, 392): continue
        e = np.array(rec['edges']); mid = np.sqrt(e[:-1] * e[1:]); rho = np.array(rec['rho'], dtype=float)
        ax1.plot(mid, rho, lw=0.9, label=f"r = {rec['r']}")
    ax1.set_xscale('log'); ax1.set_yscale('log'); ax1.set_xlabel("t"); ax1.set_ylabel("activity (% of N)"); ax1.legend(frameon=False, fontsize=6, ncol=2, loc='lower left', columnspacing=0.8, handlelength=1.5)
    recs = [json.loads(l) for fn in glob.glob("quench_B10.jsonl") for l in open(fn)]
    for rec in sorted(recs, key=lambda x: x['r']):
        t = [x['t'] for x in rec['checkpoints']]; cov = [100 * x['cover'] for x in rec['checkpoints']]
        ax2.plot(t, cov, 'o-', ms=3, lw=0.8, label=f"r = {rec['r']}")
    ax2.set_xscale('log'); ax2.set_xlabel("t"); ax2.set_ylabel("active fraction (%)"); ax2.legend(frameon=False, fontsize=6, ncol=2, loc='lower left', columnspacing=0.8, handlelength=1.5)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, va='top'); ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, va='top')
    fig.tight_layout(); fig.savefig("figP_activity.pdf"); plt.close(fig)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    which = sys.argv[1:] or ['velocity', 'island', 'activity']
    if 'velocity' in which: fig_velocity(load_fronts())
    if 'island' in which: fig_island()
    if 'activity' in which: fig_activity()
    print("done:", which)
