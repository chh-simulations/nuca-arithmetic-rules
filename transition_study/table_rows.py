#!/usr/bin/env python3
"""Prints the LaTeX rows of the velocity table of the paper from the
stored front results, for every word length present."""
import glob
import numpy as np


def fmt(v):
    if v is None: return "--"
    if abs(v) < 1e-5: return "$<10^{-5}$"
    e = int(np.floor(np.log10(abs(v)))); m = v / 10 ** e
    return f"${m:.2f}\\times10^{{{e}}}$"


data = {}
for fn in sorted(glob.glob("front_B*.npz")):
    B = int(fn[7:-4]); d = np.load(fn, allow_pickle=True)
    data[B] = {int(k[1:]): float(np.nanmean(d[k])) for k in d.files if k.startswith('v')}
Bs = sorted(data)
rows = [("$-2$", -2), ("$-1$", -1), ("$0$", 0), ("$+1$", 1), ("$+2$", 2), ("$+3$", 3), ("$+4$", 4), ("$+8$", 8), ("$u = 7$", 'u7'), ("$u = 8$", 'u8')]
print(" & " + " & ".join(f"${B}$" for B in Bs) + " & " + " & ".join(f"${B}$" for B in Bs) + r" \\")
for label, key in rows:
    vs = []
    for B in Bs:
        rc = 3 * (1 << (B - 3))
        r = rc + key if isinstance(key, int) else (7 * (1 << (B - 4)) if key == 'u7' else (1 << (B - 1)))
        vs.append(data[B].get(r))
    cells = []
    for B, v in zip(Bs, vs):
        if v is None: cells.append("[P]")
        elif key == -2: cells.append("$<10^{-5}$" if abs(v) < 1e-5 else ("$<3\\times10^{-5}$" if abs(v) < 3e-5 else fmt(v)))
        else: cells.append(fmt(v))
    scaled = []
    for B, v in zip(Bs, vs):
        if v is None or key == -2: scaled.append("--" if v is not None or key == -2 else "[P]")
        else: scaled.append(f"${v * (1 << (B - 3)):.4f}$" if v * (1 << (B - 3)) < 0.05 else f"${v * (1 << (B - 3)):.3f}$" if v * (1 << (B - 3)) < 0.5 else f"${v * (1 << (B - 3)):.2f}$")
    print(label + " & " + " & ".join(cells) + " & " + " & ".join(scaled) + r" \\")
