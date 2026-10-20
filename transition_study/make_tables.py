#!/usr/bin/env python3
"""Writes the rows of Tables II and III of the paper into P3_transition.tex
(between the %%TABLE_*_BEGIN%% / _END%% markers) from paper_numbers.json
(run paper_numbers.py json first)."""
import json, re, math

J = json.load(open('paper_numbers.json'))
rows = J['velocity_rows']


def fmt(v, err):
    """v with the uncertainty in the last digits: 1.450(21) x 10^-4."""
    if v is None:
        return '--'
    if v == 0:
        return '0'
    ex = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** ex
    # digits so that the error shows with two significant figures
    if err and err > 0:
        e_m = err / 10 ** ex
        d = max(0, 1 - int(math.floor(math.log10(e_m))))   # decimals in the mantissa
        d = min(d, 3)
        mant = f"{m:.{d}f}"; unc = int(round(e_m * 10 ** d))
        if unc == 0: unc = 1
        if float(mant) >= 10:   # rounding carried the mantissa to 10
            ex += 1; m /= 10; e_m /= 10
            d = min(3, max(0, 1 - int(math.floor(math.log10(e_m)))))
            mant = f"{m:.{d}f}"; unc = int(round(e_m * 10 ** d)) or 1
        return f"${mant}({unc}) \\times 10^{{{ex}}}$"
    return f"${m:.2f} \\times 10^{{{ex}}}$"


def fmt_scaled(v, B):
    sc = v * (1 << (B - 3))
    if sc >= 0.01:
        return f"${sc:.3f}$" if sc < 1 else f"${sc:.2f}$"
    if sc >= 0.001:
        return f"${sc:.4f}$"
    ex = int(math.floor(math.log10(sc))); m = sc / 10 ** ex
    return f"${m:.1f} \\times 10^{{{ex}}}$"


def table_v():
    out = []; out_s = []
    order = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 8, 'u7', 'u8']
    for dr in order:
        lab = f"${dr:+d}$" if isinstance(dr, int) else ("$u = 7$" if dr == 'u7' else "$u = 8$")
        vs = []; scs = []
        for B in (8, 9, 10, 11, 12):
            x = rows.get(f"{B},{dr}")
            if x is None:
                vs.append('--'); scs.append('--')
            else:
                vs.append(fmt(x['v'], x['err'])); scs.append(fmt_scaled(x['v'], B))
        out.append(lab + ' & ' + ' & '.join(vs) + r' \\')
        if not (isinstance(dr, int) and dr <= -2):     # the creep rows are in Table IV in units of 1e-5
            out_s.append(lab + ' & ' + ' & '.join(scs) + r' \\')
    return '\n'.join(out), '\n'.join(out_s)


def table_creep():
    out = []
    for dr in (-1, -2, -3, -4):
        cells = []
        for B in (8, 9, 10, 11):
            x = rows.get(f"{B},{dr}")
            if x is None or x['kind'] != 'dev':
                cells.append('--')
            else:
                sc = x['v'] * (1 << (B - 3)) * 1e5; er = x['err'] * (1 << (B - 3)) * 1e5
                d = max(0, 1 - int(math.floor(math.log10(er)))) if er > 0 else 1
                d = min(d, 3)
                cells.append(f"${sc:.{d}f}({int(round(er * 10 ** d)) or 1})$")
        out.append(f"${dr:+d}$ & " + ' & '.join(cells) + r' \\')
    return '\n'.join(out)


def splice(s, name, body):
    a, b = f"%%{name}_BEGIN%%", f"%%{name}_END%%"
    i, j = s.index(a) + len(a), s.index(b)
    return s[:i] + '\n' + body + '\n' + s[j:]


s = open('P3_transition.tex').read()
tv, tvs = table_v()
s = splice(s, 'TABLE_V', tv)
s = splice(s, 'TABLE_VS', tvs)
s = splice(s, 'TABLE_CREEP', table_creep())
open('P3_transition.tex', 'w').write(s)
print(tv); print(); print(tvs); print(); print(table_creep())
