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
    order = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8, 'u7', 'u8']
    for dr in order:
        lab = f"${dr:+d}$" if isinstance(dr, int) else ("$e_1 = 3/7$" if dr == 'u7' else "$e_1 = 1/2$")
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
    for dr in (-1, -2, -3, -4, -5, -6, -7, -8, -10, -12, -14, -16):
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


def table_G():
    """v/v_tau at fixed excitability across word lengths (paper_numbers 'fixed_e1'); half-spread of the four fronts in the last digits."""
    out = []
    G = J['fixed_e1']
    for key in sorted(G, key=lambda k: G[k]['e1']):
        g = G[key]; pts = {p['B']: p for p in g['points']}
        u = g['points'][0]['r'] / (1 << (g['points'][0]['B'] - 4))
        cells = []
        for B in (8, 9, 10, 11, 12):
            if B in pts:
                v, s_ = pts[B]['G'], pts[B]['spread']
                cells.append(f"${v:.3f}({int(round(s_ * 1000)) or 1})$")
            else:
                cells.append('--')
        ulab = f"{u:.4f}".rstrip('0').rstrip('.')
        out.append(f"${ulab}$ & ${g['e1']:.4f}$ & ${g['dist']:.4f}$ & " + ' & '.join(cells) + f" & ${g['G_mean']:.3f}$" + r' \\')
    return '\n'.join(out)


def table_surv():
    """Survival table: P(alive) at 1e3..1e8 (seeds) and the hazard per decade (seeds and transplants), from paper_numbers 'survival'."""
    S = J['survival']
    s8, s10, s10c = S['long_survival_B8_r94.jsonl'], S['long_survival_B10_r382.jsonl'], S['long_survival_B10_r382_cont.jsonl']
    t8, t10 = S['island_life_B8.jsonl:all r'], S['island_life_B10.jsonl:all r']
    out = []
    names = ['10^3', '10^4', '10^5', '10^6', '10^7', '10^8']
    for i, nm in enumerate(names):
        p8 = s8['P_alive'][i]; p10 = s10['P_alive'][i]
        if p10 is None: p10 = s10c['P_alive'][i] * s10['P_alive'][4]      # the 28 survivors of 200 followed on
        out.append(f"alive at ${nm}$ & ${p8:.3f}$ & ${p10:.3f}$ & -- & -- \\\\")
    def hz(h):
        if h is None: return '--'
        if h['deaths'] == 0: return f"$0\\ (/{h['exposure']:.0e})$".replace('e+0', '\\times 10^{').replace('e+', '\\times 10^{') + '}' if False else f"$0$ ({h['exposure']:.1e})"
        return f"${h['hazard']:.1e}$"
    def fmt_e(x):
        m_, e_ = f"{x:.1e}".split('e'); return f"{m_} \\times 10^{{{int(e_)}}}"
    def cell(h):
        if h is None: return '--'
        if h['deaths'] == 0: return f"$0$ ($0/{fmt_e(h['exposure'])}$)"
        return f"${fmt_e(h['hazard'])}$ (${h['deaths']}/{fmt_e(h['exposure'])}$)"
    labels = ['(10^3, 10^4]', '(10^4, 10^5]', '(10^5, 10^6]', '(10^6, 10^7]', '(10^7, 10^8]']
    for i, lab in enumerate(labels):
        h8 = s8['hazards'][i]; h10 = s10['hazards'][i] if i < 4 else s10c['hazards'][i]
        ht8 = t8['hazards'][i]; ht10 = t10['hazards'][i]
        if i == 3:   # transplants: the last decade is truncated at the run length
            ht8 = dict(ht8, label='(10^6, 10^7]'); ht10 = dict(ht10, label='(10^6, 4 x 10^6]')
        out.append(f"hazard in ${lab}$ & {cell(h8)} & {cell(h10)} & {cell(ht8)} & {cell(ht10)} \\\\")
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
if '%%TABLE_G_BEGIN%%' in s:
    s = splice(s, 'TABLE_G', table_G())
if '%%TABLE_SURV_BEGIN%%' in s:
    s = splice(s, 'TABLE_SURV', table_surv())
open('P3_transition.tex', 'w').write(s)
print(tv); print(); print(tvs); print(); print(table_creep())
