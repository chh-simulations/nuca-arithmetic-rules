#!/usr/bin/env python3
"""Island cores from random initial states: a ring of N = 4 2^B + r cells
(r = r_c - 8 max(1, 2^(B-10)), below the threshold) is run for 200 2^B
steps; the active clusters (cells changed within 4000 steps, gap 3) are
the islands; the core of an island is its cells with words at or above
2^(B-1) - 32 (the pinned bands); each core word is assigned to the
nearer band, 2^(B-1) or 3 2^(B-2).

usage: island_cores.py B seed [B seed ...]
output: island_cores.jsonl (appended)
"""
import sys, json
import numpy as np
import nuca_fast as nf


def cores(B, seed):
    base = 1 << B; rc = 3 * base // 8; r = rc - 8 * max(1, base // 1024); N = 4 * base + r; T = 200 * base
    c, n = nf.ic_random(N, B, seed)
    nf.run(c, n, r, B, T, stop_dead=False)
    isl = nf.islands(c, n, r, B, steps=4000)
    sizes = []; band_lo = 0; band_hi = 0; offsets = []
    for a, b in isl:
        seg = n[a:b + 1]; core = seg >= base // 2 - 32
        sizes.append(int(core.sum()))
        for w in seg[core]:
            d1 = abs(int(w) - base // 2); d2 = abs(int(w) - 3 * base // 4)
            if d1 <= d2: band_lo += 1
            else: band_hi += 1
            offsets.append(min(d1, d2))
    sizes.sort()
    mode = max(set(sizes), key=sizes.count) if sizes else None
    return dict(B=B, seed=seed, r=r, rc=rc, N=N, T=T, islands=len(isl), density_pct=round(100 * len(isl) / N, 3),
                core_sizes=sizes, mode=mode, widths=sorted(b - a + 1 for a, b in isl),
                core_words=len(offsets), band_lo=band_lo, band_hi=band_hi,
                offset_max=max(offsets) if offsets else None, offset_mean=round(float(np.mean(offsets)), 2) if offsets else None)


def main():
    a = [int(x) for x in sys.argv[1:]]
    with open('island_cores.jsonl', 'a') as out:
        for B, seed in zip(a[::2], a[1::2]):
            rec = cores(B, seed)
            out.write(json.dumps(rec) + '\n'); out.flush()
            print(f"B={B} seed={seed} r={rec['r']} N={rec['N']}: {rec['islands']} islands ({rec['density_pct']}%), core sizes {rec['core_sizes']}, mode {rec['mode']}; "
                  f"words {rec['core_words']} in bands {rec['band_lo']}/{rec['band_hi']}, |offset| max {rec['offset_max']} mean {rec['offset_mean']}", flush=True)


if __name__ == '__main__':
    main()
