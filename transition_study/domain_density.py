#!/usr/bin/env python3
"""Density of the invading domains at r_c - 1 (and the bulk active phase
at r_c + 1) at several word lengths: a random start on N = k 2^B + r
cells is run for T steps, the active domains are the clusters of cells
that change within 2000 steps, and the activity inside them (cells
changing per step divided by the domain cells) is measured over the next
2000 steps.

usage: domain_density.py B k T seed [B k T seed ...]
output: domain_density.jsonl (appended)
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def measure(B, r, k, T, seed):
    base = 1 << B; N = k * base + r
    c, n = nf.ic_random(N, B, seed)
    nf.run(c, n, r, B, T, stop_dead=False)
    isl = nf.islands(c, n, r, B, steps=2000)
    size = sum(b - a + 1 for a, b in isl)
    nch, n1, _, _, _ = nf.run(c, n, r, B, 2000, stop_dead=False)
    return dict(B=B, r=r, N=N, T=T, seed=seed, domains=len(isl), covered=round(size / N, 4),
                density=round(100 * float(nch.mean()) / size, 2), ones=round(100 * float(n1.mean()) / size, 2),
                widths=sorted(b - a + 1 for a, b in isl)[-5:])


def main():
    args = sys.argv[1:]
    with open("domain_density.jsonl", "a") as out:
        for i in range(0, len(args), 4):
            B, k, T, seed = (int(x) for x in args[i:i + 4])
            rc = 3 * (1 << (B - 3))
            for r in (rc - 1, rc + 1):
                t0 = time.time()
                rec = measure(B, r, k, T, seed); rec['secs'] = round(time.time() - t0, 1)
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(f"B={B} r={r} (r_c{r-rc:+d}) N={rec['N']} T={T}: {rec['domains']} domains covering {100*rec['covered']:.1f}%, "
                      f"activity inside {rec['density']:.1f}%, ones {rec['ones']:.1f}%, widths {rec['widths']} ({rec['secs']:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
