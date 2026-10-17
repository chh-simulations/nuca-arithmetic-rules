#!/usr/bin/env python3
"""Island lifetimes.  Islands are harvested from random initial states
(N = k 2^B + r_h, t_h steps, clusters of cells changed within 4000 steps,
width <= 64), each is transplanted with 8-cell margins into a fresh
prepared sea at residue r and followed for T steps; recorded: the step
at which the ring is dead (0 = alive at T), the late number of cells
changing per step, and the maximal half-extent reached.

usage: island_life.py B r_h k t_h nic T r1 r2 ...
output: island_life_B{B}.jsonl (one line per island and r)
"""
import sys, time, json
import numpy as np
import nuca_fast as nf


def main():
    B, rh, k, th, nic, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    rs = [int(x) for x in sys.argv[7:]]
    base = 1 << B
    harvest = []
    for ic in range(1, nic + 1):
        N = k * base + rh
        c, n = nf.ic_random(N, B, ic)
        nf.run(c, n, rh, B, th, stop_dead=False)
        for a, b in nf.islands(c, n, rh, B, steps=4000):
            if b - a + 1 <= 64 and a >= 8 and b + 9 <= N:
                harvest.append((ic, a, b, c[a - 8:b + 9].copy(), n[a - 8:b + 9].copy()))
    print(f"B={B}: harvested {len(harvest)} islands from {nic} initial states at r={rh}, t={th}", flush=True)
    with open(f"island_life_B{B}.jsonl", "a") as out:
        for r in rs:
            Nr = 4 * base + r; t0 = time.time(); dead = 0; lifetimes = []
            for j, (ic, a, b, cseg, nseg) in enumerate(harvest):
                rng = np.random.default_rng(10_000 + j)
                cs = np.zeros(Nr, dtype=np.int64); ns = rng.integers(0, r, Nr).astype(np.int64)
                x0 = Nr // 2; L = len(cseg)
                cs[x0:x0 + L] = cseg; ns[x0:x0 + L] = np.minimum(nseg, base - 1)
                nch, n1, lo, hi, td = nf.run(cs, ns, r, B, T, stop_dead=True)
                ok = lo >= 0
                ext = int(max(hi[ok].max() - x0 - L // 2, x0 + L // 2 - lo[ok].min())) if ok.any() else 0
                late = float(nch[-20000:].mean()) if td == 0 else None
                rec = dict(B=B, r=r, island=j, ic=ic, width=int(b - a + 1), tdead=td, late_changed=late, max_ext=ext)
                out.write(json.dumps(rec) + "\n"); out.flush()
                if td: dead += 1; lifetimes.append(td)
            print(f"B={B} r={r}: {dead}/{len(harvest)} islands dead within T={T}; lifetimes (sorted) {sorted(lifetimes)}; "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
