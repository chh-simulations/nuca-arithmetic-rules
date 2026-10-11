"""Width sweep: does the active/frozen split and the plateau level track
dim mod 256 (the residue mechanism of the paper) or the raw width dim?
Multiples of 256 (512, 768, 1024) share mod-dim==mod-256 arithmetic;
non-multiples (640: offset 128; 800: offset 32; 896: offset 128) differ.
Canonical simulator, plain mode, both encodings, both variants, seeds 1-2.
"""
import numpy as np
from hca_canonical import run

DIMS = [512, 640, 768, 800, 896, 1024]
STEPS = 800
SEEDS = (1, 2)

print(f"{STEPS} steps, seeds {SEEDS}, plain mode. tail = last 100.")
print(f"rank% = rank/256 (alphabet fraction); act% = modcount/dim (activity fraction)\n")
print(f"{'dim':>5} {'mod256':>6} | {'enc':>3} {'variant':>7} | {'rank%':>6} {'act%':>6}")
for enc in ('msb', 'lsb'):
    for var in ('linear', 'poly'):
        print()
        for dim in DIMS:
            rks, acts = [], []
            for s in SEEDS:
                r, m = run(dim, STEPS, s, enc, var, 'plain')
                rks.append(r[-100:].mean() / 256 * 100)
                acts.append(m[-100:].mean() / dim * 100)
            print(f"{dim:>5} {dim % 256:>6} | {enc:>3} {var:>7} | "
                  f"{np.mean(rks):>6.1f} {np.mean(acts):>6.1f}"
                  f"   (seeds: rank {rks[0]:.1f}/{rks[1]:.1f}, act {acts[0]:.1f}/{acts[1]:.1f})")
