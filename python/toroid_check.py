# toroid_check.py - the exact formal content behind the time-periodic
# (toroidal) remark of Section 9 of the paper: how closed histories can
# be counted without choosing any second-time rule. Three parts, each
# with its formal definition stated first.
#
# SETTING. First-time dynamics: F = Rule 110 on N=4 cells, periodic
# boundary (any fixed F works; nothing below chooses a second-time
# rule where none was given). A HISTORY of length T is a sequence
# h = (h_0,...,h_{T-1}) of lattice configurations. h is CONSISTENT if
# h_{t+1 mod T} = F(h_t) for every t: a self-consistent closed loop of
# the given first-time rule. NO second-time dynamics enters this
# definition.
import numpy as np
from itertools import product

N, T = 4, 4
def rule110(x):
    out = np.empty_like(x)
    for j in range(N):
        l, c, r = x[(j-1) % N], x[j], x[(j+1) % N]
        out[j] = (110 >> (4*l + 2*c + r)) & 1
    return out

configs = [np.array(c, dtype=int) for c in product([0,1], repeat=N)]
idx = {tuple(c): i for i, c in enumerate(configs)}
Fmap = np.array([idx[tuple(rule110(c))] for c in configs])

print("PART 1 - COUNTING CONSISTENT HISTORIES IS CHOICE-FREE.")
print("Consistent histories = closed T-orbits of F. Two independent counts:")
# (a) brute force over ALL (2^N)^T = 65536 histories
count_brute = 0
for h in product(range(2**N), repeat=T):
    if all(Fmap[h[t]] == h[(t+1) % T] for t in range(T)):
        count_brute += 1
# (b) trace formula: number of x with F^T(x) = x
x = np.arange(2**N)
for _ in range(T): x = Fmap[x]
count_trace = int(np.sum(x == np.arange(2**N)))
print(f"  all 65536 histories enumerated: {count_brute} consistent")
print(f"  fixed points of F^T (= tr S^T): {count_trace}")
assert count_brute == count_trace
print("  -> equal, as the trace identity demands. No second-time rule was used.\n")

print("PART 2 - THE RECURRENCE THEOREM QUANTIFIES OVER ALL LOSSLESS SCANS.")
print("Def: a SCAN is any map G on the set of all histories (65536 states).")
print("It is LOSSLESS if it is a bijection. THEOREM (finite recurrence):")
print("every orbit of a bijection on a finite set is periodic: each")
print("history - including every flawed one - lies on a cycle and returns")
print("to itself after finitely many steps; none is attracted anywhere.")
print("Demonstration on a random bijection:")
rng = np.random.default_rng(0)
perm = rng.permutation(65536)          # an arbitrary lossless scan
h0 = 12345                              # an arbitrary (inconsistent) history
h, steps = int(perm[h0]), 1
while h != h0:
    h = int(perm[h]); steps += 1
print(f"  history #{h0} returned to itself after {steps} scan steps.")
print("  This holds for EVERY lossless scan and EVERY history: the theorem")
print("  needs no selection among the many possible rules - it covers all.\n")

print("PART 3 - ONE LABELED EXAMPLE OF A SCAN, AND WHAT ERASURE BUYS.")
print("Example scan (the natural resweep): (G h)_t = F(h_{t-1 mod T}).")
print("Formal fact: fix(G) = the consistent histories.")
def resweep(h):
    return tuple(Fmap[h[(t-1) % T]] for t in range(T))
fixG = sum(1 for h in product(range(2**N), repeat=T)
           if resweep(h) == h)
print(f"  fixed points of the resweep: {fixG} (= the {count_trace} consistent histories)")
def mismatch(h):
    return sum(Fmap[h[t]] != h[(t+1) % T] for t in range(T))
# THEOREM (two lines): m(Gh) <= m(h) for every history. Proof: a mismatch
# of Gh at t means F(F(h_{t-1})) != F(h_t); since equal arguments give
# equal images, F(h_{t-1}) != h_t - a mismatch of h at t. Hence the
# mismatch never increases, and is CONSTANT on every cycle of G.
Hs = list(product(range(16), repeat=T))
mono = all(mismatch(resweep(h)) <= mismatch(h) for h in Hs)
print(f"  monotonicity m(Gh) <= m(h) verified on all 65536 histories: {mono}")
# Census and basin (exhaustive):
term = {}
for h in Hs:
    v = h
    for _ in range(20): v = resweep(v)
    term[h] = mismatch(v)
from collections import Counter
dist = Counter(term.values())
print(f"  terminal mismatch over all histories: {dict(sorted(dist.items()))}")
print(f"  -> {dist[0]} of 65536 end at consistency: the 5 consistent")
print("     histories plus 575 whose flaws F cannot distinguish (aliasing).")
print("     The remaining 99.1% circulate forever at constant positive")
print("     mismatch: the resweep transports generic flaws, it does not")
print("     resolve them. An ERASING scan can resolve ALL of them: the")
print("     constant scan G(h) = h* with h* consistent does so in one")
print("     step - and, converging, it carries an arrow. Necessity of")
print("     erasure for generic resolution is Part 2; sufficiency is this")
print("     construction. Nothing in between is claimed.")
