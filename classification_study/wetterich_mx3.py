"""
wetterich_mx3.py — Wetterich's Thirring-model cellular automaton
(C. Wetterich, Phys. Rev. D 105, 074502 / arXiv:2111.06728),
reconstructed and checked. numpy only; runs in seconds.

Run:  python wetterich_mx3.py
The output explains itself; the code below mirrors his Sec. III-V.
"""
import numpy as np

print("="*72)
print("WETTERICH'S AUTOMATON, RECONSTRUCTED")
print("="*72)
print("""
The model (Wetterich's Sec. IV): a chain of M_x sites; at each site four fermionic
occupation bits, (R1, R2, L1, L2) — right-movers and left-movers in two
colors (1 = red, 2 = green). One combined time step S = S_int o S_free:

  S_free : every right-mover hops one site right, every left-mover one
           site left (periodic chain);
  S_int  : at each site, IF exactly one single right-mover meets exactly
           one single left-mover, the colors switch:
              (R1,-,-,L2) <-> (-,R2,L1,-)   [Wetterich's eq. (48)]
              (R1,-,L1,-) <-> (-,R2,-,L2)   [Wetterich's eq. (79)]
           otherwise nothing happens.

Wetterich's claim: this automaton, with a probability distribution over initial
states, IS a discretized interacting fermionic quantum field theory
(Thirring / Gross-Neveu type). Below, every structural ingredient of that
claim is checked at M_x = 3 (all 4096 states enumerated).
""")

Mx = 3
NB = 4*Mx; NS = 1 << NB

def get(s, x, g):  return (s >> (4*x+g)) & 1
def setb(s, x, g, v):
    m = 1 << (4*x+g)
    return (s | m) if v else (s & ~m)

def step_free(s, mx=Mx):
    out = 0
    for x in range(mx):
        for g in (0,1):
            if get(s,x,g): out = setb(out,(x+1)%mx,g,1)
        for g in (2,3):
            if get(s,x,g): out = setb(out,(x-1)%mx,g,1)
    return out

def step_int(s, mx=Mx):
    out = s
    for x in range(mx):
        occ = tuple(get(s,x,g) for g in range(4))
        if   occ==(1,0,0,1): new=(0,1,1,0)
        elif occ==(0,1,1,0): new=(1,0,0,1)
        elif occ==(1,0,1,0): new=(0,1,0,1)
        elif occ==(0,1,0,1): new=(1,0,1,0)
        else: continue
        for g in range(4): out = setb(out,x,g,new[g])
    return out

# ---------------------------------------------------------------- demo ----
print("-"*72)
print("0. THE AUTOMATON IN MOTION (a wider chain, just to see the rule)")
print("-"*72)
print("""One red right-mover 'r' and one green left-mover 'G' on 8 sites
(started an even distance apart, so they land on the same site — with odd
separation they live on decoupled sublattices and pass without meeting,
Wetterich's Sec. IV.1). Lower case = right-mover, UPPER case = left-mover;
r,g = red, green; two symbols in a cell = both particles on that site.""")
mx = 8
s = 0
s = setb(s, 1, 0, 1)   # red right-mover at site 1
s = setb(s, 5, 3, 1)   # green left-mover at site 5
for t in range(7):
    row = []
    for x in range(mx):
        cell = ""
        if get(s,x,0): cell += 'r'
        if get(s,x,1): cell += 'g'
        if get(s,x,2): cell += 'R'
        if get(s,x,3): cell += 'G'
        row.append(cell.ljust(2, '.') if cell else '..')
    print(f"   t={t}:  " + " ".join(row))
    s = step_int(step_free(s, mx), mx)
print("""=> Before the meeting: red right-mover, green left-mover. They land on
   one site (the 'gR' frame: the switch has just fired there) and AFTER it
   a GREEN right-mover 'g' exits right and a RED left-mover 'R' exits
   left — both colors switched. That conditional switch is the entire
   interaction of the model. (At t=6 they have wrapped around the ring
   and met a second time — the switch fires again, restoring the
   original colors: the 'rG' frame.)
""")

# ------------------------------------------------------------- checks ----
F  = np.array([step_free(s) for s in range(NS)])
I_ = np.array([step_int(s)  for s in range(NS)])
S  = I_[F]

print("-"*72)
print("1. THE STEP IS A 'UNIQUE JUMP' OPERATOR — THE UNITARITY IS BUILT IN")
print("-"*72)
print("""A unique-jump (permutation) matrix maps every configuration to exactly
one configuration — deterministic AND invertible. Wetterich: this property
alone guarantees the unitary evolution. Checks over all 4096 states:""")
print("   S_free is a permutation:", len(set(F.tolist()))==NS)
print("   S_int  is a permutation:", len(set(I_.tolist()))==NS)
print("   S_int^2 = identity (scattering twice undoes itself):",
      np.array_equal(I_[I_], np.arange(NS)))
print("   combined step S = S_int o S_free is a permutation:",
      len(set(S.tolist()))==NS)

print()
print("-"*72)
print("2. CONSERVATION LAWS (Wetterich's Sec. IV.2, point 1)")
print("-"*72)
NR = np.array([sum(get(s,x,g) for x in range(Mx) for g in (0,1)) for s in range(NS)])
NL = np.array([sum(get(s,x,g) for x in range(Mx) for g in (2,3)) for s in range(NS)])
print("   number of right-movers conserved by S:", np.array_equal(NR[S], NR))
print("   number of left-movers  conserved by S:", np.array_equal(NL[S], NL))

print()
print("-"*72)
print("3. THE 'WAVE FUNCTION' IS CLASSICAL PROBABILITY IN DISGUISE")
print("-"*72)
print("""Wetterich's probabilistic layer (Sec. V): give the INITIAL state a probability
distribution p_tau, write q_tau = sqrt(p_tau); then q evolves by the same
permutation matrix, q(t+1) = S q(t) — Wetterich's eqs. (100)-(105). Checks with a
random unit vector q:""")
rng = np.random.default_rng(0)
q = rng.normal(size=NS); q /= np.linalg.norm(q)
Smat = np.zeros((NS,NS)); Smat[S, np.arange(NS)] = 1.0
q2 = Smat @ q
print("   norm of q preserved (S is orthogonal):", bool(np.isclose(np.linalg.norm(q2),1.0)))
print("   probabilities transported exactly, p_tau(t+1) = p at the")
print("   preimage configuration:", bool(np.allclose(q2[S]**2, q**2)))

print()
print("-"*72)
print("4. NONCOMMUTING OPERATORS — AND WHERE THEY COME FROM")
print("-"*72)
print("""Quantum mechanics is famous for observables that do not commute. In this
model they appear already: the occupation number n_R1(x=0) is a DIAGONAL
matrix; the evolution S is a permutation. Their commutator:""")
n_op = np.diag([float(get(s,0,0)) for s in range(NS)])
comm = Smat @ n_op - n_op @ Smat
print(f"   max |[S, n_R1(0)]| element = {np.abs(comm).max():.0f}  (nonzero => they do not commute)")
print("""Where does it come from? Restrict S to the simplest sector: exactly one
red right-mover, nothing else. There S is just the cyclic SHIFT of its
position:""")
one = [1 << (4*x+0) for x in range(Mx)]
sub = np.array([[1.0 if S[c]==r else 0.0 for c in one] for r in one])
print("   one-particle sector of S:")
for row in sub.astype(int): print("      ", row)
print("   equals the 3x3 cyclic shift:", np.array_equal(sub, np.roll(np.eye(Mx),1,axis=0)))
w = np.exp(2j*np.pi/Mx); Ck = np.diag([w**k for k in range(Mx)])
span = np.array([(np.linalg.matrix_power(Ck,a) @ np.linalg.matrix_power(sub,b)).flatten()
                 for a in range(Mx) for b in range(Mx)])
print(f"""   Together with the position 'clock' (diagonal of roots of unity), the
   shift generates the FULL 3x3 matrix algebra:
   span of clock^a shift^b has dimension {np.linalg.matrix_rank(span)} of {Mx*Mx} -> all of M_3(C),
   with the finite Weyl relation  clock*shift = w * shift*clock :""",
      bool(np.allclose(Ck@sub, w*sub@Ck)))
print("""
CONCLUSION. Every quantum ingredient checks out, and every one of them is
built in rather than emergent: unitarity = the unique-jump (permutation)
property; the wave function = square roots of a classical initial
probability distribution; the noncommutativity = the clock-and-shift pair
that ANY cyclic dynamics carries. What is remarkable is how little is
built in and how much comes out: an interacting fermionic quantum field
theory. What is NOT delivered: noncommutativity from the ontology itself,
a unique complex structure, an unambiguous Hamiltonian.""")
