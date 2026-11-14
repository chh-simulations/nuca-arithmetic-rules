# Classification study

The verification script of the paper

> P. Stoychev, *From automata to operator algebras: possible,
> impossible, and open* (2026).

- `verify_p2.py` regenerates every computational claim of the paper:
  158 labelled checks with embedded expected values, in fourteen
  sections that follow the paper (the finite core and spectrum of
  Rule 110 at N = 8, 10, 12; the clock-and-shift algebra; the
  natural-extension remark and the transfer norm; the Walsh system and
  the Cuntz relations at resolution 256; the reversal identity for
  K = 2..12; the time-periodic counts; the wrapped states, the
  positivity of the additive generator and the branch statement;
  Wetterich's automaton at three sites, with the four one-particle
  copies and the algebra they generate; the Clifford criterion on 3 and
  8 bits; the reading-dependence identities; the affine boundaries of
  the lifted automaton, with the reversible embedding of the 256 rule
  tables on four qubits; the companion automaton's word map, its core,
  transient and rotations, the affine rotations, the excitability as a
  character density and the lookup as a character; the exact counts of
  periodic states of the automaton's two readings, by a transfer matrix
  over periodic histories, which show that the two readings are two
  dynamics; and the bijection between the classes of states with a
  common future and the core). Requires numpy only; runs in about ten
  seconds; prints PASS / FAIL per check and a SUMMARY line, and writes
  the same to `verify_p2_results.txt`.
- `wetterich_mx3.py` is the reconstruction of Wetterich's Thirring-model
  automaton (Phys. Rev. D 105, 074502 (2022)) at three sites, with an
  explained output; `verify_p2.py` reuses its two step functions
  verbatim and re-checks every property the paper states.
- `results/` holds the recorded outputs of the verification runs.

Sections 1 to 10 of the script were written from the paper's
statements, independently of the computations that produced the draft;
the later sections and the extensions of Sections 8, 11 and 12 were
written together with the statements they check, in the revisions of
7 and 13 November 2026. A disagreement with the paper is reported as
FAIL rather than tuned away; the one such disagreement found in the
first form (an eigenvalue bound quoted at two significant figures) was
corrected in the paper.

Revision of 13 November 2026. An audit of the paper found that its
first form called the automaton's two readings conjugate. They are not:
on a ring of 20 cells with four-bit words one reading has
4^20 * 38609 states of period dividing four and the other 2^51 + 2560
(Section 13 of the script; the first number is the number of frozen
states of the phase-structure paper's Proposition 2). The paper and the
script were corrected together; the script grew from 143 to 158 checks.
