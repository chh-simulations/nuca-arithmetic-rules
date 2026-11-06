# Classification study

The verification script of the paper

> P. Stoychev, *From automata to operator algebras: possible,
> impossible, and open* (2026).

- `verify_p2.py` regenerates every computational claim of the paper:
  143 labelled checks with embedded expected values, in twelve sections
  that follow the paper (the finite core and spectrum of Rule 110 at
  N = 8, 10, 12; the clock-and-shift algebra; the natural-extension
  remark and the transfer norm; the Walsh system and the Cuntz
  relations at resolution 256; the reversal identity for K = 2..12; the
  time-periodic counts; the wrapped states, the positivity of the
  additive generator and the branch obstruction; Wetterich's automaton
  at three sites; the Clifford criterion on 3 and 8 bits; the
  reading-dependence identities; the affine boundaries of the lifted
  automaton; the companion automaton's word map, its core, transient
  and rotations, the affine rotations, the excitability as a character
  density and the lookup as a character). Requires numpy only; runs
  in a few seconds; prints PASS / FAIL per check and a SUMMARY line, and
  writes the same to `verify_p2_results.txt`.
- `wetterich_mx3.py` is the reconstruction of Wetterich's Thirring-model
  automaton (Phys. Rev. D 105, 074502 (2022)) at three sites, with an
  explained output; `verify_p2.py` reuses its two step functions
  verbatim and re-checks every property the paper states.
- `results/` holds the recorded outputs of the verification runs.

The script was written from the paper's statements, independently of
the computations that produced the draft, and a disagreement with the
paper is reported as FAIL rather than tuned away; the one such
disagreement found (an eigenvalue bound quoted at two significant
figures) was corrected in the paper.
