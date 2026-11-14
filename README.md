# Arithmetic-rule non-uniform cellular automata

[![DOI](https://zenodo.org/badge/1414496132.svg)](https://doi.org/10.5281/zenodo.23322968)

Reference implementations and the verification suite for the paper

> P. Stoychev, *Phase structure and contraction timescales of a
> non-uniform cellular automaton with arithmetic rule dynamics*
> (submitted, 2026).

The system implemented here is due to T. E. Raptis
([github.com/rtheo](https://github.com/rtheo)), who introduced it in
a private communication of 2026, building on his spectral framework
for cellular automata (Chaos, Solitons & Fractals 91, 503 (2016)), and
whose original MATLAB implementation precedes this repository; his
account is listed here, at his request, as a sideways contribution.
He is not responsible for any errors in these ports.

## Origin

The automaton implemented here, and every simulator of it in this
repository, descends from three MATLAB programs that T. E. Raptis wrote
and sent in 2026: `hca.m` (the automaton),
`ruleupdate.m` (the rule-word update) and `cfilter.m` (the
neighbourhood kernel). Specifically:

- The neighbourhood sum of every cell, h = c(j-1) + c(j) + c(j+1), is
  in his programs the product K c of the configuration with the
  circulant kernel matrix built by `cfilter.m`, the construction of his
  spectral framework for cellular automata (Chaos, Solitons & Fractals
  91, 503 (2016)), which realises an arbitrary neighbourhood as one
  matrix. `python/hca_canonical.py` keeps that form (its `kernel`
  builds the same circulant matrix and computes h = K @ c); the
  MATLAB port `matlab/hca_canonical.m` computes the same sum with
  `circshift`, which is numerically identical for the radius-1
  neighbourhood but hides the construction.
- The rule-word update, n <- mod(n + sgn*h - 1, N) kept to its low
  eight bits, with sgn = 2c - 1 (and its quadratic variant), is his
  `ruleupdate.m`, the number-theoretic function on which the whole
  phase structure rests; the derivation of its form is his (private
  communication), as the paper states.
- The lookup of the new state as the bit of the rule word at the
  position given by h is his `hca.m`.

`python/hca_canonical.py` is a line-by-line port of these three
programs and says so in its header; `matlab/hca_canonical.m` is a
second port, made from the Python reference so that the checks could
be run in a second language, and carries checksums that tie it to the
reference. His original programs are not included here because they
are his to publish; they will be added under his name and the license
of his choice if he wishes, or linked where he places them.

## Layout

- `python/verify_p1.py` - regenerates every number of the paper: 407
  labelled checks in ten sections (S1-S10), each printing measured
  against expected with PASS / WARN / FAIL. Requires numpy only.
  Full run 30-50 minutes; `python verify_p1.py S8 S9` runs chosen
  sections; `--quick` uses one seed where two are used.
- `python/hca_canonical.py` - the canonical simulator (pure Python,
  numpy, scipy), faithful to the original MATLAB implementation.
- `python/fig2_wordlength.py` - regenerates Figure 2 of the paper
  (the threshold across word lengths and the interface velocity) by
  simulation; needs matplotlib.
- `python/hca_phase_map.py`, `python/hca_probes.py`,
  `python/gate_b_dim_sweep.py` - the probe scripts (increment sweep,
  width sweep, palindrome distance, convergence windows).
- `python/toroid_check.py` - the exact content behind the
  time-periodic remark of Section 9.
- `matlab/hca_canonical.m` - MATLAB/Octave port of the canonical
  simulator (made from the Python reference, not from the original
  programs; see Origin) with an embedded self-test whose checksums
  come from the Python reference on deterministic initial data. Run
  the file; it errors on any disagreement. Self-test passed under GNU
  Octave (2026-10-11).
- `results/` - the recorded output of the full verification run:
  version 1 of the script (399 checks) on two independent machines
  (Linux and Windows, numpy 2.4.4), digit-identical; version 2 (407
  checks: the interface velocity of Fig. 2b re-measured with the
  changed cells counted in the prepared half of the ring, 17 Oct 2026)
  on the same two machines, `verify_p1_results_v2_linux.txt` and
  `verify_p1_results_v2_windows.txt`, digit-identical.
- `transition_study/` - the follow-up study of the frozen-to-active
  transition (the transition paper, "Front pinning and depinning at
  exact rational thresholds in a cellular automaton with arithmetic
  rule words"): a compiled runner gated against the canonical loop,
  the exact reachability of the pinning floor, the measurement drivers
  (fronts of both preparations, bulk runs, quench of the developed
  phase, creep, island cores, lifetimes and seeds, a reduced edge
  model), the stored results of the five campaigns of 16-28 October
  2026 at word lengths 8 to 13, `paper_numbers.py`, which prints every
  number quoted in that paper from the stored results, and
  `verify_pre.py`, which regenerates the numbers, or re-analyses the
  stored records of the runs too long to repeat, and compares them
  with the recorded values. Requires numba. See its README, which also
  records the correction of 13 November 2026 to the word-front
  analysis.
- `classification_study/` - the verification script of the
  classification paper ("From automata to operator algebras: possible,
  impossible, and open"): `verify_p2.py` (158 checks, numpy only,
  about ten seconds), the Wetterich reconstruction `wetterich_mx3.py`,
  and the recorded outputs. See its README.

## Verification discipline

Every quantitative claim in the paper was produced by a gated
computation: a script with embedded expected values that fails loudly
on any deviation. No port is trusted until its self-test passes.

## Requirements

Python 3.9 or later; `pip install numpy scipy matplotlib` (and `numba`
for `transition_study/`).

## Citation

See `CITATION.cff`. Release v1.0.0 is archived at Zenodo,
doi:10.5281/zenodo.23322969, release v1.1.0 (the interface velocity
re-measured, the transition study added) at doi:10.5281/zenodo.23434888,
and release v1.2.0 (the transition study complete: five campaigns,
the exact pinning floor, the sectioned verification suite with its
190-check runs on two machines, the dated predictions record) at
doi:10.5281/zenodo.24015703. Release v1.3.0 (November 2026) is the
version cited by the transition paper and the classification paper: it
carries the corrected word-front analysis of the transition study with
its three added scripts, the 190-check run on the corrected suite, and
the classification study (158 checks). The concept DOI
10.5281/zenodo.23322968 always resolves to the latest version. The paper's preprint is
deposited at doi:10.5281/zenodo.23323144 (all versions); version 2 of
17 October 2026, doi:10.5281/zenodo.23434750, carries the re-measured
Fig. 2b (version 1: doi:10.5281/zenodo.23323145).

## License

MIT (see `LICENSE`).
