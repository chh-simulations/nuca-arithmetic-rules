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
whose
original MATLAB implementation precedes this repository; his account
is listed here, at his request, as a sideways contribution. He is not
responsible for any errors in these ports.

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
  simulator with an embedded self-test whose checksums come from the
  Python reference on deterministic initial data. Run the file; it
  errors on any disagreement. Self-test passed under GNU Octave
  (2026-10-11).
- `results/` - the recorded output of the full verification run:
  version 1 of the script (399 checks) on two independent machines
  (Linux and Windows, numpy 2.4.4), digit-identical; version 2 (407
  checks: the interface velocity of Fig. 2b re-measured with the
  changed cells counted in the prepared half of the ring, 17 Oct 2026)
  on the same two machines, `verify_p1_results_v2_linux.txt` and
  `verify_p1_results_v2_windows.txt`, digit-identical.
- `transition_study/` - the follow-up study of the frozen-to-active
  transition (a separate paper in preparation): a compiled runner
  gated against the canonical loop, the measurement drivers (fronts,
  bulk runs, quench of the developed phase, creep, island lifetimes,
  a reduced edge model), the stored results of the campaign at word
  lengths 8 to 12, and `verify_pre.py`, which regenerates the numbers
  and compares them with the recorded values. Requires numba. See its
  README.

## Verification discipline

Every quantitative claim in the paper was produced by a gated
computation: a script with embedded expected values that fails loudly
on any deviation. No port is trusted until its self-test passes.

## Requirements

Python 3.9 or later; `pip install numpy scipy matplotlib` (and `numba`
for `transition_study/`).

## Citation

See `CITATION.cff`. Release v1.0.0 is archived at Zenodo,
doi:10.5281/zenodo.23322969, and release v1.1.0 (the interface velocity
re-measured, the transition study added) under the same concept DOI;
the concept DOI 10.5281/zenodo.23322968
always resolves to the latest version. The paper's preprint is
deposited at doi:10.5281/zenodo.23323145.

## License

MIT (see `LICENSE`).
