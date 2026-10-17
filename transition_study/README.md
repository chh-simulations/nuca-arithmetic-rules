# The transition study

Scripts and results of the study of the frozen-to-active transition
of the automaton (the companion study to the paper of this
repository; a separate paper is in preparation).

Scripts, in the order of use:

- `nuca_fast.py`: compiled runner (numba) for the B-bit automaton, msb
  reading, linear increment, in the residue form (the local rule depends
  on N only through r = N mod 2^B; the ring can be any N = k 2^B + r).
  `run` returns per-step series (changed cells, 1-cells, outermost
  changed cells) and the step at which the ring became dead; `islands`
  marks the cells that change within a window and clusters them.
- `gate_fast.py`: the runner against the canonical mod-N loop of the
  companion paper (118 checks).
- `front.py`: random block in a prepared sea; front positions and
  velocities. Results: `front_B{B}.npz`.
- `threshold.py`: bulk runs from a random state around r_c; activity in
  log windows, islands at t_mid and at the end. Results:
  `threshold_B{B}.jsonl`.
- `quench.py`: the developed active phase (prepared at r = 2^(B-1)) run
  below threshold; covered fraction and domains at six checkpoints.
  Results: `quench_B{B}.jsonl`.
- `creep.py`: bulk growth of the number of 1-cells per island (creep
  below the direct threshold). Results: `creep_B{B}.jsonl`.
- `island_life.py`: islands harvested from random states, transplanted
  into fresh seas; lifetimes. Results: `island_life_B{B}.jsonl`.
- `domain_density.py`: the activity inside the growing domains at
  r_c - 1 and in the bulk at r_c + 1. Results: `domain_density.jsonl`.
- `edge_model.py`: the reduced driven-edge model (creep tail, the
  factors at r_c - 1 and r_c + 1, the B-independence per clock period).
- `spread.py`: spreading from a random patch (written for the
  directed-percolation design; not used in the paper).
- `analyze.py`: tables from the stored results. `table_rows.py`: the
  LaTeX rows of the velocity table. `figures.py`: the figures.
- `make_expected.py` writes `expected_pre.py` from the stored results;
  `verify_pre.py quick|full` regenerates the numbers and compares.
- The stored results of the campaign are in `results/`; the scripts
  look for them in their own folder, so copy or link them there before
  running `analyze.py`, `figures.py`, `make_expected.py` or
  `verify_pre.py` (or run the measurement drivers to regenerate them).

Measurement campaign of 16-17 October 2026 (two cores): fronts at
B = 8 (5e5 steps), 9 (1e6), 10 (2e6), 11 (2-4e6), 12 (4-8e6) on rings of
24 2^B + r (B <= 10) or 8 2^B + r cells; thresholds at B = 8, 9, 11, 12
(4e6 to 1.6e7 steps); quench at B = 8, 10 (4e6); creep and island
lifetimes at B = 8 (1e7); island lifetimes at B = 10 (4e6).
