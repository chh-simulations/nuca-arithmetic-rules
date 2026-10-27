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
- `spread.py`: spreading from a random patch of 32 cells in the
  prepared sea; "alive at T" and the extent of the changed cells.
  Results: `spread_B{B}_r{r}_w32_a.npz` (B = 8: r = 80..96; 9: 190..192;
  10: 382..384; 1e6-2e6 steps).
- `long_survival.py`: the same seeds followed in chunks to long times
  (1e8 steps), restart-safe (per-chunk checkpoints in `ckpt/`, one
  record file per seed in `records/`); with `seeds=...` a list of seeds
  of an earlier run is followed further. Results:
  `long_survival_B8_r94.jsonl` (1000 seeds, 1e8),
  `long_survival_B10_r382.jsonl` (200 seeds, 2e7) and
  `long_survival_B10_r382_cont.jsonl` (its 28 survivors to 1e8).
- `front_dev.py`, `front_dev_long.py`: fronts of the developed active
  phase (prepared at r = 2^(B-1)) against the prepared sea; the long
  version runs in chunks with checkpoints. Results: `front_dev_B8.jsonl`
  (r = 80..96, 4e6 steps), `front_dev_long_B{9,10,11}.jsonl` (the
  fixed-excitability chains: B = 9 r = 190; B = 10 r = 380, 381, 382;
  B = 11 r = 764, 766; 5e7-1e8 steps, eight fronts each).
- `floor_exact.py`: exact reachability of the words of one sea cell
  under arbitrary neighbours (the pinning floor): for r <= 5 2^(B-4) the
  reachable words are exactly [0, r) (B = 6..13, every r; B = 14 at the
  floor), for r > 5 2^(B-4) a climb to 2^(B-1) + 1 exists. Results:
  `floor_exact.jsonl`. `floor_exact2.py`, `floor_exact3.py`: the same
  with two and three exact sea cells (same threshold).
- `island_cores.py`, `phase_words.py`: island cores from random states
  (B = 8..12) and the word statistics of the developed active phase.
  Results: `island_cores.jsonl`, `phase_words.jsonl`.
- `front_natural.py`: developed fronts against the sea left by a
  collapsed random state, with the uniform sea as control. Results:
  `front_natural_B8.jsonl`.
- `quench_long.py`: the developed phase below threshold to 1e8 steps
  (chunked, checkpointed). Results: `quench_long_B10.jsonl` (r = 376,
  378, 380, 382).
- `hole_words.py`: the holes of the quenched developed phase in the
  final states of `quench_long.py` (sizes, words, activity). Results:
  `hole_words.jsonl`. `word_front.py`: the word-defined front of the
  final states of the developed-phase runs against the changed-cell
  front. Results: `word_front.jsonl`.
- `floor_models.py`: the two- and three-cell models of the pinning
  floor at B = 8 and 10 (largest reachable word one residue below, at
  and above the floor). Results: `floor_models.jsonl`. The reduced
  edge model's record is `edge_model.jsonl`.
- `paper_numbers.py`: every number quoted in the paper from the stored
  results (writes `paper_numbers.json`); `make_tables.py` writes the
  rows of Tables II-IV into the manuscript from it;
  `rebuild_front_dev_records.py` rebuilds a missing summary line of
  `front_dev_long_B*.jsonl` from the per-seed files in `results/records/`.
- `analyze.py`: tables from the stored results (first campaign). `table_rows.py`: the
  LaTeX rows of the velocity table. `figures.py`: the figures.
- `make_expected.py` writes `expected_pre.py` from the stored results;
  `verify_pre.py quick|full` regenerates the numbers and compares.
- `results/final_states/`: the final states (cells and words) of the
  developed-phase runs of 5e7-1e8 steps at k = 4 (B = 8..11) and of the
  long quench (B = 10, r = 376..382), which `word_front.py`,
  `hole_words.py` and the suite re-analyse.
- `notes/predictions_16Oct2026.md`: the predictions for B = 9, 11 and
  12 as written on 16 October 2026 from the lookup table, before those
  runs, with the entries that recorded their outcome (verbatim excerpts
  of the working record; "P1" is the companion paper, "the PRE paper"
  this one).
- The stored results of the campaign are in `results/`; the scripts
  look for them in their own folder, so copy or link them there before
  running `analyze.py`, `figures.py`, `make_expected.py` or
  `verify_pre.py` (or run the measurement drivers to regenerate them).

Fourth campaign, 20 October 2026: the developed-phase creep at B = 10,
r = 380 rerun (four seeds, 1e8 steps), a control run at the floor
(B = 10, r = 320, 1e8 steps: the word front stays at the block edge),
the long quench at r = 380 and 382, the analysis of the holes and of
the word fronts, and the two- and three-cell floor models as stored
records.

Third campaign, 19-20 October 2026 (for the revised paper): bulk runs
and island lifetimes at B = 10, the natural-sea check, developed-phase
fronts at dr = -1, 0, +1 for B = 8..12 on rings of 24 2^B + r cells, the
developed phase to 1e8 steps at B = 10 (r = 376..382).

Second campaign, 18-19 October 2026: the 28 survivors of B = 10,
r = 382 followed to 1e8 steps (all grow, 3.7e-7 cells per step per
front), and the developed-phase creep at fixed excitability across word
lengths (B = 9, 10, 11; 1e8 steps).

Measurement campaign of 16-17 October 2026 (two cores): fronts at
B = 8 (5e5 steps), 9 (1e6), 10 (2e6), 11 (2-4e6), 12 (4-8e6) on rings of
24 2^B + r (B <= 10) or 8 2^B + r cells; thresholds at B = 8, 9, 11, 12
(4e6 to 1.6e7 steps); quench at B = 8, 10 (4e6); creep and island
lifetimes at B = 8 (1e7); island lifetimes at B = 10 (4e6).

The quick form of `verify_pre.py` (168 checks, the runner gate counted
as one; 70 to 100 minutes on two cores) has passed with zero failures
on two machines: Linux (Python 3.13, numpy 2.5.3, numba 0.68.0, 25
October 2026), `results/verify_pre_results_quick_linux.txt`, and
Windows 10 (Python 3.13.15, numpy 2.5.3, numba 0.68.0, 26 October
2026), `results/verify_pre_results_quick_windows.txt`. The 323 lines of
the two files are identical apart from the timings in parentheses and
one line, the S4b slopes at B = 8, r = 96: those slopes are exact
rationals n / (2 x 10^6), six of the eight end in 5 at the seventh
decimal, and the sixth printed digit rounds the tie one way or the
other depending on the last bits of the floating-point fit (the
BLAS of the machine); the compared values agree to 1e-9, as the
check requires, and the integer front positions behind them are the
same on both machines. The earlier form of the suite (71 checks,
17 October) passed with digit-identical lines on the same two
machines: `results/verify_pre_results_quick_linux_17Oct.txt` and
`results/verify_pre_results_quick_windows_17Oct.txt`. To run the
suite, copy the contents of `results/` (files and the folders
`records/` and `final_states/`) beside the scripts, on a local disk.
