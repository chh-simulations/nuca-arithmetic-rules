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
  `hole_words.jsonl`. `word_front.py`: the word-defined front in the
  final states of the developed-phase runs: on each side the outermost
  cell whose word lies outside the sea range, as a displacement from
  its position in the prepared state (one convention for both sides,
  retreats included), the number of cells of the prepared sea that
  hold such a word (the converted cells), and the changed-cell front
  in the same convention. Results: `word_front.jsonl`.
- `floor_models.py`: the two- and three-cell models of the pinning
  floor at B = 8 and 10 (largest reachable word one residue below, at
  and above the floor). Results: `floor_models.jsonl`. The reduced
  edge model's record is `edge_model.jsonl`.
- `sea_range_dies.py`: rings of random cells with words uniform in the
  sea range, at five residues from 2^(B-2) + 2 to the floor at B = 8
  and 10: every configuration is dead within 105 steps (B = 8) and 304
  steps (B = 10). Plain numpy, independent of the runner. Results:
  `sea_range_dies.txt`.
- `island_sensitivity.py`: the transplanted island of the suite's S3
  and a copy with one core word raised by one, followed to 2e6 steps:
  the two runs differ on at most 29 words, all inside the island and
  its halo. Results: `island_sensitivity.txt`.
- `companion_front_geometry.py`: the companion paper's front
  measurement repeated at B = 10, r = 376 to 384 (a random half ring
  against the prepared sea, 2e5 steps): the distance of the nearest
  changing cell from the sea at each of the twenty boundaries (7 to
  193 cells at eighteen, within one cell at two) and the converted sea
  cells (none). Results: `companion_front_geometry.txt`.
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

Correction of 13 November 2026. An audit of the paper found an error
in the first form of `word_front.py`: it measured the right-hand front
one cell short of the left-hand one, and for a front with no converted
sea cell it returned the initial offset of the block edge in place of
the final position, so that a front that had retreated into the block
was entered as if it had not moved. The script now uses one convention
for both sides, takes the displacement from the prepared state and
counts the converted cells of the prepared sea; `word_front.jsonl` was
regenerated from the stored final states, and `paper_numbers.py`,
`expected_pre.py` and the suite's check of the word fronts (S9) follow
the new records. What changed in the paper: the check of the front
velocities against the words is stated with the converted cells
(36.0(34), 14.8(27) and 6.4(16) per front at B = 10, r = 382, 381 and
380; 14.0(9) and 3.1(6) at B = 11, r = 766 and 764; 55.3(38) at B = 9,
r = 190); the run at B = 11, r = 760 has four converted cells on three
of its eight fronts and one front seven cells inside its block, where
the earlier text said that no sea cell was converted; and the loss of
block-edge cells to the sea early in the runs is reported (12 of 48
fronts, 2 to 17 cells). The velocities of the tables, which are fitted
to the changed-cell front, did not change, and the three figures
regenerate pixel for pixel. The same audit corrected the third
dead-ring check of `gate_fast.py`, which had been called with a word
length of 50 and passed without testing anything (it now compares
every word with the explicit drift), the hole counts printed by
`paper_numbers.py` (the stored counts are domains on the unwrapped
ring, one more than the holes), and added the three small scripts
above for statements of the paper that had no script.

Fifth campaign, 27-28 October 2026: the developed-phase creep at B = 10,
dr = -5 to -8 (2e8 steps), at B = 9, dr = -3 to -8 (1e8), and at B = 11,
dr = -8 (2e8, the fourth point of the chain e_1 = 0.3263); random-block
fronts at B = 13 at e_1 = 3/7 and 1/2 (`front_B13.npz`, not repeated by
the suite); `alternation.py` (period-two statistics of the developed
phase, `alternation.jsonl`); the word fronts of the new final states.

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

The quick form of `verify_pre.py` after the correction of 13 November
2026 (190 checks, the runner gate counted as one; 70 to 100 minutes on
two cores) has passed with zero failures on Linux (Python 3.13.16,
numpy 2.5.3, numba 0.68.0, 13 November 2026,
`results/verify_pre_results_quick_linux.txt`). Against the form of
28 October, twenty of the 190 checks changed in content: the runner
gate (its third dead-ring check; the line of the suite reads the same)
and the nineteen checks of the word fronts in S9, which now compare
displacements, converted cells, initial positions and changed-cell
fronts with the regenerated records; the other 170 print the same
values as on 28 October, timings apart. The corrected suite has not
yet been run on Windows.

The form of 28 October (190 checks after the fifth campaign of 27-28
October) passed with zero failures on two machines: Linux (Python
3.13, numpy 2.5.3, numba 0.68.0, 28 October 2026,
`results/verify_pre_results_quick_linux_28Oct.txt`) and Windows 10
(the machine of 26 October, Python 3.13.15, numpy 2.5.3, numba 0.68.0
as recorded then; 29 October 2026,
`results/verify_pre_results_quick_windows.txt`). The 364 lines of the
two files are identical apart from the timings in parentheses and one
line, the S4b slopes at B = 8, r = 96: those slopes are exact
rationals n / (2 x 10^6), six of the eight end in 5 at the seventh
decimal, and the sixth printed digit rounds the tie one way or the
other depending on the last bits of the floating-point fit (the
BLAS of the machine); the compared values agree to 1e-9, as the
check requires, and the integer front positions behind them are the
same on both machines. The 168-check form of 25 October had passed
with zero failures on the same two machines
(`results/verify_pre_results_quick_linux_25Oct.txt`, 25 October;
`results/verify_pre_results_quick_windows_26Oct.txt`, 26 October),
each machine printing on that one line the same digits as in its
190-check run. The earlier form of the suite (71 checks,
17 October) passed with digit-identical lines on the same two
machines: `results/verify_pre_results_quick_linux_17Oct.txt` and
`results/verify_pre_results_quick_windows_17Oct.txt`. To run the
suite, copy the contents of `results/` (files and the folders
`records/` and `final_states/`) beside the scripts, on a local disk.
