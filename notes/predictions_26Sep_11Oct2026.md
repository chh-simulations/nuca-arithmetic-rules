# The predictions of 26 September and 11 October 2026 and their tests (record excerpts)

Verbatim excerpts from the author's working record of the first paper
("Phase structure and contraction timescales of a non-uniform cellular
automaton with arithmetic rule dynamics"). The record was not under
version control when these entries were written; this file documents
them and cannot prove their dates. It was added to the repository on
16 November 2026.

The first excerpt (26 September 2026) lists the verdicts written down
from the consistency table before the runs at residues absent from the
paper's Table 1, and the outcome. Six of the eight verdicts were met.
The two that were not are the residues 80 and 96: the verdict was
"active", because the table admits no frozen state containing 1s
there; the runs froze into the all-zero sea, the one invariant state
the table does permit in that regime. The paper's Section 5 reports
the six residues that are not in Table 1 (four verdicts met, two not);
residue 128 and the lsb control complete the eight.

The second excerpt (11 October 2026) lists what was written down for
other word lengths before those runs, and the outcome. Every verdict
was met at the time scale of the runs. The longer runs of November
2026 (verify_p1.py, section S11) show that the freezing at r_c and
r_c - 1 is a statement at that time scale: a slow invasion follows.

## Excerpt 1 (26 September 2026)

## 3. Out-of-sample confirmation (predictions recorded before runs)
msb-linear tail activity at previously untested residues, seeds 1-2:
  N=520 (r=8):   6.1 / 7.0 %   frozen  - predicted frozen  HIT
  N=528 (r=16):  7.6 / 6.2 %   frozen  - predicted frozen  HIT
  N=560 (r=48):  4.9 / 5.8 %   frozen  - predicted frozen  HIT
  N=576 (r=64):  6.5 / 5.9 %   frozen  - predicted frozen  HIT
  N=592 (r=80):  1.4 / 2.7 %   frozen  - predicted active  MISS*
  N=608 (r=96):  4.4 / 4.3 %   frozen  - predicted active  MISS*
  N=640 (r=128): 46.0 / 46.2 % active  - predicted active  HIT
  lsb control N=528: 48.8 %    active  - predicted active  HIT
(*) The two misses are the refinement, not a refutation: for
64 < r <= 128 the only invariant state is the all-zero sea, and at
r = 80, 96 the dynamics FINDS it (deepest freezing measured, and the
frozen core carries ZERO ones - confirmed below), while at r = 128
it does not. The naive prediction ignored the sea.

## Excerpt 2 (11 October 2026)

## 48.1 THE PREDICTION. Nothing in the confinement lemma or the
consistency table uses 8 except as the modulus: with B-bit words and
msb lookup at bit B-1-h, confinement is below r = N mod 2^B, the
regimes are (0,2^(B-3)], (2^(B-3),2^(B-2)], (2^(B-2),2^(B-1)], r=0
degenerate, static bit B-2 on an isolated 1; the one-neighbour sea
excitability is e_1 = (r - 2^(B-2))/r, equal to 1/3 at r_c =
3*2^(B-3) (= 96 at B=8). Written down: regimes at B=9,10; threshold
at r_c for B=6..12 (B=9: 192/193; B=10: 384/385).
## 48.2 REGIMES CONFIRMED (B=9: 4000 steps; B=10: 8000; N = 2^B+r,
seeds 1,2): regime 1 frozen WITH 101 motifs (B=10 r=32,128: act
6.5-7.7%, motifs 2-22); regime 2 frozen WITHOUT (r=192,256: 4.3-4.9%,
0 motifs, isolated 1s with bit B-2 set 100%); regime 3 sea only
(r=320,384: 0.2-0.9%, zero core 1s); r=2^(B-1) and r=0 active 45-47%;
lsb controls 48%; core 0-words below r 100% everywhere. Every
structural prediction at 100%.
## 48.3 THRESHOLD CONFIRMED AT SEVEN WORD LENGTHS (B=6..12, T_B =
max(1500, 3000*2^(B-8))): activity <= 5% for r <= r_c (max 4.6 at
B=7 r=48); the invading state at r_c+1 (B=8: 24.5/17.3 at N=353;
B=9: 10.3/6.2; B=10: 5.4/9.6; B=11: 4.8/4.5; B=12: 5.4/4.7; B=6,7
already active 25-37%); active from r_c+2 (24-44%). TIME GROWTH at
r_c+1 strictly monotone over T,2T,4T at every B=8..12 (B=12: 5.4 ->
8.8 -> 14.6%); at r_c <= 2% at 4T (B=8..10). NOTE the B=8 runs in S9
use N = 256+r (class 350-355), a third width class beside 608-610
and 864-866; values differ from S8's but the verdicts coincide.
