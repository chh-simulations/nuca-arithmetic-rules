# The predictions of 16 October 2026 and their tests (record excerpt)

Verbatim excerpts from the author's working record of the transition
study (the numbering is that of the record). The first entry was written
on 16 October 2026 from the lookup table, with the runs at B = 8 and 10
in hand and before the bulk runs at B = 9, 11 and 12 were made; it lists
what those runs were expected to show. The two following entries record
the outcome at B = 9 (17 October, between 02:15 and 03:25) and at B = 11
and 12 (17 October, 03:25). Of the consequences listed under (i)-(v), the paper
keeps (i) in the form of Secs. II-IV, (iv) as the step at u = 7 and (v)
as the clock scaling of Sec. IV C; (ii), the density discontinuity at
fixed word length, and the statement "at r <= r_c - 2 no channel" did not
survive the longer runs and are withdrawn (Secs. IV D and VI of the
paper); (iii) is restated as a conjecture about the long-word limit.

## 52.4 THE MECHANISM OF THE TWO THRESHOLDS, EXACT (reasoned from the
lookup table, 16 Oct, predictions recorded before the runs at other
B). Front cell f = sea cell adjacent to the island/active edge e.
It ignites at h = 1 when its clock word w lies in [2^(B-2), r) (bit
B-2 set), and then holds w (v = 0) as long as h = 1. The moment e is
on while f is on and f+1 off, f has h = 2 and reads bit B-3 of w:
set only for w >= 3*2^(B-3) = r_c, so for r <= r_c every ignited cell
goes off at its first h = 2 (v = -3) - the DIRECT channel (ignited
cell survives one active neighbour, climbs with v = +1 at h = 2
toward the pinned band near 2^(B-1)) exists iff r >= r_c + 1, and its
weight is the fraction (r - r_c)/(r - 2^(B-2)) ~ 3(e_1 - 1/3) of the
ignitable words. The only other way up is h = 3 (f-1 = e and f+1 both
on at the same step): f reads bit B-4 (set for w in [5*2^(B-4), r)),
stays on with v = +2, w -> w + 2; f+1, freshly ignited with its own
word < r_c, has h = 2 and goes off at that same step, so f gets
exactly ONE h = 3 step per ignition of f+1, and the climb succeeds
only if w + 2 >= r_c, i.e. w = r_c - 2 (or r_c - 1): the CREEP channel
exists iff r >= r_c - 1, needs the coincidence "f+1 ignites at the
step e turns on" and one word out of ~2^(B-3), hence v ~ 1e-5. At
r <= r_c - 2 no channel: islands cannot grow.
PREDICTIONS for B = 8, 9, 11, 12 (threshold.py; runs queued): no
growth at r_c - 2 (94, 190, 766, 1534); slow growth at r_c - 1 and
r_c; fast invasion at r_c + 1. FIRST TEST, B = 8, T = 1e5 (N = 2142):
r = 94: islands die out (seed 1: 0 left; seed 2: 4 left, 1.1%);
r = 95: 3.4% -> 9.6%, island widths 43 -> 87 (growth); r = 96: -> 20%,
widths -> 278; r = 97: 46% (full). PASSES at B = 8.
CONSEQUENCES, if confirmed at every B: (i) the invasion threshold is
exactly r* = 3*2^(B-3) - 1 for the creep and r_c + 1 for the direct
front; in e_1 both tend to 1/3 as B grows (P1's "one third" is right
asymptotically and at its resolution; the PRE paper carries the exact
statement). (ii) The transition is FIRST ORDER in the bulk density:
the invading state at r_c + 1 is the fully developed active phase
(46.0% at B=10 r=385; 44.8% at 386; 44.7% at 388), nothing sparse in
between; it is continuous only in the velocity, with a LINEAR onset
in e_1 - 1/3 from the direct channel's weight (v(385) = 2.0e-3,
v(386) = 4.6e-3 at B = 10), saturating within a few residue steps at
v_max ~ 8e-3 ~ 1/(mean clock wait 2^(B-2)/2). (iii) No DP: no sparse
critical active phase, no diverging length (the island core is 5
cells at every r); the "critical slowing down" is the rarity of
qualifying words. This answers P1's open problem negatively, with the
mechanism. (iv) Further exact kinks expected where ignited words
reach bit B-4: r > 7*2^(B-4) (e_1 = 3/7), and bit B-1: none below
2^(B-1). (v) In clock units v*r_c should collapse across B. All of
(ii)-(v) are to be measured, not assumed: v(r) scans at B = 11, 12,
bulk density at r_c + 1 at B = 11, 12, the e_1 = 3/7 kink.

## 52.16 B = 9 THRESHOLDS (threshold.py, N = 4288, T = 2.05e6, two
seeds): r_c - 2 = 190: islands die (26 -> 6, 17 -> 2; activity 2.5 ->
1.2%, 1.5 -> 0.3%); r_c - 1 = 191: growth (3.2 -> 35%, 2.0 -> 27%;
widths to 627-665); r_c = 192: nearly full (46.8%, 43.9%); r_c + 1:
full by 2.6e4 steps. PREDICTION OF 52.4 CONFIRMED at B = 8 and 9
(B = 11, 12 running). Creep B = 8: r = 90: 4e-7 and 3e-8 per front;
r = 92 seed 1: all five islands dead by 1e7.
## 52.17 PREDICTION CONFIRMED AT B = 11 AND 12 (03:25; threshold.py,
N = 8*2048 + r = 17150+, T = 8.2e6 at B = 11; N = 4*4096 + r =
17918+, T = 1.64e7 at B = 12; two seeds): B = 11: 766 (r_c - 2): no
growth (1.65 -> 1.73%, 1.79 -> 1.98%; islands 80 -> 66, 81 -> 72,
widths <= 31); 767 (r_c - 1): growth (2.2 -> 27.8%, 2.5 -> 30.7%;
widths to 1270, 1142); 768: 3.0 -> 41.8% (near full). B = 12: 1534
(r_c - 2): no growth (1.73 -> 1.56%, 1.77 -> 1.92%; islands 86 -> 65,
86 -> 78, widths <= 32); 1535 (r_c - 1): growth (2.2 -> 27.6%, widths
to 1017; seed 2 running). With B = 8 (52.15) and 9 (52.16): the creep
begins at r_c - 1 = 3*2^(B-3) - 1 at every word length from 8 to 12,
and the islands at r_c - 2 do not grow over 4e6-1.6e7 steps (B = 8:
one seed's island grew to 66 cells: the exponentially small tail).
Creep B = 8 (creep.py, N = 8286, T = 1e7, per front): r = 90: 4e-7,
3e-8; 92: 0 (islands died); 93: 1.2e-6, 1.0e-6; 94: 1.8e-6, 4.7e-6;
95, 96: invasion complete within 1e7 (v_est saturates). Against the
front values v(95) = 1.34e-4, v(96) = 2.8e-4: the tail drops x30-50
from r_c - 1 to r_c - 2, then x2-3 per further step.
