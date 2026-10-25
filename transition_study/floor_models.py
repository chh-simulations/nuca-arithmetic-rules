#!/usr/bin/env python3
"""The two- and three-cell exact models of the pinning floor
(floor_exact2.py, floor_exact3.py) at the residues around the floor
5 2^(B-4): the largest word any of the cells can reach from the sea states
under free inputs at the ends of the group.

usage: floor_models.py            (B = 8: two and three cells; B = 10: two cells)
output: floor_models.jsonl (rewritten)
"""
import json, time
from floor_exact2 import reachable2
from floor_exact3 import reach3

PLAN = [(2, 8), (3, 8), (2, 10)]


def run():
    recs = []
    for cells, B in PLAN:
        floor = 5 * (1 << (B - 4)); base = 1 << B
        for r in (floor - 1, floor, floor + 1):
            t0 = time.time()
            if cells == 2:
                mx, m1, m2, cnt = reachable2(B, r)
            else:
                mx, cnt, _ = reach3(B, r)
            recs.append(dict(cells=cells, B=B, r=r, floor=floor, max_word=mx, states=cnt, reach_climb_band=mx >= floor, reach_active_band=mx >= base // 2, secs=round(time.time() - t0, 1)))
            print(json.dumps(recs[-1]), flush=True)
    return recs


if __name__ == '__main__':
    recs = run()
    with open('floor_models.jsonl', 'w') as f:
        for x in recs:
            f.write(json.dumps(x) + '\n')
