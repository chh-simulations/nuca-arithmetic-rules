import sys, json, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from floor_exact2 import step_tables
# three exact sea cells, free inputs at both ends; sparse BFS over tuples encoded as int64
def reach3(B, r):
    base = 1 << B; S = 2 * base
    T = step_tables(B, r)
    cp_tab = np.zeros((2, 4, base), dtype=np.int64); nn_tab = np.zeros((2, 4, base), dtype=np.int64)
    for (c, h), (cp, nn) in T.items():
        cp_tab[c, h] = cp; nn_tab[c, h] = nn
    # initial: all off, words in [0, r)
    g = np.arange(r)
    s1, s2, s3 = np.meshgrid(g, g, g, indexing='ij')
    cur = (s1.ravel() * S + s2.ravel()) * S + s3.ravel()
    seen = np.zeros(S * S * S, dtype=bool)   # 2^27 for B=8 = 134 MB
    seen[cur] = True
    frontier = cur
    it = 0
    while frontier.size:
        it += 1
        st1 = frontier // (S * S); st2 = (frontier // S) % S; st3 = frontier % S
        c1, n1 = st1 // base, st1 % base; c2, n2 = st2 // base, st2 % base; c3, n3 = st3 // base, st3 % base
        nxt = []
        for a in (0, 1):
            for d in (0, 1):
                h1 = a + c1 + c2; h2 = c1 + c2 + c3; h3 = c2 + c3 + d
                e1 = cp_tab[c1, h1, n1] * base + nn_tab[c1, h1, n1]
                e2 = cp_tab[c2, h2, n2] * base + nn_tab[c2, h2, n2]
                e3 = cp_tab[c3, h3, n3] * base + nn_tab[c3, h3, n3]
                nxt.append((e1 * S + e2) * S + e3)
        nxt = np.unique(np.concatenate(nxt))
        nxt = nxt[~seen[nxt]]
        seen[nxt] = True
        frontier = nxt
    allst = np.flatnonzero(seen)
    w = np.concatenate([(allst // (S * S)) % base, ((allst // S) % S) % base, (allst % S) % base])
    return int(w.max()), int(allst.size), it
if __name__ == '__main__':
    B = int(sys.argv[1]); base = 1 << B; rc = 3 * (1 << (B - 3)); floor = 5 * (1 << (B - 4))
    for r in (int(x) for x in sys.argv[2:]):
        mx, cnt, it = reach3(B, r)
        print(json.dumps(dict(B=B, r=r, u=round(r / (1 << (B - 4)), 4), max_word=mx, states=cnt, iterations=it, reach_climb_band=mx >= floor, reach_active_band=mx >= base // 2)), flush=True)
