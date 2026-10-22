#!/usr/bin/env python3
"""Which reflectors of the bulge chase can share one compact WY block in the back-transformation, and how dense can
such a block be? Brute force on a small band.

  python3 scripts/bulge_blocks.py [n] [b]

Reflector H(s, k) of sweep s, bulge k acts on rows s + 1 + k b ..< min(n, s + 1 + k b + b) (BulgeChasing.swift). The
back-transformation applies them in reverse reduction order (sweeps descending, bulges descending within a sweep); two
of them must keep their relative order if their row ranges overlap. A set can be one block only if it is convex in
that order: every reflector that must come after one member and before another is a member too. For a seed set, the
smallest convex superset is computed, and its density reported: the members' nonzeros over (rows spanned) x (members).
"""
import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
b = int(sys.argv[2]) if len(sys.argv) > 2 else 6
H = [(s, k) for s in range(n - 2) for k in range(n) if s + 1 + k * b < n]
rows = {h: (h[0] + 1 + h[1] * b, min(n, h[0] + 1 + h[1] * b + b)) for h in H}
order = sorted(H, key=lambda h: (-h[0], -h[1]))          # application order of the back-transformation
pos = {h: i for i, h in enumerate(order)}


def overlap(a, c):
    return rows[a][0] < rows[c][1] and rows[c][0] < rows[a][1]


# Reachability "a must precede c": transitive closure over overlapping pairs, in application order.
after = {h: set() for h in order}
for i in range(len(order) - 1, -1, -1):
    a = order[i]
    for c in order[i + 1:]:
        if overlap(a, c):
            after[a] |= {c} | after[c]


def closure(seed):
    block = set(seed)
    while True:
        add = {x for x in H if x not in block
               and any(x in after[a] for a in block) and any(c in after[x] for c in block)}
        if not add:
            return block
        block |= add


def density(block):
    lo = min(rows[h][0] for h in block); hi = max(rows[h][1] for h in block)
    return sum(rows[h][1] - rows[h][0] for h in block) / ((hi - lo) * len(block)), hi - lo


s, k = 3 * b, 2
cases = [
    ('parallelogram today: sweeps s-b+1 .. s, bulge k', [(s - j, k) for j in range(b)]),
    ('same rows: H(s, k), H(s - b, k + 1)', [(s, k), (s - b, k + 1)]),
    ('same rows x3: H(s, k), H(s-b, k+1), H(s-2b, k+2)', [(s, k), (s - b, k + 1), (s - 2 * b, k + 2)]),
    ('one sweep, bulges k, k+1', [(s, k), (s, k + 1)]),
    ('half parallelogram: sweeps s-b/2+1 .. s, bulge k', [(s - j, k) for j in range(b // 2)]),
]
print(f"n = {n}, b = {b}")
for name, seed in cases:
    blk = closure(seed)
    d, h = density(blk)
    print(f"  {name:52s} -> {len(blk):3d} reflectors over {h:3d} rows, density {d:.2f}")
