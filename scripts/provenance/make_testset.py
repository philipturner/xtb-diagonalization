#!/usr/bin/env python3
"""PROVENANCE ONLY (ran once in the mechanosynthesis repo, claude-molecular-renderer/Sources/Workspace/Data/diagonalization,
whose .build/ held the sources). Kept to document how each structure was made; it does not run from this repo.

Mac: GFN2 benchmark/validation structures for the xTB diagonalization work (BRIEF), copied from this repo's .build
into Data/diagonalization/structures/ with a manifest (structures/MANIFEST.tsv: id, file, atoms, n basis functions,
uhf, charge, shell type, source, what it exercises). Charge 0 everywhere; uhf as production runs it.

  ~/miniforge3/bin/python Data/diagonalization/make_testset.py
"""
import collections, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.abspath(os.path.join(HERE, '../../../../.build'))
OUT = f"{HERE}/structures"
NB = {'H': 1, 'C': 4, 'O': 4, 'Si': 9, 'Ge': 9}  # GFN2 basis functions per atom (src/xtb/gfn2.f90)
VE = {'H': 1, 'C': 4, 'O': 6, 'Si': 4, 'Ge': 4}

# id, source (relative to .build), uhf list, what it exercises
COPY = [
    ('t2_R', 'stage3/htx/dft_stage1/cc_inputs/depth_z2.8__R.xyz', [1], 'T2 cluster Si8H14 + Ge(CH3)3-C#C* (reactant), small n'),
    ('t2_TS', 'stage3/htx/dft_stage1/cc_inputs/depth_z2.8__TS.xyz', [1], 'T2 H-transfer saddle (H shared by C(t) and Si): radical, near-degenerate'),
    ('s12_C00', 'dft/sp-common/C00.xyz', [1], 'Stage 1/2 common region, MAOC-C2 tool on Si(100) 2x3'),
    ('s12_C31', 'dft/sp-common/C31.xyz', [1], 'Stage 1/2 common region, EAOGe-C2 tool on Si(100) 2x3'),
    ('facet_dbpair_tool', 'stage3/podkit_dbwall/kit/stage3/inputs/dbwall/dbw_h0_near.xyz', [1, 3],
     'Stage 3 production: 40-Si facet with a BARE DB pair + C2 radical tool (3 unpaired; production uhf 1); hardest SCF'),
    ('facet_capped_tool', 'stage3/podkit_capsi/kit/stage3/inputs/capsi/cap_d3.60.xyz', [1], 'fully H-capped facet + radical tool'),
    ('facet_capsi_product', 'stage3/t1/edgecheck/x+4.0_y+3.5_g0.330.xyz', [1], 'C(t) bonded to a capped Si, two Si-Si back bonds broken (strained)'),
    ('facet_h_abstracted', 'stage3/t1/edgecheck/x+2.5_y+3.5_g0.300.xyz', [1], 'H abstraction product: C-H on the tool, new surface DB radical'),
    ('s3_S40_156', 'stage3/fullcheck_S40_p002.xyz', [1], 'Stage 3 explicit GFN2 region (S40)'),
    ('s3_S42_159', 'stage3/springs_renders/S42_prerupture.xyz', [1], 'Stage 3 explicit GFN2 region (S42), just before a bond rupture'),
    ('s3_S49_163', 'stage3/springs_renders/S49_final.xyz', [1], 'Stage 3 explicit GFN2 region (S49), largest n'),
    ('diamond122', 'xtb_bench/diamond122.xyz', [0], 'swift-xtb diamond cell (closed shell, degenerate levels)'),
    ('diamond222', 'xtb_bench/diamond222.xyz', [0], 'swift-xtb diamond cell'),
    ('diamond233', 'xtb_bench/diamond233.xyz', [0], 'swift-xtb diamond cell (n 776 profile system)'),
]


def read(fn):
    L = open(fn).read().split('\n'); n = int(L[0])
    return [l.split()[0] for l in L[2:2 + n]], np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]]), L[1]


def write(fn, el, x, comment):
    with open(fn, 'w') as f:
        f.write(f"{len(el)}\n{comment}\n" + ''.join(f"{e:2s} {a:14.8f} {b:14.8f} {c:14.8f}\n" for e, (a, b, c) in zip(el, x)))


def bonds(el, x):
    r = {'H': 0.32, 'C': 0.75, 'O': 0.66, 'Si': 1.11, 'Ge': 1.20}
    d = np.linalg.norm(x[:, None] - x[None], axis=2)
    lim = np.array([[r[a] + r[b] + 0.4 for b in el] for a in el])
    return (d < lim) & (d > 0.1), d


def tool_atoms(el, x):
    """the Ge(CH3)3-C#C tool: everything bonded-connected to the Ge"""
    A, _ = bonds(el, x)
    g = el.index('Ge'); seen = {g}; st = [g]
    while st:
        i = st.pop()
        for j in np.where(A[i])[0]:
            if j not in seen and el[j] != 'Si':
                seen.add(j); st.append(j)
    return seen


def drop(el, x, idx):
    keep = [i for i in range(len(el)) if i not in idx]
    return [el[i] for i in keep], x[keep]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    def add(sid, el, x, uhfs, src, what, comment):
        fn = f"{sid}.xyz"; write(f"{OUT}/{fn}", el, x, f"{sid}: {comment}")
        nb = sum(NB[e] for e in el); ne = sum(VE[e] for e in el)
        for u in uhfs:
            if (ne - u) % 2:
                raise SystemExit(f"{sid}: uhf {u} inconsistent with {ne} electrons")
            shell = 'closed' if u == 0 else f"open (uhf {u})"
            rows.append((f"{sid}" + (f"_u{u}" if len(uhfs) > 1 else ''), fn, len(el), nb, u, 0, shell,
                         dict(collections.Counter(el)), src, what))

    for sid, src, uhfs, what in COPY:
        el, x, c = read(f"{B}/{src}")
        add(sid, el, x, uhfs, src, what, c.strip()[:100])
    # derived: tool removed (closed-shell facet), one bare dimer, bare DB pair without the tool
    el, x, _ = read(f"{B}/stage3/podkit_capsi/kit/stage3/inputs/capsi/cap_d3.60.xyz")
    fe, fx = drop(el, x, tool_atoms(el, x))
    add('facet_closed', fe, fx, [0], 'derived from cap_d3.60 (tool removed)', 'H-passivated 40-Si facet, closed shell', 'cap_d3.60 minus tool')
    A, d = bonds(fe, fx)
    # bare dimer: the topmost Si carrying one H, plus its Si neighbour that also carries one H (its dimer partner)
    nH = lambda i: sum(1 for j in np.where(A[i])[0] if fe[j] == 'H')
    top = sorted([i for i, e in enumerate(fe) if e == 'Si' and nH(i) == 1], key=lambda i: -fx[i, 2])
    a = top[0]
    b = min([j for j in np.where(A[a])[0] if fe[j] == 'Si' and nH(j) == 1], key=lambda j: d[a, j])
    hs = {j for i in (a, b) for j in np.where(A[i])[0] if fe[j] == 'H'}
    de, dx = drop(fe, fx, hs)
    add('facet_bare_dimer', de, dx, [0], 'derived from facet_closed (2 H removed)',
        'one bare Si=Si dimer: small HOMO-LUMO gap, Fermi smearing active', f"bare dimer Si{a}-Si{b}, {d[a, b]:.2f} A")
    el, x, _ = read(f"{B}/stage3/podkit_dbwall/kit/stage3/inputs/dbwall/dbw_h0_near.xyz")
    pe, px = drop(el, x, tool_atoms(el, x))
    add('facet_dbpair', pe, px, [0, 2], 'derived from dbw_h0_near (tool removed)',
        'bare DB pair on two dimers, no tool: diradical, singlet vs triplet', 'dbw_h0_near minus tool')

    # perturbed copies: every atom displaced by a seeded Gaussian (per-Cartesian sigma = rms / sqrt 3), so forces are
    # non-zero everywhere (the originals carry forces mainly on frozen / cut edge atoms)
    for sid, uhfs in (('t2_R', [1]), ('diamond222', [0]), ('facet_closed', [0]), ('s12_C31', [1]),
                      ('facet_dbpair_tool', [1, 3]), ('s3_S42_159', [1])):
        el, x, _ = read(f"{OUT}/{sid}.xyz")
        for k, rms in enumerate((0.05, 0.15)):
            rng = np.random.default_rng(1000 + 17 * k + sum(map(ord, sid)))
            xp = x + rng.normal(0.0, rms / np.sqrt(3), x.shape)
            tag = f"{sid}_rms{int(rms * 100):02d}"
            add(tag, el, xp, uhfs, f"derived from {sid} (seeded Gaussian displacement)",
                f"{sid} displaced {rms} A rms: off-equilibrium forces" + (' (thermal-size)' if rms < 0.1 else ' (far from equilibrium)'),
                f"{sid} + {rms} A rms Gaussian displacement, seed {1000 + 17 * k + sum(map(ord, sid))}")
    with open(f"{OUT}/MANIFEST.tsv", 'w') as f:
        f.write("id\tfile\tatoms\tn_basis\tuhf\tcharge\tshell\telements\tsource (.build/)\texercises\n")
        for r in sorted(rows, key=lambda r: r[3]):
            f.write('\t'.join(str(v) for v in r) + '\n')
    print(f"{len(rows)} cases, {len(os.listdir(OUT)) - 1} xyz -> {OUT}")
    for r in sorted(rows, key=lambda r: r[3]):
        print(f"  {r[0]:22s} atoms {r[2]:3d}  n {r[3]:3d}  uhf {r[4]}")


if __name__ == '__main__':
    main()
