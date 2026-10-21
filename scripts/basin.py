#!/usr/bin/env python3
"""Does FP32 (XTB_SCC_FP32_RMSQ) send cold SCF runs to a different solution more often than FP64 does? Seeded random
displaced copies of the fragile structures, each run cold in FP64 and in FP32 with the same library; per geometry,
uhf and temperature, whether the two reach the same SCF solution (noise.py's criteria), converge, and their cycles.

  python3 scripts/basin.py LIB [--copies 20] [--rms 0.03] [--rmsq 1e-3] [--workers 4] [--out build/basin]

Copies: <base>_d<k>, Gaussian displacement of every coordinate (seed k), scaled to the given rms in Angstrom, written
to OUT/structures/. Runs (make_reference.one, production settings, cold): OUT/fp64/, OUT/fp32/. Summary: OUT/BASIN.tsv
and the counts printed at the end. Nothing is written outside OUT.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_reference as mr  # noqa: E402
from noise import same_solution  # noqa: E402

BASES = ('facet_bare_dimer', 's12_C00', 's12_C31', 'facet_closed_rms15')


def arg(name, default, cast=str):
    a = sys.argv
    return cast(a[a.index(name) + 1]) if name in a else default


def copies(out, n, rms):
    os.makedirs(f"{out}/structures", exist_ok=True)
    names = []
    for base in BASES:
        el, x = mr.read(f"{mr.ST}/{base}.xyz")
        for k in range(n):
            d = np.random.default_rng(k).normal(size=x.shape)
            d *= rms / np.sqrt((d ** 2).sum(axis=1).mean())
            y = x + d
            sid = f"{base}_d{k}"
            with open(f"{out}/structures/{sid}.xyz", 'w') as f:
                f.write(f"{len(el)}\n{sid}: {base} + seeded Gaussian displacement, rms {rms} A\n")
                f.writelines(f"{e} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}\n" for e, p in zip(el, y))
            ne = sum(mr.VE[e] for e in el)
            names += [(sid, u, t) for u in ((0, 2) if ne % 2 == 0 else (1, 3)) for t in mr.ETEMPS]
    return names


def work(job):
    lib, out, mode, rmsq, (sid, u, t) = job
    if os.path.exists(f"{out}/{mode}/{sid}_u{u}_T{t:g}.json"):
        return
    os.environ.update(OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    os.environ.pop('XTB_SCC_FP32_RMSQ', None)
    if mode == 'fp32':
        os.environ['XTB_SCC_FP32_RMSQ'] = rmsq
    mr.LIB, mr.ST, mr.REF = lib, f"{out}/structures", f"{out}/{mode}"
    mr.one(sid, u, t)


def main():
    lib = os.path.abspath(sys.argv[1])
    out = os.path.abspath(arg('--out', 'build/basin'))
    n, rms, rmsq, workers = arg('--copies', 20, int), arg('--rms', 0.03, float), arg('--rmsq', '1e-3'), \
        arg('--workers', 4, int)
    runs = copies(out, n, rms)
    jobs = [(lib, out, m, rmsq, r) for r in runs for m in ('fp64', 'fp32')]
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(work, jobs, chunksize=1))
    if os.path.exists('.sccnotconverged'):
        os.unlink('.sccnotconverged')
    rows, count = [], dict(same=0, other=0, fp64_only_conv=0, fp32_only_conv=0, neither=0, restart=0)
    it64 = it32 = 0
    for sid, u, t in runs:
        a = json.load(open(f"{out}/fp64/{sid}_u{u}_T{t:g}.json"))
        b = json.load(open(f"{out}/fp32/{sid}_u{u}_T{t:g}.json"))
        ea, eb = np.array(a['orbital_eigenvalues_Eh']), np.array(b['orbital_eigenvalues_Eh'])
        d = dict(dE=abs(a['energy_Eh'] - b['energy_Eh']),
                 dg=float(np.abs(np.array(a['gradient_Eh_bohr']) - np.array(b['gradient_Eh_bohr'])).max()),
                 dgap=abs((a['gap_Eh'] or 0) - (b['gap_Eh'] or 0)),
                 dq=float(np.abs(np.array(a['charges_e']) - np.array(b['charges_e'])).max()),
                 docc=float(np.abs(np.array(a['orbital_occupations']) - np.array(b['orbital_occupations'])).max()))
        restarted = (b['scc_iterations'] or 0) >= 160 and a['converged']
        count['restart'] += restarted
        if a['converged'] and b['converged']:
            key = 'same' if same_solution(d) else 'other'
            it64 += a['scc_iterations']; it32 += b['scc_iterations']
        else:
            key = 'fp64_only_conv' if a['converged'] else 'fp32_only_conv' if b['converged'] else 'neither'
        count[key] += 1
        rows.append((sid, u, t, key, a['scc_iterations'], b['scc_iterations'], restarted, d['dE'], d['dq'], d['docc'],
                     a['energy_Eh'], b['energy_Eh']))
    with open(f"{out}/BASIN.tsv", 'w') as f:
        f.write("id\tuhf\tetemp_K\toutcome\tit_fp64\tit_fp32\tfp32_restarted\tdE\tdq\tdocc\tE_fp64\tE_fp32\n")
        for r in rows:
            f.write('\t'.join(f"{v:.3e}" if isinstance(v, float) and v < 1 else str(v) for v in r) + '\n')
    print(f"{len(runs)} geometries x spin x T (copies {n}, rms {rms} A, FP32 rmsq {rmsq}): {count}")
    print(f"cycles over both-converged runs (fp32 counts include a failed FP32 attempt before a restart): "
          f"FP64 {it64}, FP32 {it32}")
    for r in rows:
        if r[3] != 'same':
            print('  ', *r[:7], f"dE {r[7]:.1e} dq {r[8]:.1e} docc {r[9]:.1e}")


if __name__ == '__main__':
    main()
