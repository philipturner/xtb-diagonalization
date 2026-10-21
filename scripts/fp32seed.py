#!/usr/bin/env python3
"""FP32 early cycles and fixed-seed warm starts, alone and together, along scans (production settings: accuracy 1,
maxiter 250).

  python3 scripts/fp32seed.py LIB OUT.json [--cases id:uhf:to,...] [--points 11] [--reps 5] [--rmsq 1e-3]

Scans interpolate linearly from structures/<id>.xyz (t = 0) to structures/<to>.xyz (t = 1), as in warmstart.py. Every
point is computed four ways, interleaved so drift hits all alike:
  cold64  fresh xtb_newResults, FP64 throughout (production)
  cold32  fresh xtb_newResults, FP32 eigensolves while the charge rms change > rmsq (xtb_setSCCPrecision)
  seed64  copy (xtb_copyResults) of the converged cold64 result at t = 0, like swift-xtb's xTB_Seed
  seed32  the same seed, FP32 as cold32
Each single point gets its own calculator with the mode's precision set (xtb_setSCCPrecision, FP32 attempt capped at
160 cycles). The measurements in LOG.md predate this: xtb then read the variable XTB_SCC_FP32_RMSQ and had no FP64
restart (none of those runs failed, so the restart wouldn't have changed them). Per point and mode: one
run with xtb's output for the cycle count and the FP32 cycle count (cycle k runs in FP32 when k = 1 or the printed rms
charge change of cycle k-1 exceeds rmsq), then --reps timed runs with output off (median). Each mode's results are
compared with cold64 against the production-accuracy gate (|dE| <= 5e-7 Eh, max|dg| <= 1e-4 Eh/bohr, gap 5e-5 Eh,
charges 2e-3 e, occupations 5e-3) and for convergence. Set OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 yourself.
"""
import ctypes as C, json, os, re, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, '..', 'structures')
DEFAULT = ('t2_R:1:t2_R_rms15,facet_closed:0:facet_closed_rms15,s12_C31:1:s12_C31_rms15,'
           'facet_dbpair_tool:1:facet_dbpair_tool_rms15,s3_S42_159:1:s3_S42_159_rms15')
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
GATE = dict(E=5e-7, g=1e-4, gap=5e-5, q=2e-3, occ=5e-3)
MODES = ('cold64', 'cold32', 'seed64', 'seed32')
CYCLE = re.compile(r'^\s+(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+E[-+]\d+\s+(\d+\.\d+E[-+]\d+)', re.M)


def read(sid):
    T = open(f"{ST}/{sid}.xyz").read().split('\n'); n = int(T[0])
    return [l.split()[0] for l in T[2:2 + n]], np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]])


def main():
    a = sys.argv[1:]
    lib, out = os.path.abspath(a[0]), a[1]
    cases = [c.split(':') for c in (a[a.index('--cases') + 1] if '--cases' in a else DEFAULT).split(',')]
    npt = int(a[a.index('--points') + 1]) if '--points' in a else 11
    reps = int(a[a.index('--reps') + 1]) if '--reps' in a else 5
    rmsq = a[a.index('--rmsq') + 1] if '--rmsq' in a else '1e-3'
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule', 'xtb_copyResults'):
        getattr(L, f).restype = C.c_void_p
    L.xtb_checkEnvironment.restype = C.c_int
    L.xtb_copyResults.argtypes = [C.c_void_p]
    if not hasattr(L, 'xtb_setSCCPrecision'):
        raise SystemExit(f'{lib} has no xtb_setSCCPrecision: the FP32 modes cannot be requested')
    results = []
    for sid, uhf, to in cases:
        uhf = int(uhf)
        el, x0 = read(sid); el1, x1 = read(to); assert el == el1
        n = len(el)
        log = tempfile.NamedTemporaryFile(suffix='.out', delete=False).name

        def setup(fp32):
            # a fresh environment per single point: the C API keeps an error after a failed SCF
            env = C.c_void_p(L.xtb_newEnvironment()); L.xtb_setVerbosity(env, 0)
            mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                              (C.c_double * (3 * n))(*(x0 / 0.52917721067).ravel()),
                                              C.byref(C.c_double(0.0)), C.byref(C.c_int(uhf)), None, None))
            calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
            L.xtb_setAccuracy(env, calc, C.c_double(1.0)); L.xtb_setElectronicTemp(env, calc, C.c_double(300.0))
            L.xtb_setMaxIter(env, calc, C.c_int(250))
            L.xtb_setSCCPrecision(env, calc, C.c_double(float(rmsq) if fp32 else 0.0), C.c_int(160), C.c_int(0))
            return env, mol, calc

        def free(env, mol, calc):
            L.xtb_delCalculator(C.byref(calc)); L.xtb_delMolecule(C.byref(mol)); L.xtb_delEnvironment(C.byref(env))

        def sp(t, seed, fp32, record):
            """One single point at scan position t; returns wall time, plus results and cycles if record."""
            env, mol, calc = setup(fp32)
            x = (x0 + t * (x1 - x0)) / 0.52917721067
            L.xtb_updateMolecule(env, mol, (C.c_double * (3 * n))(*x.ravel()), None)
            res = C.c_void_p(L.xtb_copyResults(seed) if seed is not None else L.xtb_newResults())
            if record:
                L.xtb_setVerbosity(env, 2); L.xtb_setOutput(env, log.encode())
            t0 = time.perf_counter(); L.xtb_singlepoint(env, mol, calc, res); dt = time.perf_counter() - t0
            if record:
                L.xtb_releaseOutput(env)
            ok = not L.xtb_checkEnvironment(env)
            r = dict(wall_s=dt)
            if record:
                txt = open(log, errors='replace').read(); open(log, 'w').close()
                cyc = CYCLE.findall(txt)
                rms = [float(m[1]) for m in cyc]
                thr = float(rmsq) if fp32 else None
                n32 = sum(1 for k in range(len(rms)) if thr is not None and (k == 0 or rms[k - 1] > thr))
                e = C.c_double(); L.xtb_getEnergy(env, res, C.byref(e))
                nao = C.c_int(); L.xtb_getNao(env, res, C.byref(nao))
                ev = (C.c_double * nao.value)(); oc = (C.c_double * nao.value)()
                L.xtb_getOrbitalEigenvalues(env, res, ev); L.xtb_getOrbitalOccupations(env, res, oc)
                ev, oc = np.array(ev), np.array(oc)
                g = (C.c_double * (3 * n))(); q = (C.c_double * n)()
                L.xtb_getGradient(env, res, g); L.xtb_getCharges(env, res, q)
                occ, vir = np.where(oc > 0.5)[0], np.where(oc < 0.5)[0]
                r.update(cycles=len(cyc), fp32_cycles=n32, converged=ok, energy_Eh=e.value,
                         gap_Eh=float(ev[vir].min() - ev[occ].max()), occ=oc, grad=np.array(g), q=np.array(q))
            keep = res if (record and seed is None and t == 0.0 and not fp32) else None
            if keep is None:
                L.xtb_delResults(C.byref(res))
            free(env, mol, calc)
            return r, keep

        sp(0.0, None, False, False)  # warm-up
        s0, seed = sp(0.0, None, False, True)
        assert s0['converged'], f"{sid}: seed did not converge"
        pts = []
        for k in range(npt):
            t = k / (npt - 1)
            rec, walls = {}, {m: [] for m in MODES}
            for m in MODES:
                rec[m] = sp(t, seed if m.startswith('seed') else None, m.endswith('32'), True)[0]
            for _ in range(reps):
                for m in MODES:
                    walls[m].append(sp(t, seed if m.startswith('seed') else None, m.endswith('32'), False)[0]['wall_s'])
            ref = rec['cold64']
            p = dict(t=t)
            for m in MODES:
                w = rec[m]
                d = dict(E=abs(w['energy_Eh'] - ref['energy_Eh']), g=float(np.abs(w['grad'] - ref['grad']).max()),
                         gap=abs(w['gap_Eh'] - ref['gap_Eh']), q=float(np.abs(w['q'] - ref['q']).max()),
                         occ=float(np.abs(w['occ'] - ref['occ']).max()))
                p[m] = dict(cycles=w['cycles'], fp32_cycles=w['fp32_cycles'], converged=w['converged'],
                            wall_s=float(np.median(walls[m])), energy_Eh=w['energy_Eh'], d_vs_cold64=d,
                            passes=bool(w['converged'] and all(d[k] <= GATE[k] for k in GATE)))
            pts.append(p)
            print(f"{sid:18s} t {t:.2f}  " + "  ".join(
                f"{m} {p[m]['cycles']:3d}/{p[m]['fp32_cycles']:2d} {1e3 * p[m]['wall_s']:6.0f} ms{'' if p[m]['passes'] else ' FAIL'}"
                for m in MODES), flush=True)
        L.xtb_delResults(C.byref(seed))
        os.unlink(log)
        tot = {m: dict(cycles=sum(p[m]['cycles'] for p in pts), fp32_cycles=sum(p[m]['fp32_cycles'] for p in pts),
                       wall_s=sum(p[m]['wall_s'] for p in pts), fails=sum(not p[m]['passes'] for p in pts))
               for m in MODES}
        print(f"{sid}: " + "  ".join(f"{m} cycles {tot[m]['cycles']} (FP32 {tot[m]['fp32_cycles']}) "
                                     f"{tot[m]['wall_s']:.2f} s fails {tot[m]['fails']}" for m in MODES), flush=True)
        results.append(dict(id=sid, uhf=uhf, to=to, etemp_K=300.0, rmsq=float(rmsq), reps=reps, points=pts,
                            totals=tot))
    json.dump(dict(library=lib, cases=results), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
