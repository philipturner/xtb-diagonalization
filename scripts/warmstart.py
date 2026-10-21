#!/usr/bin/env python3
"""Cold vs warm SCF starts along a scan, through the C API with production settings (accuracy 1, maxiter 250).

  python3 scripts/warmstart.py LIB OUT.json [--cases id:uhf:to,...] [--points 21] [--etemp 300] [--accuracy 1]

The scan interpolates linearly from structures/<id>.xyz (t = 0) to structures/<to>.xyz (t = 1), e.g. a structure and
its _rms15 copy. Every point is computed three ways:
  cold    fresh xtb_newResults per point (production)
  chain   one results handle along the scan (xtb_updateMolecule), as a chained relaxation would
  seed    every point starts from a copy (xtb_copyResults) of the converged t = 0 results, like swift-xtb's xTB_Seed
Records SCC cycles (from xtb's own output), energy, HOMO-LUMO gap and occupations per point and mode, the warm runs'
largest differences from the cold run (energy, gap, occupations, gradient, charges), and whether they reach the cold
run's SCF solution. Cycle counts need a library whose xtb_releaseOutput closes the file
(main; not v6.7.1).
"""
import ctypes as C, json, os, re, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, '..', 'structures')
DEFAULT = 'facet_closed:0:facet_closed_rms15,facet_dbpair_tool:1:facet_dbpair_tool_rms15,s12_C31:1:s12_C31_rms15'
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}


def read(sid):
    T = open(f"{ST}/{sid}.xyz").read().split('\n'); n = int(T[0])
    return [l.split()[0] for l in T[2:2 + n]], np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]])


def main():
    a = sys.argv[1:]
    lib, out = os.path.abspath(a[0]), a[1]
    cases = [c.split(':') for c in (a[a.index('--cases') + 1] if '--cases' in a else DEFAULT).split(',')]
    npt = int(a[a.index('--points') + 1]) if '--points' in a else 21
    etemp = float(a[a.index('--etemp') + 1]) if '--etemp' in a else 300.0
    acc = float(a[a.index('--accuracy') + 1]) if '--accuracy' in a else 1.0
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule', 'xtb_copyResults'):
        getattr(L, f).restype = C.c_void_p
    L.xtb_checkEnvironment.restype = C.c_int
    L.xtb_copyResults.argtypes = [C.c_void_p]
    results = []
    for sid, uhf, to in cases:
        uhf = int(uhf)
        el, x0 = read(sid); el1, x1 = read(to); assert el == el1
        n = len(el)
        log = tempfile.NamedTemporaryFile(suffix='.out', delete=False).name
        env = C.c_void_p(L.xtb_newEnvironment())
        L.xtb_setOutput(env, log.encode()); L.xtb_setVerbosity(env, 2)
        mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                          (C.c_double * (3 * n))(*(x0 / 0.52917721067).ravel()),
                                          C.byref(C.c_double(0.0)), C.byref(C.c_int(uhf)), None, None))
        calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
        L.xtb_setAccuracy(env, calc, C.c_double(acc)); L.xtb_setElectronicTemp(env, calc, C.c_double(etemp))
        L.xtb_setMaxIter(env, calc, C.c_int(250))
        L.xtb_releaseOutput(env)

        def sp(res, t):
            x = (x0 + t * (x1 - x0)) / 0.52917721067
            L.xtb_updateMolecule(env, mol, (C.c_double * (3 * n))(*x.ravel()), None)
            L.xtb_setOutput(env, log.encode())
            L.xtb_singlepoint(env, mol, calc, res)
            L.xtb_releaseOutput(env)
            ok = not L.xtb_checkEnvironment(env)
            if not ok:
                b = C.create_string_buffer(4096); L.xtb_getError(env, b, C.byref(C.c_int(4096)))
            txt = open(log, errors='replace').read(); open(log, 'w').close()
            it = [int(m) for m in re.findall(r'^\s+(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+E[-+]\d+', txt, re.M)]
            e = C.c_double(); L.xtb_getEnergy(env, res, C.byref(e))
            nao = C.c_int(); L.xtb_getNao(env, res, C.byref(nao))
            ev = (C.c_double * nao.value)(); oc = (C.c_double * nao.value)()
            L.xtb_getOrbitalEigenvalues(env, res, ev); L.xtb_getOrbitalOccupations(env, res, oc)
            ev, oc = np.array(ev), np.array(oc)
            g = (C.c_double * (3 * n))(); q = (C.c_double * n)()
            L.xtb_getGradient(env, res, g); L.xtb_getCharges(env, res, q)
            occ, vir = np.where(oc > 0.5)[0], np.where(oc < 0.5)[0]
            # with SCC iterations counted per call; xtb restarts the count from the restart file's history
            return dict(cycles=max(it) if it else None, converged=ok, energy_Eh=e.value,
                        gap_Eh=float(ev[vir].min() - ev[occ].max()), occ=oc, grad=np.array(g), q=np.array(q))

        seed = C.c_void_p(L.xtb_newResults()); s0 = sp(seed, 0.0)
        chain = C.c_void_p(L.xtb_newResults())
        pts = []
        for k in range(npt):
            t = k / (npt - 1)
            r = C.c_void_p(L.xtb_newResults()); cold = sp(r, t); L.xtb_delResults(C.byref(r))
            ch = sp(chain, t)
            r = C.c_void_p(L.xtb_copyResults(seed)); sd = sp(r, t); L.xtb_delResults(C.byref(r))
            p = dict(t=t)
            for name, w in (('cold', cold), ('chain', ch), ('seed', sd)):
                p[name] = dict(cycles=w['cycles'], converged=w['converged'], energy_Eh=w['energy_Eh'], gap_Eh=w['gap_Eh'])
                if name != 'cold':
                    dE = abs(w['energy_Eh'] - cold['energy_Eh']); docc = float(np.abs(w['occ'] - cold['occ']).max())
                    p[name].update(dE_vs_cold=dE, docc_vs_cold=docc,
                                   dgap_vs_cold=abs(w['gap_Eh'] - cold['gap_Eh']),
                                   dgrad_vs_cold=float(np.abs(w['grad'] - cold['grad']).max()),
                                   dq_vs_cold=float(np.abs(w['q'] - cold['q']).max()),
                                   same_solution=bool(dE <= 1e-7 and docc <= 1e-3 and abs(w['gap_Eh'] - cold['gap_Eh']) <= 1e-5))
            pts.append(p)
            print(f"{sid:18s} t {t:.2f}  cycles cold {cold['cycles']!s:>3s} chain {ch['cycles']!s:>3s} seed {sd['cycles']!s:>3s}"
                  f"  same chain {p['chain']['same_solution']!s:5s} seed {p['seed']['same_solution']!s:5s}", flush=True)
        L.xtb_delResults(C.byref(chain)); L.xtb_delResults(C.byref(seed))
        L.xtb_delCalculator(C.byref(calc)); L.xtb_delMolecule(C.byref(mol)); L.xtb_delEnvironment(C.byref(env))
        os.unlink(log)
        tot = {m: sum(p[m]['cycles'] or 0 for p in pts) for m in ('cold', 'chain', 'seed')}
        print(f"{sid}: total cycles {tot}; seed point t=0 cold {s0['cycles']}", flush=True)
        results.append(dict(id=sid, uhf=uhf, to=to, etemp_K=etemp, accuracy=acc, points=pts, total_cycles=tot))
    json.dump(dict(library=lib, cases=results), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
