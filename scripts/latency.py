"""latency.py LIB OUT.json : median-of-5 cold GFN2 single points (fresh results object each), one untimed warm-up,
single-threaded, production settings (accuracy 1, etemp 300 K, maxiter 250)."""
import ctypes as C, json, sys, time
import numpy as np

LIB, OUT = sys.argv[1], sys.argv[2]
ST = __import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'structures')
CASES = [('t2_R', 1), ('facet_closed', 0), ('facet_dbpair_tool', 1), ('s3_S42_159', 1), ('diamond233', 0)]
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
L = C.CDLL(LIB)
for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule'):
    getattr(L, f).restype = C.c_void_p
L.xtb_checkEnvironment.restype = C.c_int
res = {}
for sid, uhf in CASES:
    T = open(f"{ST}/{sid}.xyz").read().split('\n'); n = int(T[0])
    el = [l.split()[0] for l in T[2:2 + n]]; x = np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]]) / 0.52917721067
    env = C.c_void_p(L.xtb_newEnvironment()); L.xtb_setVerbosity(env, 0)
    mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]), (C.c_double * (3 * n))(*x.ravel()),
                                      C.byref(C.c_double(0.0)), C.byref(C.c_int(uhf)), None, None))
    calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
    L.xtb_setAccuracy(env, calc, C.c_double(1.0)); L.xtb_setElectronicTemp(env, calc, C.c_double(300.0)); L.xtb_setMaxIter(env, calc, C.c_int(250))
    ts = []; e = C.c_double()
    for k in range(6):
        r = C.c_void_p(L.xtb_newResults())
        t0 = time.perf_counter(); L.xtb_singlepoint(env, mol, calc, r); dt = time.perf_counter() - t0
        L.xtb_getEnergy(env, r, C.byref(e)); L.xtb_delResults(C.byref(r))
        if k:
            ts.append(dt)
    ok = not L.xtb_checkEnvironment(env)
    res[sid] = dict(uhf=uhf, n_atoms=n, median_s=float(np.median(ts)), min_s=min(ts), max_s=max(ts), energy_Eh=e.value, clean=ok)
    print(sid, f"{np.median(ts):.3f} s", f"E {e.value:.8f}", flush=True)
json.dump(dict(library=LIB, cases=res), open(OUT, 'w'), indent=1)
