#!/usr/bin/env python3
"""Contract test of the SCC precision C API (xtb_setSCCPrecision / xtb_getSCCPrecision, diagonalization branch).

  python3 scripts/test_sccprecision.py LIB [PRODUCTION_LIB]

LIB must have the API: the default is FP64; settings set on a calculator come back through its results; invalid
values are errors that leave the calculator unchanged; the getter is an error on results without an SCC; GFN-FF
refuses FP32; the effective mode is printed at minimal verbosity; two threads of one process with different settings
each get exactly their serial results (the settings live in the calculator, not in process-global state).
PRODUCTION_LIB (optional, without the API): FP64 runs; sccprecision.apply refuses an FP32 request.
Prints one line per check and PASS/FAIL. Small structure (t2_R, n = 124) and facet_closed (n = 412).
"""
import ctypes as C, os, sys, tempfile, threading
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sccprecision  # noqa: E402

Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
failures = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)
    if not ok:
        failures.append(name)


def load(lib):
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule'):
        getattr(L, f).restype = C.c_void_p
    L.xtb_checkEnvironment.restype = C.c_int
    return L


def setup(L, sid, uhf, verbosity=0, log=None, gfnff=False):
    T = open(f"{HERE}/../structures/{sid}.xyz").read().split('\n'); n = int(T[0])
    el = [l.split()[0] for l in T[2:2 + n]]
    x = np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]]) / 0.52917721067
    env = C.c_void_p(L.xtb_newEnvironment())
    if log:
        L.xtb_setOutput(env, log.encode())
    L.xtb_setVerbosity(env, verbosity)
    mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                      (C.c_double * (3 * n))(*x.ravel()), C.byref(C.c_double(0.0)),
                                      C.byref(C.c_int(uhf)), None, None))
    calc = C.c_void_p(L.xtb_newCalculator())
    (L.xtb_loadGFNFF if gfnff else L.xtb_loadGFN2xTB)(env, mol, calc, None)
    if not gfnff:
        L.xtb_setAccuracy(env, calc, C.c_double(1.0)); L.xtb_setElectronicTemp(env, calc, C.c_double(300.0))
        L.xtb_setMaxIter(env, calc, C.c_int(250))
    return env, mol, calc, n


def error(L, env):
    if not L.xtb_checkEnvironment(env):
        return ''
    b = C.create_string_buffer(4096); L.xtb_getError(env, b, C.byref(C.c_int(4096)))
    return b.value.decode(errors='replace').strip()


def precision(L, env, res):
    r = C.c_double(); i = [C.c_int() for _ in range(5)]
    L.xtb_getSCCPrecision(env, res, C.byref(r), *[C.byref(v) for v in i])
    return (r.value, i[0].value, i[1].value), dict(fp32_cycles=i[2].value, fp64_restart=i[3].value,
                                                   scc_cycles=i[4].value)


def single(L, env, mol, calc, n):
    res = C.c_void_p(L.xtb_newResults())
    L.xtb_singlepoint(env, mol, calc, res)
    e = C.c_double(); g = (C.c_double * (3 * n))()
    L.xtb_getEnergy(env, res, C.byref(e)); L.xtb_getGradient(env, res, g)
    return res, e.value, list(g)


def main():
    lib = os.path.abspath(sys.argv[1])
    L = load(lib)
    check('library exports xtb_setSCCPrecision and xtb_getSCCPrecision',
          hasattr(L, 'xtb_setSCCPrecision') and hasattr(L, 'xtb_getSCCPrecision'))

    env, mol, calc, n = setup(L, 't2_R', 1)
    res, e64, g64 = single(L, env, mol, calc, n)
    got, out = precision(L, env, res)
    check('default calculator runs FP64', got == (0.0, 0, 0) and out['fp32_cycles'] == 0, f'{got} {out}')

    L.xtb_setSCCPrecision(env, calc, C.c_double(1e-3), C.c_int(120), C.c_int(0))
    res, e32, _ = single(L, env, mol, calc, n)
    got, out = precision(L, env, res)
    check('FP32 settings reported back', got == (1e-3, 120, 0) and out['fp32_cycles'] > 0, f'{got} {out}')
    check('FP32 energy within the production gate of FP64', abs(e32 - e64) <= 5e-7, f'dE {abs(e32 - e64):.1e}')

    for args, what in (((-1e-3, 160, 0), 'negative fp32_rmsq'), ((float('nan'), 160, 0), 'NaN fp32_rmsq'),
                       ((1e-3, 0, 0), 'fp32_maxiter 0'), ((1e-3, 160, 2), 'fp32_refine 2'),
                       ((0.0, 160, 1), 'fp32_refine without FP32')):
        e2, m2, c2, n2 = setup(L, 't2_R', 1)
        L.xtb_setSCCPrecision(e2, c2, C.c_double(1e-2), C.c_int(100), C.c_int(0))   # a valid setting first
        L.xtb_setSCCPrecision(e2, c2, C.c_double(args[0]), C.c_int(args[1]), C.c_int(args[2]))
        msg = error(L, e2)
        e3, m3, c3, _ = setup(L, 't2_R', 1)  # fresh environment: the error state stays in e2
        L.xtb_setSCCPrecision(e3, c3, C.c_double(1e-2), C.c_int(100), C.c_int(0))
        L.xtb_setSCCPrecision(e3, c3, C.c_double(args[0]), C.c_int(args[1]), C.c_int(args[2]))
        L.xtb_delEnvironment(C.byref(e3))
        e4 = C.c_void_p(L.xtb_newEnvironment()); L.xtb_setVerbosity(e4, 0)
        r4 = C.c_void_p(L.xtb_newResults()); L.xtb_singlepoint(e4, m3, c3, r4)
        got, _ = precision(L, e4, r4)
        check(f'rejects {what}, calculator unchanged', bool(msg) and got == (1e-2, 100, 0), f'"{msg}" -> {got}')

    e5, m5, c5, _ = setup(L, 't2_R', 1)
    r5 = C.c_void_p(L.xtb_newResults())
    precision(L, e5, r5)
    msg = error(L, e5)
    check('getter is an error on results without an SCC', bool(msg), msg)

    e6, m6, c6, _ = setup(L, 't2_R', 1, gfnff=True)
    L.xtb_setSCCPrecision(e6, c6, C.c_double(1e-3), C.c_int(160), C.c_int(0))
    msg = error(L, e6)
    check('GFN-FF refuses FP32', bool(msg), msg)

    log = tempfile.NamedTemporaryFile(suffix='.out', delete=False).name
    e7, m7, c7, n7 = setup(L, 't2_R', 1, verbosity=1, log=log)
    L.xtb_setSCCPrecision(e7, c7, C.c_double(1e-3), C.c_int(160), C.c_int(0))
    single(L, e7, m7, c7, n7); L.xtb_releaseOutput(e7)
    text = open(log, errors='replace').read(); os.unlink(log)
    line = next((l.strip() for l in text.splitlines() if 'SCC eigensolves' in l), '')
    check('mode printed at minimal verbosity', 'FP32 while rmsq' in line, line)

    # Two threads, one FP32 and one FP64, each against its serial result.
    def run(rmsq, out, key):
        ev, mv, cv, nv = setup(L, 'facet_closed', 0)
        L.xtb_setSCCPrecision(ev, cv, C.c_double(rmsq), C.c_int(160), C.c_int(0))
        for _ in range(3):
            r, e, g = single(L, ev, mv, cv, nv)
            out.setdefault(key, []).append((e, g, precision(L, ev, r)[0]))
    serial, threaded = {}, {}
    run(1e-3, serial, 32); run(0.0, serial, 64)
    ts = [threading.Thread(target=run, args=(1e-3, threaded, 32)), threading.Thread(target=run, args=(0.0, threaded, 64))]
    [t.start() for t in ts]; [t.join() for t in ts]
    same = all(threaded[k][i][:2] == serial[k][0][:2] for k in (32, 64) for i in range(3))
    modes = all(threaded[32][i][2][0] == 1e-3 and threaded[64][i][2][0] == 0.0 for i in range(3))
    check('two threads (FP32 and FP64) each bit-identical to their serial results', same and modes,
          f"FP32 vs FP64 dE {abs(serial[32][0][0] - serial[64][0][0]):.1e}")

    if len(sys.argv) > 2:
        P = load(os.path.abspath(sys.argv[2]))
        check('production library has no SCC precision API', not hasattr(P, 'xtb_setSCCPrecision'))
        ep, mp, cp, nump = setup(P, 't2_R', 1)
        os.environ.pop('XTB_REF_FP32_RMSQ', None)
        sccprecision.apply(P, ep, cp)
        _, ep64, _ = single(P, ep, mp, cp, nump)
        check('production library runs FP64 through sccprecision', ep64 != 0.0 and not error(P, ep))
        os.environ['XTB_REF_FP32_RMSQ'] = '1e-3'
        try:
            sccprecision.apply(P, ep, cp); refused = False
        except SystemExit:
            refused = True
        os.environ.pop('XTB_REF_FP32_RMSQ', None)
        check('FP32 request to the production library is refused', refused)

    print('PASS' if not failures else f"FAIL ({len(failures)}: {', '.join(failures)})")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
