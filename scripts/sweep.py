#!/usr/bin/env python3
"""Concurrency sweep: throughput of N concurrent processes x T threads running cold GFN2 single points.

  python3 scripts/sweep.py LIB OUT.json --case id:uhf [--configs 1x1,2x1,4x1,8x1,10x1,5x2,6x2] [--jobs K]

Each worker process loads LIB, runs one untimed warm-up single point, waits for a common start time, then runs K cold
single points (fresh results object each, production settings). Threads per process are set through
OMP_NUM_THREADS and VECLIB_MAXIMUM_THREADS. Throughput = N*K / (time until the last worker finishes).
Per-worker elapsed times are recorded: on M1 Max (8 P-cores, 2 E-cores) a worker much slower than the rest most
likely ran on an E-core.
--fp32 RMSQ [--fp32-maxiter N] [--fp32-refine]: FP32 SCC eigensolves (sccprecision.py), checked on every single point
when the library can report it.
"""
import json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def worker(lib, sid, uhf, jobs, start):
    import ctypes as C
    T = open(f"{HERE}/../structures/{sid}.xyz").read().split('\n'); n = int(T[0])
    Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
    el = [l.split()[0] for l in T[2:2 + n]]
    x = np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]]) / 0.52917721067
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule'):
        getattr(L, f).restype = C.c_void_p
    env = C.c_void_p(L.xtb_newEnvironment()); L.xtb_setVerbosity(env, 0)
    mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                      (C.c_double * (3 * n))(*x.ravel()), C.byref(C.c_double(0.0)),
                                      C.byref(C.c_int(uhf)), None, None))
    calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
    L.xtb_setAccuracy(env, calc, C.c_double(1.0)); L.xtb_setElectronicTemp(env, calc, C.c_double(300.0))
    L.xtb_setMaxIter(env, calc, C.c_int(250))
    import sccprecision
    sccprecision.apply(L, env, calc)

    def sp():
        r = C.c_void_p(L.xtb_newResults()); L.xtb_singlepoint(env, mol, calc, r); sccprecision.check(L, env, r)
        L.xtb_delResults(C.byref(r))
    sp()  # warm-up
    late = time.time() > start
    while time.time() < start:
        time.sleep(0.001)
    t0 = time.time(); per = []
    for _ in range(jobs):
        a = time.time(); sp(); per.append(time.time() - a)
    print(json.dumps(dict(begin=t0, end=time.time(), per_job=per, late=late)), flush=True)


def main():
    a = sys.argv[1:]
    sys.path.insert(0, HERE)
    import sccprecision
    sccprecision.parse(a)
    if a[0] == '--worker':
        return worker(a[1], a[2], int(a[3]), int(a[4]), float(a[5]))
    lib, out = os.path.abspath(a[0]), a[1]
    sid, uhf = a[a.index('--case') + 1].split(':')
    configs = [tuple(int(v) for v in c.split('x')) for c in
               (a[a.index('--configs') + 1] if '--configs' in a else '1x1,2x1,4x1,8x1,10x1,5x2,6x2').split(',')]
    jobs = int(a[a.index('--jobs') + 1]) if '--jobs' in a else 6
    res = []
    for nproc, nthr in configs:
        env = dict(os.environ, OMP_NUM_THREADS=str(nthr), VECLIB_MAXIMUM_THREADS=str(nthr))
        start = time.time() + 8.0 + 0.5 * nproc  # time for every worker to load and warm up
        ps = [subprocess.Popen([sys.executable, __file__, '--worker', lib, sid, uhf, str(jobs), str(start)], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for _ in range(nproc)]
        w = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in ps]
        span = max(x['end'] for x in w) - min(x['begin'] for x in w)
        el = sorted(x['end'] - x['begin'] for x in w)
        r = dict(procs=nproc, threads=nthr, jobs_per_proc=jobs, span_s=span, jobs_per_s=nproc * jobs / span,
                 worker_elapsed_s=el, median_job_s=float(np.median([t for x in w for t in x['per_job']])),
                 any_late=any(x['late'] for x in w))
        res.append(r)
        print(f"{sid} {nproc:2d} x {nthr} thr: {r['jobs_per_s']:.3f} jobs/s, median job {r['median_job_s']:.3f} s, "
              f"worker elapsed {el[0]:.2f}-{el[-1]:.2f} s{'  (a worker started late)' if r['any_late'] else ''}", flush=True)
    json.dump(dict(library=lib, scc_eigensolves=sccprecision.label(), case=sid, uhf=int(uhf), load_avg=os.getloadavg(), date=time.strftime('%Y-%m-%d %H:%M'),
                   configs=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
