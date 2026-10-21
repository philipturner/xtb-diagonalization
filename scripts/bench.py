#!/usr/bin/env python3
"""Timed cold GFN2 single points through the C API, production settings (accuracy 1, maxiter 250, fresh results object
per single point), one process per case: one untimed warm-up, then the median of --reps timed runs.

  python3 scripts/bench.py LIB OUT.json [--cases id:uhf,id:uhf,...] [--reps 5] [--etemp 300]

Set the thread variables yourself (OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1) and run only on an idle machine.
If LIB was built with the SCC timers (xtb/src/scc_timer.f90), the per-stage breakdown of every timed run is recorded:
seconds per single point in each stage, the overlap factorization, the SCC loop total and everything outside the loop
(integrals, gradient). The timers write one line per cycle, so time wall clock with a build without them.
Also records each case process's peak resident memory (ru_maxrss, includes Python and numpy).
"""
import ctypes as C, json, os, resource, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, '..', 'structures')
DEFAULT = 't2_R:1,facet_closed:0,s12_C31:1,facet_dbpair_tool:1,s3_S42_159:1,diamond233:0'
STAGES = ('potential', 'hbuild', 'eigensolve', 'fermi', 'dmat', 'population', 'mixing')
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}


def parse_timers(text):
    """Timer file of one single point -> dict of seconds per stage, setup, loop total, cycles."""
    cyc = [l.split() for l in text.splitlines() if l.startswith('cycle')]
    setup = [float(l.split()[2]) for l in text.splitlines() if l.startswith('setup')]
    done = [l.split() for l in text.splitlines() if l.startswith('done')]
    t = np.array([[float(v) for v in c[3:]] for c in cyc]) if cyc else np.zeros((0, len(STAGES)))
    r = {s: float(t[:, i].sum()) for i, s in enumerate(STAGES)}
    r.update(setup=sum(setup), scc_loop=sum(float(d[3]) for d in done), cycles=sum(int(d[2]) for d in done))
    return r


def one(lib, sid, uhf, reps, etemp):
    T = open(f"{ST}/{sid}.xyz").read().split('\n'); n = int(T[0])
    el = [l.split()[0] for l in T[2:2 + n]]
    x = np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]]) / 0.52917721067
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule'):
        getattr(L, f).restype = C.c_void_p
    L.xtb_checkEnvironment.restype = C.c_int
    env = C.c_void_p(L.xtb_newEnvironment()); L.xtb_setVerbosity(env, 0)
    mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                      (C.c_double * (3 * n))(*x.ravel()), C.byref(C.c_double(0.0)),
                                      C.byref(C.c_int(uhf)), None, None))
    calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
    L.xtb_setAccuracy(env, calc, C.c_double(1.0)); L.xtb_setElectronicTemp(env, calc, C.c_double(etemp))
    L.xtb_setMaxIter(env, calc, C.c_int(250))
    tfile = os.environ.get('XTB_SCC_TIMERS')
    walls, stages, e = [], [], C.c_double()
    for k in range(reps + 1):
        if tfile:
            open(tfile, 'w').close()
        r = C.c_void_p(L.xtb_newResults())
        t0 = time.perf_counter(); L.xtb_singlepoint(env, mol, calc, r); dt = time.perf_counter() - t0
        L.xtb_getEnergy(env, r, C.byref(e)); L.xtb_delResults(C.byref(r))
        if k:
            walls.append(dt)
            if tfile:
                s = parse_timers(open(tfile).read()); s['outside'] = dt - s['scc_loop']; stages.append(s)
    out = dict(id=sid, uhf=uhf, n_atoms=n, etemp_K=etemp, walls_s=walls, median_s=float(np.median(walls)),
               min_s=min(walls), max_s=max(walls), energy_Eh=e.value, clean=not L.xtb_checkEnvironment(env),
               peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    if stages:
        out['stages_s'] = {k: float(np.median([s[k] for s in stages])) for k in stages[0]}
    print(json.dumps(out), flush=True)


def main():
    a = sys.argv[1:]
    if a[0] == '--one':
        return one(a[1], a[2], int(a[3]), int(a[4]), float(a[5]))
    lib, out = os.path.abspath(a[0]), a[1]
    cases = [c.split(':') for c in (a[a.index('--cases') + 1] if '--cases' in a else DEFAULT).split(',')]
    reps = int(a[a.index('--reps') + 1]) if '--reps' in a else 5
    etemp = float(a[a.index('--etemp') + 1]) if '--etemp' in a else 300.0
    res = []
    for sid, uhf in cases:
        env = dict(os.environ)
        tmp = None
        if 'XTB_SCC_TIMERS' in os.environ:
            tmp = tempfile.NamedTemporaryFile(suffix='.timers', delete=False).name; env['XTB_SCC_TIMERS'] = tmp
        p = subprocess.run([sys.executable, __file__, '--one', lib, sid, uhf, str(reps), str(etemp)], env=env,
                           capture_output=True, text=True)
        if tmp:
            os.unlink(tmp)
        r = json.loads(p.stdout.strip().splitlines()[-1]); res.append(r)
        s = r.get('stages_s')
        print(f"{sid:20s} u{uhf} median {r['median_s']:.3f} s (min {r['min_s']:.3f}, max {r['max_s']:.3f})  peak {r['peak_rss_MB']:.0f} MB"
              + (f"  cycles {s['cycles']:.0f}  eigensolve {s['eigensolve'] / r['median_s']:.0%}" if s else ''), flush=True)
    meta = dict(library=lib, reps=reps, etemp_K=etemp, threads=dict((k, os.environ.get(k)) for k in
                ('OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')), timers='XTB_SCC_TIMERS' in os.environ,
                date=time.strftime('%Y-%m-%d %H:%M'), load_avg=os.getloadavg())
    json.dump(dict(meta=meta, cases=res), open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
