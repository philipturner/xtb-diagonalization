#!/usr/bin/env python3
"""Per-run SCC noise floor (BRIEF.md, Phase 2 step 0): how far a converged single point moves when only its starting
wavefunction changes.

  python3 scripts/noise.py LIB OUTDIR [--accuracy 1] [--workers 4] [--only ID]

For every reference run (structure x uhf x 100/300/1000 K, production settings except --accuracy), the single point is
computed cold (fresh results object) and again from perturbed starts, each a copy (xtb_copyResults) of another
converged result:
  T<etemp>   the same structure and uhf at a neighbouring electronic temperature (100 <-> 300 <-> 1000 K)
  <relative> the nearest displaced copy at the same uhf and temperature (base <-> _rms05, _rms15 <- _rms05)
Seeds that did not converge themselves are skipped. Per perturbed start: differences from the cold run in energy, gradient, gap, charges and occupations, SCC cycles and
convergence. A start counts toward the noise floor only if it reaches the cold run's SCF solution (max |d occupation|
< 0.05, |d gap| < 1e-3 Eh, |dE| < 1e-5 Eh, max |d charge| < 5e-3 e); others are listed as 'other solution'. Writes OUTDIR/<id>_u<uhf>.json and
OUTDIR/NOISE.tsv (per run: maxima over same-solution starts).
"""
import ctypes as C, json, os, re, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.join(HERE, '..', 'structures')
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
VE = {'H': 1, 'C': 4, 'O': 6, 'Si': 4, 'Ge': 4}
ETEMPS = (100.0, 300.0, 1000.0)
NEIGHBOUR_T = {100.0: (300.0,), 300.0: (100.0, 1000.0), 1000.0: (300.0,)}
KEYS = ('dE', 'dg', 'dgap', 'dq', 'docc')


def same_solution(d):
    """Same SCF solution as the cold run. The charge test catches near-degenerate solutions: facet_closed_rms15 u2
    at 300 K has two, 1.3e-6 Eh apart, whose charges differ by 2.6e-2 e and gradients by 2e-3 Eh/bohr."""
    return d['docc'] < 0.05 and d['dgap'] < 1e-3 and d['dE'] < 1e-5 and d['dq'] < 5e-3


def read(sid):
    T = open(f"{ST}/{sid}.xyz").read().split('\n'); n = int(T[0])
    return [l.split()[0] for l in T[2:2 + n]], np.array([[float(v) for v in l.split()[1:4]] for l in T[2:2 + n]])


def relative(sid):
    """Nearest displaced copy to seed from, or None."""
    files = {f[:-4] for f in os.listdir(ST) if f.endswith('.xyz')}
    if sid.endswith('_rms05'):
        return sid[:-6]
    if sid.endswith('_rms15'):
        return sid[:-6] + '_rms05'
    return sid + '_rms05' if sid + '_rms05' in files else None


def groups():
    files = sorted(f[:-4] for f in os.listdir(ST) if f.endswith('.xyz'))
    out = []
    for f in files:
        el, _ = read(f)
        ne = sum(VE[e] for e in el)
        for u in ((0, 2) if ne % 2 == 0 else (1, 3)):
            out.append((f, u))
    return out


def one(lib, sid, uhf, acc, outdir):
    L = C.CDLL(lib)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule', 'xtb_copyResults'):
        getattr(L, f).restype = C.c_void_p
    L.xtb_checkEnvironment.restype = C.c_int
    L.xtb_copyResults.argtypes = [C.c_void_p]
    log = tempfile.NamedTemporaryFile(suffix='.out', delete=False).name
    env = C.c_void_p(L.xtb_newEnvironment())

    def setup(s):
        el, x = read(s); n = len(el)
        mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[e] for e in el]),
                                          (C.c_double * (3 * n))(*(x / 0.52917721067).ravel()),
                                          C.byref(C.c_double(0.0)), C.byref(C.c_int(uhf)), None, None))
        calc = C.c_void_p(L.xtb_newCalculator()); L.xtb_loadGFN2xTB(env, mol, calc, None)
        L.xtb_setAccuracy(env, calc, C.c_double(acc)); L.xtb_setMaxIter(env, calc, C.c_int(250))
        return n, mol, calc

    def sp(n, mol, calc, etemp, res):
        # a fresh environment per single point: an SCF failure leaves an error in the environment, and later
        # single points on that environment would not run
        env = C.c_void_p(L.xtb_newEnvironment())
        L.xtb_setElectronicTemp(env, calc, C.c_double(etemp))
        L.xtb_setOutput(env, log.encode()); L.xtb_setVerbosity(env, 2)
        L.xtb_singlepoint(env, mol, calc, res)
        L.xtb_releaseOutput(env)
        ok = not L.xtb_checkEnvironment(env)
        txt = open(log, errors='replace').read(); open(log, 'w').close()
        it = [int(m) for m in re.findall(r'^\s+(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+E[-+]\d+', txt, re.M)]
        e = C.c_double(); L.xtb_getEnergy(env, res, C.byref(e))
        g = (C.c_double * (3 * n))(); q = (C.c_double * n)()
        L.xtb_getGradient(env, res, g); L.xtb_getCharges(env, res, q)
        nao = C.c_int(); L.xtb_getNao(env, res, C.byref(nao))
        ev = (C.c_double * nao.value)(); oc = (C.c_double * nao.value)()
        L.xtb_getOrbitalEigenvalues(env, res, ev); L.xtb_getOrbitalOccupations(env, res, oc)
        ev, oc = np.array(ev), np.array(oc)
        occ, vir = np.where(oc > 0.5)[0], np.where(oc < 0.5)[0]
        gap = float(ev[vir].min() - ev[occ].max()) if len(occ) and len(vir) else 0.0
        L.xtb_delEnvironment(C.byref(env))
        return dict(E=e.value, g=np.array(g), q=np.array(q), occ=oc, gap=gap, cycles=max(it) if it else None,
                    converged=ok and ('SCC converged' in txt or 'convergence criteria satisfied' in txt))

    n, mol, calc = setup(sid)
    cold = {}
    for t in ETEMPS:  # cold results, kept as seeds for the neighbouring temperatures
        r = C.c_void_p(L.xtb_newResults()); cold[t] = (r, sp(n, mol, calc, t, r))
    rel = relative(sid)
    relres = {}
    if rel:
        rn, rmol, rcalc = setup(rel)
        for t in ETEMPS:
            r = C.c_void_p(L.xtb_newResults()); w = sp(rn, rmol, rcalc, t, r)
            if w['converged']:
                relres[t] = r
    runs = []
    for t in ETEMPS:
        c = cold[t][1]
        # seeds must be converged results; an unconverged seed says nothing about noise
        starts = [(f"T{tt:g}", cold[tt][0]) for tt in NEIGHBOUR_T[t] if cold[tt][1]['converged']] \
            + ([(rel, relres[t])] if t in relres else [])
        out = []
        for name, seed in starts:
            r = C.c_void_p(L.xtb_copyResults(seed)); w = sp(n, mol, calc, t, r); L.xtb_delResults(C.byref(r))
            d = dict(dE=abs(w['E'] - c['E']), dg=float(np.abs(w['g'] - c['g']).max()), dgap=abs(w['gap'] - c['gap']),
                     dq=float(np.abs(w['q'] - c['q']).max()), docc=float(np.abs(w['occ'] - c['occ']).max()))
            same = same_solution(d)
            out.append(dict(start=name, cycles=w['cycles'], converged=w['converged'], same_solution=same, **d))
        same = [o for o in out if o['same_solution'] and o['converged']]
        runs.append(dict(etemp_K=t, cold_cycles=c['cycles'], cold_converged=c['converged'], starts=out,
                         noise={k: (max(o[k] for o in same) if same else None) for k in KEYS}))
    json.dump(dict(id=sid, uhf=uhf, accuracy=acc, library=lib, relative=rel, runs=runs),
              open(f"{outdir}/{sid}_u{uhf}.json", 'w'), indent=1)
    os.unlink(log)
    print(f"{sid} u{uhf}: " + "  ".join(f"T{r['etemp_K']:g} dg {r['noise']['dg'] if r['noise']['dg'] is not None else float('nan'):.1e}"
                                        for r in runs), flush=True)


def table(outdir):
    rows = []
    for f in sorted(os.listdir(outdir)):
        if not f.endswith('.json'):
            continue
        D = json.load(open(f"{outdir}/{f}"))

        for r in D['runs']:
            for o in r['starts']:  # re-apply the current criterion to stored differences
                o['same_solution'] = same_solution(o)
            same = [o for o in r['starts'] if o['same_solution'] and o['converged']]
            r['noise'] = {k: (max(o[k] for o in same) if same else None) for k in KEYS}
            other = [o['start'] for o in r['starts'] if not o['same_solution']]
            nonconv = [o['start'] for o in r['starts'] if r['cold_converged'] and not o['converged']]
            rows.append((D['id'], D['uhf'], r['etemp_K'], r['cold_converged'], r['noise'], len(r['starts']), other, nonconv))
    with open(f"{outdir}/NOISE.tsv", 'w') as f:
        f.write("id\tuhf\tetemp_K\tcold_converged\tstarts\t" + "\t".join(KEYS) + "\tother_solution\tnot_converged\n")
        for sid, u, t, cc, nz, ns, oth, nc in rows:
            f.write(f"{sid}\t{u}\t{t:g}\t{cc}\t{ns}\t" + "\t".join('' if nz[k] is None else f"{nz[k]:.3e}" for k in KEYS)
                    + f"\t{','.join(oth)}\t{','.join(nc)}\n")
    print(f"{len(rows)} runs -> {outdir}/NOISE.tsv")


def main():
    a = sys.argv[1:]
    if a[0] == '--one':
        return one(a[1], a[2], int(a[3]), float(a[4]), a[5])
    lib, outdir = os.path.abspath(a[0]), os.path.abspath(a[1])
    acc = float(a[a.index('--accuracy') + 1]) if '--accuracy' in a else 1.0
    W = int(a[a.index('--workers') + 1]) if '--workers' in a else 4
    os.makedirs(outdir, exist_ok=True)
    todo = [g for g in groups() if '--only' not in a or g[0] == a[a.index('--only') + 1]]
    todo = [g for g in todo if not os.path.exists(f"{outdir}/{g[0]}_u{g[1]}.json")]
    todo.sort(key=lambda g: -os.path.getsize(f"{ST}/{g[0]}.xyz"))  # largest first
    env = dict(os.environ, OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    run = []
    while todo or run:
        while todo and len(run) < W:
            sid, u = todo.pop(0)
            run.append(subprocess.Popen([sys.executable, __file__, '--one', lib, sid, str(u), str(acc), outdir], env=env))
        time.sleep(0.5)
        run = [p for p in run if p.poll() is None]
    table(outdir)


if __name__ == '__main__':
    main()
