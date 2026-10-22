#!/usr/bin/env python3
"""Mac: reference GFN2 results for the diagonalization test set (structures/MANIFEST.tsv), through the xTB C API exactly
as production calls it: fresh results object per single point (cold SCF), accuracy 1.0, maxiter 250.
Library: --lib PATH, else $XTB_LIB, else the PRODUCTION baseline (xtb 6.7.1 + Accelerate, behind reference/), archived at
../claude-molecular-renderer/.build/archive/2026-10-22-xtb-production/libxtb.dylib
(read-only). The mechanosynthesis repo's root libxtb.dylib is this fork's diagonalization build since 2026-10-22.
--out DIR writes somewhere other than reference/ (use it for every non-production library).

Settings grid per structure: electronic temperature 100 / 300 (production) / 1000 K, uhf = the two lowest spin states
consistent with the electron count (0 and 2 if even, 1 and 3 if odd).
Per run -> reference/<id>_u<uhf>_T<etemp>.json: energy (Eh), gradient (Eh/bohr, n x 3), Mulliken charges (e), orbital
eigenvalues (Eh) and occupations (alpha + beta summed, as the API returns them), HOMO-LUMO gap (Eh, from occupations
see 'gap_def'), SCC iterations (parsed from xtb's own output), converged flag, error text, wall time,
library path + sha256. Plus reference/SUMMARY.tsv.

  python3 scripts/make_reference.py [--lib PATH] [--out DIR] [--accuracy 1.0] [--workers 4] [--only ID] [--one ID UHF ETEMP]

--accuracy changes xtb_setAccuracy for validation runs only (e.g. 0.01 for the tight-accuracy gate); production is 1.0.
--threads sets OMP_NUM_THREADS and VECLIB_MAXIMUM_THREADS per worker (default 1; production Python uses 2).
--fp32 RMSQ [--fp32-maxiter N] [--fp32-refine]: FP32 SCC eigensolves through xtb_setSCCPrecision (sccprecision.py;
diagonalization branch, a library without it is an error). Whenever the library has xtb_getSCCPrecision, each run
records what the results report ('scc_precision') and a mismatch with what was set is an error (default set: FP64).
"""
import ctypes as C, hashlib, json, os, re, subprocess, sys, tempfile, time
import numpy as np
import sccprecision

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIB = os.path.abspath(os.environ.get('XTB_LIB') or os.path.join(ROOT, '../claude-molecular-renderer/.build/archive/2026-10-22-xtb-production/libxtb.dylib'))
ST, REF = f"{ROOT}/structures", f"{ROOT}/reference"
BOHR = 0.52917721067
Z = {'H': 1, 'C': 6, 'O': 8, 'Si': 14, 'Ge': 32}
VE = {'H': 1, 'C': 4, 'O': 6, 'Si': 4, 'Ge': 4}
ETEMPS = (100.0, 300.0, 1000.0)
ACC, MAXIT = float(os.environ.get('XTB_REF_ACCURACY', '1.0')), 250


def read(fn):
    L = open(fn).read().split('\n'); n = int(L[0])
    return [l.split()[0] for l in L[2:2 + n]], np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])


def cases():
    rows = [l.rstrip('\n').split('\t') for l in open(f"{ST}/MANIFEST.tsv")][1:]
    files = sorted({r[1] for r in rows})
    out = []
    for f in files:
        el, _ = read(f"{ST}/{f}")
        ne = sum(VE[e] for e in el)
        for u in ((0, 2) if ne % 2 == 0 else (1, 3)):
            for t in ETEMPS:
                out.append((f[:-4], u, t))
    return out


def one(sid, uhf, etemp):
    el, x = read(f"{ST}/{sid}.xyz"); n = len(el)
    L = C.CDLL(LIB)
    for f in ('xtb_newEnvironment', 'xtb_newCalculator', 'xtb_newResults', 'xtb_newMolecule'):
        getattr(L, f).restype = C.c_void_p
    for f in ('xtb_checkEnvironment', 'xtb_getAPIVersion'):
        getattr(L, f).restype = C.c_int
    env = C.c_void_p(L.xtb_newEnvironment())
    log = tempfile.NamedTemporaryFile(suffix='.out', delete=False).name
    L.xtb_setOutput(env, log.encode()); L.xtb_setVerbosity(env, 2)  # full output: SCC iteration table
    mol = C.c_void_p(L.xtb_newMolecule(env, C.byref(C.c_int(n)), (C.c_int * n)(*[Z[s] for s in el]),
                                      (C.c_double * (3 * n))(*(x / BOHR).ravel()), C.byref(C.c_double(0.0)),
                                      C.byref(C.c_int(uhf)), None, None))
    calc = C.c_void_p(L.xtb_newCalculator())
    L.xtb_loadGFN2xTB(env, mol, calc, None)
    L.xtb_setAccuracy(env, calc, C.c_double(ACC)); L.xtb_setElectronicTemp(env, calc, C.c_double(etemp))
    L.xtb_setMaxIter(env, calc, C.c_int(MAXIT))
    sccprecision.apply(L, env, calc)

    def err():
        if not L.xtb_checkEnvironment(env):
            return ''
        b = C.create_string_buffer(4096); L.xtb_getError(env, b, C.byref(C.c_int(4096)))
        return b.value.decode(errors='replace').strip().replace('\n', ' | ')
    setup_msg = err()
    res = C.c_void_p(L.xtb_newResults())  # cold start, as production
    t0 = time.time(); L.xtb_singlepoint(env, mol, calc, res); wall = time.time() - t0
    msg = err()
    prec = None
    try:
        prec = sccprecision.check(L, env, res)
    except RuntimeError as x:
        msg = msg or str(x)
    msg = msg or err()
    e = C.c_double(); g = (C.c_double * (3 * n))(); q = (C.c_double * n)()
    L.xtb_getEnergy(env, res, C.byref(e)); L.xtb_getGradient(env, res, g); L.xtb_getCharges(env, res, q)
    nao = C.c_int(); L.xtb_getNao(env, res, C.byref(nao))
    ev = (C.c_double * nao.value)(); oc = (C.c_double * nao.value)()
    L.xtb_getOrbitalEigenvalues(env, res, ev); L.xtb_getOrbitalOccupations(env, res, oc)
    msg2 = err()
    L.xtb_delResults(C.byref(res)); L.xtb_delCalculator(C.byref(calc)); L.xtb_delMolecule(C.byref(mol))
    L.xtb_releaseOutput(env); L.xtb_delEnvironment(C.byref(env))
    out = open(log, errors='replace').read(); os.unlink(log)
    it = [int(m) for m in re.findall(r'^\s+(\d+)\s+-?\d+\.\d+\s+-?\d+\.\d+E[-+]\d+', out, re.M)]
    ev, oc = np.array(ev), np.array(oc)
    occ = np.where(oc > 0.5)[0]; vir = np.where(oc < 0.5)[0]
    gap = float(ev[vir].min() - ev[occ].max()) if len(occ) and len(vir) else None
    r = dict(id=sid, uhf=uhf, charge=0, etemp_K=etemp, accuracy=ACC, maxiter=MAXIT, n_atoms=n, nao=nao.value,
             energy_Eh=e.value, gradient_Eh_bohr=np.array(g).reshape(n, 3).tolist(), charges_e=list(q),
             orbital_eigenvalues_Eh=ev.tolist(), orbital_occupations=oc.tolist(), gap_Eh=gap,
             gap_def='lowest eigenvalue with occupation < 0.5 minus highest with > 0.5 (alpha+beta summed occupations)',
             scc_iterations=max(it) if it else None, converged=('SCC converged' in out or 'convergence criteria satisfied' in out) and not msg,
             error=' | '.join(m for m in (setup_msg, msg, msg2) if m), wall_s=wall, scc_precision=prec,
             library=LIB, library_sha256=hashlib.sha256(open(LIB, 'rb').read()).hexdigest(),
             api_version=L.xtb_getAPIVersion(), threads=os.environ.get('OMP_NUM_THREADS'))
    os.makedirs(REF, exist_ok=True)
    json.dump(r, open(f"{REF}/{sid}_u{uhf}_T{etemp:g}.json", 'w'))
    print(f"{sid} u{uhf} T{etemp:g}: E {e.value:.10f} it {r['scc_iterations']} conv {r['converged']} {wall:.1f} s {r['error'][:80]}", flush=True)


def summary():
    rows = [json.load(open(f"{REF}/{f}")) for f in sorted(os.listdir(REF)) if f.endswith('.json')]
    with open(f"{REF}/SUMMARY.tsv", 'w') as f:
        f.write("id\tuhf\tetemp_K\tnao\tenergy_Eh\tmax|grad|_Eh_bohr\tgap_eV\tscc_iterations\tconverged\twall_s\terror\n")
        for r in sorted(rows, key=lambda r: (r['nao'], r['id'], r['uhf'], r['etemp_K'])):
            gm = float(np.abs(np.array(r['gradient_Eh_bohr'])).max())
            gap = f"{r['gap_Eh'] * 27.211386:.4f}" if r['gap_Eh'] is not None else ''
            f.write(f"{r['id']}\t{r['uhf']}\t{r['etemp_K']:g}\t{r['nao']}\t{r['energy_Eh']:.10f}\t{gm:.6e}\t{gap}\t"
                    f"{r['scc_iterations']}\t{r['converged']}\t{r['wall_s']:.2f}\t{r['error'][:120]}\n")
    print(f"{len(rows)} runs -> {REF}/SUMMARY.tsv")


def main():
    global LIB, REF, ACC
    a = sys.argv[1:]
    if '--accuracy' in a:
        i = a.index('--accuracy'); ACC = float(a[i + 1]); os.environ['XTB_REF_ACCURACY'] = a[i + 1]; del a[i:i + 2]
    if '--lib' in a:
        i = a.index('--lib'); LIB = os.path.abspath(a[i + 1]); os.environ['XTB_LIB'] = LIB; del a[i:i + 2]
    sccprecision.parse(a)
    if '--out' in a:
        i = a.index('--out'); REF = os.path.abspath(a[i + 1]); os.environ['XTB_REF_OUT'] = REF; del a[i:i + 2]
    REF = os.environ.get('XTB_REF_OUT', REF)
    if not os.path.exists(LIB):
        raise SystemExit(f'no library at {LIB}')
    if a[:1] != ['--one']:
        print(f'library {LIB}, SCC eigensolves {sccprecision.label()}', flush=True)
    if a[:1] == ['--one']:
        return one(a[1], int(a[2]), float(a[3]))
    W = int(a[a.index('--workers') + 1]) if '--workers' in a else 4
    T = a[a.index('--threads') + 1] if '--threads' in a else '1'
    todo = [c for c in cases() if '--only' not in a or c[0] == a[a.index('--only') + 1]]
    todo = [c for c in todo if not os.path.exists(f"{REF}/{c[0]}_u{c[1]}_T{c[2]:g}.json")]
    todo.sort(key=lambda c: -os.path.getsize(f"{ST}/{c[0]}.xyz"))  # largest first
    env = dict(os.environ, OMP_NUM_THREADS=T, VECLIB_MAXIMUM_THREADS=T, OMP_STACKSIZE='2G')
    run = []
    while todo or run:
        while todo and len(run) < W:
            c = todo.pop(0)
            run.append(subprocess.Popen([sys.executable, __file__, '--one', c[0], str(c[1]), str(c[2])], env=env))
        time.sleep(0.5)
        run = [p for p in run if p.poll() is None]
    summary()


if __name__ == '__main__':
    main()
