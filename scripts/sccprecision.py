"""SCC eigensolve precision for the driver scripts (make_reference, bench, sweep, basin), through the C API of the
diagonalization branch: xtb_setSCCPrecision on the calculator, xtb_getSCCPrecision on the results.

Command line: --fp32 RMSQ [--fp32-maxiter N] [--fp32-refine] (default FP64). parse() moves them into the variables
XTB_REF_FP32_RMSQ / _MAXITER / _REFINE so that worker processes inherit them; these belong to the scripts, the library
never reads them. A library without xtb_setSCCPrecision runs FP64 only: requesting FP32 from it is an error. Whenever the
library has xtb_getSCCPrecision, check() compares what the results report with what was set.
"""
import ctypes as C
import os


def parse(a):
    """Remove the precision flags from the argument list a (in place) and export them for worker processes."""
    for flag, var in (('--fp32', 'XTB_REF_FP32_RMSQ'), ('--fp32-maxiter', 'XTB_REF_FP32_MAXITER')):
        if flag in a:
            i = a.index(flag); os.environ[var] = a[i + 1]; del a[i:i + 2]
    if '--fp32-refine' in a:
        a.remove('--fp32-refine'); os.environ['XTB_REF_FP32_REFINE'] = '1'
    return requested()


def requested():
    """(fp32_rmsq, fp32_maxiter, fp32_refine) requested; fp32_rmsq 0 = FP64 only."""
    return (float(os.environ.get('XTB_REF_FP32_RMSQ', '0')), int(os.environ.get('XTB_REF_FP32_MAXITER', '160')),
            int(os.environ.get('XTB_REF_FP32_REFINE', '0')))


def label():
    r, m, f = requested()
    return 'FP64' if r == 0 else f"FP32 rmsq {r:g}, maxiter {m}" + (', refined' if f else '')


def apply(L, env, calc):
    """Set the requested precision on the calculator; FP32 from a library without the setter is an error."""
    want = requested()
    if not hasattr(L, 'xtb_setSCCPrecision'):
        if want[0] > 0:
            raise SystemExit(f'{L._name} has no xtb_setSCCPrecision: FP32 eigensolves cannot be requested')
        return
    L.xtb_setSCCPrecision(env, calc, C.c_double(want[0]), C.c_int(want[1]), C.c_int(want[2]))


def check(L, env, res):
    """What the results report (dict), or None if the library can't tell; raises if it differs from the request."""
    if not hasattr(L, 'xtb_getSCCPrecision'):
        return None
    r = C.c_double(); i = [C.c_int() for _ in range(5)]
    L.xtb_getSCCPrecision(env, res, C.byref(r), *[C.byref(v) for v in i])
    got = dict(fp32_rmsq=r.value, fp32_maxiter=i[0].value, fp32_refine=i[1].value, fp32_cycles=i[2].value,
               fp64_restart=i[3].value, scc_cycles=i[4].value)
    if got['scc_cycles'] == 0:
        raise RuntimeError('results report no SCC (xtb_getSCCPrecision): the single point did not run its SCC')
    want = requested()
    want = (want[0], want[1] if want[0] > 0 else 0, want[2])
    if (got['fp32_rmsq'], got['fp32_maxiter'], got['fp32_refine']) != want:
        raise RuntimeError(f'SCC precision mismatch: set {want}, results report {got}')
    return got
