#!/usr/bin/env python3
"""Inclusive share of Time Profiler samples per routine, relative to all samples under xtb_singlepoint.

  xcrun xctrace export --input X.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]' > X.xml
  python3 scripts/callshare.py X.xml [--top 40]

Prints the routines that matter for the eigensolve breakdown (LAPACK/BLAS drivers and xTB's SCC kernels) and the
--top N routines by inclusive share (a routine counts once per sample, however deep it recurs).
"""
import re, sys, xml.etree.ElementTree as ET
from collections import Counter

KEY = ['xtb_singlepoint', '__xtb_scc_core_MOD_scc', '__xtb_mctc_lapack_eigensolve_MOD_mctc_dsygvd_factorized',
       '__xtb_xtb_hamiltonian_MOD_build_dsdqh0_noreset', '__xtb_intgrad_MOD_get_grad_multiint', 'dsygvd_', 'dsyevd_', 'dsygst_', 'dpotrf_', 'dtrsm_', 'dsytrd_',
       'dlatrd_', 'dsymv_', 'dgemv_', 'dsyr2k_', 'dstedc_', 'dlaed4', 'dormtr_', 'dgemm_', 'gomp_parallel', '__xtb_scc_core_MOD_dmat',
       '__xtb_aespot_MOD_mmompop', '__xtb_aespot_MOD_setvsdq', '__xtb_scc_core_MOD_mpopsh',
       '__xtb_scc_core_MOD_buildisoanisotropich1', '__xtb_scc_core_MOD_fermismear', '__xtb_broyden_MOD_broyden']


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index('--top') + 1]) if '--top' in sys.argv else 40
    frames = {}  # frame id -> name
    total, cnt = 0, Counter()
    for _, el in ET.iterparse(path, events=('end',)):
        if el.tag != 'row':
            continue
        names = []
        bt = el.find('backtrace')
        if bt is None or bt.get('ref') is not None:
            # a referenced backtrace: resolve from the cache
            names = frames.get(('bt', bt.get('ref')), []) if bt is not None else []
        else:
            for fr in bt.findall('frame'):
                if fr.get('ref') is not None:
                    names.append(frames.get(fr.get('ref'), '?'))
                else:
                    frames[fr.get('id')] = fr.get('name', '?'); names.append(frames[fr.get('id')])
            frames[('bt', bt.get('id'))] = names
        # strip LAPACK/BLAS variant prefixes such as APL_ / $NEWLAPACK suffix for matching
        norm = {re.sub(r'\$.*$', '', n).lower() for n in names}
        if any(n.startswith('xtb_singlepoint') for n in norm):
            total += 1
            cnt.update(norm)
        el.clear()
    print(f"samples under xtb_singlepoint: {total} (1 ms each)")
    for k in KEY:
        base = k.lower().rstrip('_')
        hits = sum(v for n, v in cnt.items() if n.rstrip('_') in (base, 'cblas_' + base))
        print(f"  {k:42s} {hits / total:6.1%}")
    print(f"\ntop {top} by inclusive share:")
    for n, v in cnt.most_common(top):
        print(f"  {n[:70]:70s} {v / total:6.1%}")


if __name__ == '__main__':
    main()
