#!/usr/bin/env python3
"""Tables for the two benchmark cases (scripts/bench_cases.sh): latency of one cold single point with 8 OpenMP threads
(bench.py, median of 5), and throughput of 8 processes x 1 thread (sweep.py, jobs/s), per library and SCC precision.

  python3 scripts/cases_table.py [DIR]     (default build/cases)

Libraries: prod (the renderer's production libxtb), e2e5bee (the diagonalization tip before 2026-10-21's speedups),
par (xtb d0691d6); par1 / prod1: 1 thread. Latency ratios are against prod with 1 thread when present, else prod with
8; throughput ratios against prod FP64.
"""
import glob, json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'build', 'cases')
VARIANTS = [('prod', 'fp64'), ('e2e5bee', 'fp64'), ('e2e5bee', 'fp32'), ('par', 'fp64'), ('par', 'fp32')]


def latency(prefix):
    out = {}
    for lib, mode in VARIANTS + [('prod1', 'fp64'), ('par1', 'fp64'), ('par1', 'fp32')]:
        name = f'latency1-{lib[:-1]}-{mode}.json' if lib.endswith('1') else f'{prefix}-{lib}-{mode}.json'
        f = os.path.join(D, name)
        if os.path.exists(f):
            out[(lib, mode)] = {c['id']: c for c in json.load(open(f))['cases']}
    return out


def main():
    lat = latency('latency8')
    if lat:
        ids = list(next(iter(lat.values())).keys())
        print('Case B, latency: one cold single point, OMP_NUM_THREADS=8 (prod1, par1: 1 thread), ms, median of 5')
        cols = [k for k in [('prod1', 'fp64')] + VARIANTS + [('par1', 'fp64'), ('par1', 'fp32')] if k in lat]
        print(f"{'case':20s}" + ''.join(f"{l + ' ' + m:>15s}" for l, m in cols))
        for i in ids:
            ref = ('prod1', 'fp64') if ('prod1', 'fp64') in lat else ('prod', 'fp64')
            base = lat[ref][i]['median_s'] if ref in lat else None
            row = f"{i:20s}"
            for k in cols:
                t = lat[k][i]['median_s']
                row += f"{1e3 * t:9.0f}" + (f" {base / t:4.2f}x" if base else '      ')
            print(row)
    files = sorted(glob.glob(os.path.join(D, 'tput-*.json')))
    if files:
        print('\nCase A, throughput: 8 processes x 1 thread, jobs/s (x: vs prod FP64)')
        cases = sorted({os.path.basename(f).split('-')[1] for f in files})
        print(f"{'case':20s}" + ''.join(f"{l + ' ' + m:>15s}" for l, m in VARIANTS))
        for c in cases:
            row, base = f"{c:20s}", None
            for lib, mode in VARIANTS:
                f = os.path.join(D, f'tput-{c}-{lib}-{mode}.json')
                if not os.path.exists(f):
                    row += f"{'-':>15s}"
                    continue
                r = [x for x in json.load(open(f))['configs'] if x['procs'] == 8][0]['jobs_per_s']
                base = base or (r if (lib, mode) == ('prod', 'fp64') else None)
                row += f"{r:9.3f}" + (f" {r / base:4.2f}x" if base else '      ')
            print(row)


if __name__ == '__main__':
    main()
