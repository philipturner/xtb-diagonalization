#!/usr/bin/env python3
"""Check a run folder (make_reference.py --out DIR) against reference/ with the BRIEF.md validation gates.

  python3 scripts/compare.py DIR [--gates exact|approx] [--ref REFDIR] [--all]

exact  = Phase 2 (reformulation): |dE| <= 1e-10 Eh, max|dg| <= 1e-9 Eh/bohr, gap 1e-8 Eh, charges 1e-8 e,
         SCC cycles unchanged, occupations 1e-6.
approx = Phases 3-4: 1e-7, 1e-6, 1e-5, 1e-5, at most 1 more cycle, occupations 1e-3.
Both: every run converged in the reference still converges. Prints the failing runs (all runs with --all), the
worst deviation per quantity, and PASS/FAIL.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATES = {'exact': dict(E=1e-10, g=1e-9, gap=1e-8, q=1e-8, occ=1e-6, extra_it=0),
         'approx': dict(E=1e-7, g=1e-6, gap=1e-5, q=1e-5, occ=1e-3, extra_it=1)}


def main():
    a = sys.argv[1:]
    run = a[0]
    gate = a[a.index('--gates') + 1] if '--gates' in a else 'exact'
    ref = a[a.index('--ref') + 1] if '--ref' in a else f"{ROOT}/reference"
    G = GATES[gate]
    names = sorted(f for f in os.listdir(ref) if f.endswith('.json'))
    worst = dict(E=0.0, g=0.0, gap=0.0, q=0.0, occ=0.0)
    nfail = nmiss = 0
    print(f"{'run':38s} {'dE':>9s} {'dg':>9s} {'dgap':>9s} {'dq':>9s} {'docc':>9s} {'it':>8s} conv")
    for f in names:
        R = json.load(open(f"{ref}/{f}"))
        if not os.path.exists(f"{run}/{f}"):
            nmiss += 1
            continue
        X = json.load(open(f"{run}/{f}"))
        d = dict(E=abs(X['energy_Eh'] - R['energy_Eh']),
                 g=float(np.abs(np.array(X['gradient_Eh_bohr']) - np.array(R['gradient_Eh_bohr'])).max()),
                 gap=abs((X['gap_Eh'] or 0.0) - (R['gap_Eh'] or 0.0)) if (X['gap_Eh'] is None) == (R['gap_Eh'] is None) else np.inf,
                 q=float(np.abs(np.array(X['charges_e']) - np.array(R['charges_e'])).max()),
                 occ=float(np.abs(np.array(X['orbital_occupations']) - np.array(R['orbital_occupations'])).max()))
        it_ok = (X['scc_iterations'] is not None and R['scc_iterations'] is not None
                 and 0 <= X['scc_iterations'] - R['scc_iterations'] <= G['extra_it']) if gate == 'approx' \
            else X['scc_iterations'] == R['scc_iterations']
        conv_ok = X['converged'] or not R['converged']
        # a run that did not converge in the reference is compared, but only its convergence status is gated
        ok = conv_ok and (not R['converged'] or (all(d[k] <= G[k] for k in d) and it_ok))
        if R['converged']:
            for k in d:
                worst[k] = max(worst[k], d[k])
        nfail += not ok
        if not ok or '--all' in a:
            print(f"{f[:-5]:38s} {d['E']:9.1e} {d['g']:9.1e} {d['gap']:9.1e} {d['q']:9.1e} {d['occ']:9.1e} "
                  f"{R['scc_iterations']!s:>3s}->{X['scc_iterations']!s:<3s} {R['converged']!s:.1s}->{X['converged']!s:.1s}"
                  f"{'' if ok else '  FAIL'}")
    n = len(names) - nmiss
    print(f"\nworst over converged reference runs: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    print(f"{gate} gates: {n - nfail}/{n} pass" + (f", {nmiss} missing" if nmiss else "") +
          f" -> {'PASS' if nfail == 0 and nmiss == 0 else 'FAIL'}")


if __name__ == '__main__':
    main()
