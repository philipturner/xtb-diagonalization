#!/usr/bin/env python3
"""Check a run folder (make_reference.py --out DIR) against reference/ with the BRIEF.md validation gates.

  python3 scripts/compare.py DIR [--gates phase2|approx|identical] [--ref REFDIR] [--all]

Gates (BRIEF.md, Validation), set against the measured SCC noise floor and the same for every phase:
|dE| <= 5e-7 Eh, max|dg| <= 6e-5 Eh/bohr (4.9 pN), gap 5e-5 Eh, charges 1e-3 e, occupations 5e-3.
Per-run gradient limits from the Phase 2 noise-floor measurement are not wired in yet.
phase2    = Phase 2 (reformulation): at most 2 more SCC cycles per run, total cycles within +-1%.
approx    = Phases 3-4: at most 10% (and at least 2) more SCC cycles per run.
identical = bit-for-bit: every difference 0 and every cycle count equal (e.g. a build that only adds timers).
All: every run converged in the reference still converges. Prints the failing runs (all runs with --all), the
worst deviation per quantity, and PASS/FAIL.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOISE_GATES = dict(E=5e-7, g=6e-5, gap=5e-5, q=1e-3, occ=5e-3)
GATES = {'phase2': dict(NOISE_GATES, extra_it=lambda r: 2, total_it=0.01),
         'approx': dict(NOISE_GATES, extra_it=lambda r: max(2, int(0.1 * r)), total_it=None),
         'identical': dict(E=0.0, g=0.0, gap=0.0, q=0.0, occ=0.0, extra_it=lambda r: 0, total_it=0.0)}


def main():
    a = sys.argv[1:]
    run = a[0]
    gate = a[a.index('--gates') + 1] if '--gates' in a else 'phase2'
    ref = a[a.index('--ref') + 1] if '--ref' in a else f"{ROOT}/reference"
    G = GATES[gate]
    names = sorted(f for f in os.listdir(ref) if f.endswith('.json'))
    worst = dict(E=0.0, g=0.0, gap=0.0, q=0.0, occ=0.0)
    nfail = nmiss = 0
    tot_ref = tot_run = 0
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
        ri, xi = R['scc_iterations'], X['scc_iterations']
        it_ok = ri is not None and xi is not None and xi - ri <= G['extra_it'](ri)
        if gate == 'identical':
            it_ok = it_ok and xi == ri
        if R['converged'] and ri is not None and xi is not None:
            tot_ref += ri; tot_run += xi
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
    tot_ok = True
    if G['total_it'] is not None and tot_ref:
        tot_ok = abs(tot_run - tot_ref) <= G['total_it'] * tot_ref
        print(f"\ntotal SCC cycles over converged reference runs: {tot_ref} -> {tot_run} "
              f"({(tot_run - tot_ref) / tot_ref:+.2%}, gate +-{G['total_it']:.0%}){'' if tot_ok else '  FAIL'}")
    print(f"\nworst over converged reference runs: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    print(f"{gate} gates: {n - nfail}/{n} pass" + (f", {nmiss} missing" if nmiss else "") +
          f" -> {'PASS' if nfail == 0 and nmiss == 0 and tot_ok else 'FAIL'}")


if __name__ == '__main__':
    main()
