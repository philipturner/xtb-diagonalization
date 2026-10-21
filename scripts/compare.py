#!/usr/bin/env python3
"""Check a run folder (make_reference.py --out DIR) against a reference folder with the BRIEF.md validation gates.

  python3 scripts/compare.py DIR [--gates phase2|approx|tight|identical] [--ref REFDIR] [--all]

Two gate sets, both set against the SCC noise floor measured over the whole grid (Phase 2 step 0, build/noise/):
Production accuracy (accuracy 1.0, against reference/): catches a different SCF solution or worse convergence.
  |dE| <= 5e-7 Eh, max|dg| <= 1e-4 Eh/bohr (8.2 pN), gap 5e-5 Eh, charges 2e-3 e, occupations 5e-3.
  phase2 = Phase 2 (reformulation): at most 2 more SCC cycles per run, total cycles within +-1%.
  approx = Phases 3-4: at most 10% (and at least 2) more SCC cycles per run.
Tight accuracy (both sides run with --accuracy 0.01, against build/reference-tight/): the method's own error.
  tight = |dE| <= 1e-10 Eh, max|dg| <= 2e-6 Eh/bohr (0.16 pN), gap 1e-6 Eh, charges 1e-5 e, occupations 5e-5;
          SCC cycles reported, not gated.
identical = bit-for-bit: every difference 0 and every cycle count equal (e.g. a build that only adds timers).
All: every run converged in the reference still converges. Runs with two near-degenerate SCF solutions (MULTI) are
gated only on reaching the reference's solution. Prints the failing runs (all runs with --all), the worst deviation per
quantity, and PASS/FAIL.
"""
import json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCTION = dict(E=5e-7, g=1e-4, gap=5e-5, q=2e-3, occ=5e-3)
TIGHT = dict(E=1e-10, g=2e-6, gap=1e-6, q=1e-5, occ=5e-5)
GATES = {'phase2': dict(PRODUCTION, extra_it=lambda r: 2, total_it=0.01),
         'approx': dict(PRODUCTION, extra_it=lambda r: max(2, int(0.1 * r)), total_it=None),
         'tight': dict(TIGHT, extra_it=None, total_it=None),
         'identical': dict(E=0.0, g=0.0, gap=0.0, q=0.0, occ=0.0, extra_it=lambda r: 0, total_it=0.0)}
# Runs where perturbed starts reach a second SCF solution almost degenerate with the reference's (build/noise/):
# facet_closed_rms15 u2 300 K has two, 1.3e-6 Eh apart, whose gradients differ by 2e-3 Eh/bohr.
MULTI = {'facet_closed_rms15_u2_T300.json'}


def same_solution(d):
    """Same criterion as scripts/noise.py."""
    return d['occ'] < 0.05 and d['gap'] < 1e-3 and d['E'] < 1e-5 and d['q'] < 5e-3


def main():
    a = sys.argv[1:]
    run = a[0]
    gate = a[a.index('--gates') + 1] if '--gates' in a else 'phase2'
    default_ref = f"{ROOT}/build/reference-tight" if gate == 'tight' else f"{ROOT}/reference"
    ref = a[a.index('--ref') + 1] if '--ref' in a else default_ref
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
        it_ok = G['extra_it'] is None or (ri is not None and xi is not None and xi - ri <= G['extra_it'](ri))
        if gate == 'identical':
            it_ok = it_ok and xi == ri
        if R['converged'] and ri is not None and xi is not None:
            tot_ref += ri; tot_run += xi
        conv_ok = X['converged'] or not R['converged']
        # a run that did not converge in the reference is compared, but only its convergence status is gated
        if f in MULTI and gate != 'identical':
            values_ok = same_solution(d)
        else:
            values_ok = all(d[k] <= G[k] for k in d)
        ok = conv_ok and (not R['converged'] or (values_ok and it_ok))
        if R['converged'] and f not in MULTI:
            for k in d:
                worst[k] = max(worst[k], d[k])
        nfail += not ok
        if not ok or '--all' in a:
            print(f"{f[:-5]:38s} {d['E']:9.1e} {d['g']:9.1e} {d['gap']:9.1e} {d['q']:9.1e} {d['occ']:9.1e} "
                  f"{R['scc_iterations']!s:>3s}->{X['scc_iterations']!s:<3s} {R['converged']!s:.1s}->{X['converged']!s:.1s}"
                  f"{'' if ok else '  FAIL'}")
    n = len(names) - nmiss
    tot_ok = True
    if tot_ref:
        line = (f"\ntotal SCC cycles over converged reference runs: {tot_ref} -> {tot_run} "
                f"({(tot_run - tot_ref) / tot_ref:+.2%}")
        if G['total_it'] is not None:
            tot_ok = abs(tot_run - tot_ref) <= G['total_it'] * tot_ref
            line += f", gate +-{G['total_it']:.0%}"
        print(line + f"){'' if tot_ok else '  FAIL'}")
    print(f"\nworst over converged reference runs (multi-solution runs excluded): "
          + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    print(f"{gate} gates: {n - nfail}/{n} pass" + (f", {nmiss} missing" if nmiss else "") +
          f" -> {'PASS' if nfail == 0 and nmiss == 0 and tot_ok else 'FAIL'}")


if __name__ == '__main__':
    main()
