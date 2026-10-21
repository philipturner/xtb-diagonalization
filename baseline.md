# Phase 1 baseline

Measured 2026-10-20 on the M1 Max (32 GB, macOS 15.7.3, AC power). Low Power Mode could not be checked from inside
the sandbox (`pmset -g` printed no power-mode line). Every timed run used `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`,
one process at a time, one untimed warm-up single point, then the median of 5 cold single points (fresh results
object each, production settings: accuracy 1.0, 300 K, maxiter 250). No GFN2 agents were running. Background load
during the timed runs was about half a core (this Claude session plus one Chrome renderer); the load average read 2.7–3.1
including the benchmark process itself. Spread within each set of 5 was ≤ 4%.

## Builds

| Build | Source | How |
| --- | --- | --- |
| production | `../claude-molecular-renderer/libxtb.dylib` | fork branch `swift-xtb-2026` at `8ea5fcd` (see below) |
| main | `xtb/` at `68fe777` (= grimme-lab/xtb `main`, 2026-10-08) | `scripts/build_xtb.sh` → `build/xtb-build-main/` |
| main + timers | `diagonalization` at `db30346` | same flags → `build/xtb-build-main-timers/` |
| v6.7.1 | tag `v6.7.1` (`26b2801`, 2024-07-23), worktree `build/xtb-v6.7.1/` | same flags, plus `-std=gnu17` and an mctc-lib pin (below) → `build/xtb-build-v6.7.1/` |
| v6.7.1 + timers | side branch `diag/timers-v6.7.1` at `bb9a2e5` | same → `build/xtb-build-v6.7.1-timers/` |

All builds use the recipe production is built with (swift-xtb `compile-macos.sh`, grimme-lab/xtb#1319): CMake,
Release, Homebrew `gcc-16` / `gfortran-16` (GCC 16.2.0), `-DBLA_VENDOR=Apple`, `-DWITH_TBLITE=OFF -DWITH_CPCMX=OFF
-DWITH_TESTS=OFF -DMCTCLIB_FIND_METHOD=fetch`. Link checks, all three ways:

- `otool -L`: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`, no OpenBLAS, same as production.
- Symbols: the legacy Fortran LAPACK entry points (`_dsyevd_`, …; 90 in main and production, 89 in v6.7.1), no
  `$NEWLAPACK` symbols. grimme-lab/xtb#1371 (open) would switch to the `$NEWLAPACK` interface; that is a separate
  experiment, not the baseline.
- At run time, `DYLD_PRINT_LIBRARIES=1` shows vecLib's `libBLAS`/`libLAPACK` loaded and no OpenBLAS; wall times match
  the Accelerate column of `reference/LATENCY.md` (n ≥ 400 within 2%), not the OpenBLAS one.

### Production is upstream main, not v6.7.1

The mechanosynthesis notes (`Sources/Workspace/NOTES.md`, 2026-10-19) record that production's "6.7.1" was built from
fork commit `8ea5fcd`. That commit is upstream `main` `68fe777` plus two commits that only stop GFN-FF from reading or
writing files in the working directory; GFN2 code is identical to `main` (its `meson.build` still says 6.7.1). So
production already has the cached Cholesky factor of S (#1248, 2025-05), which v6.7.1 lacks, and the BRIEF's
"production = v6.7.1, full `dsygvd` every cycle" is out of date. Production's binary also contains no tblite, CPCM-X,
dftd4 or multicharge library code (only xTB's own stub modules).

### Reproduction of `reference/`

`scripts/make_reference.py --lib build/xtb-build-main/libxtb.dylib --out build/ref-main`, then
`scripts/compare.py build/ref-main`: **174/174 single points bit-identical** to production: energy, gradient,
charges, gap and occupations differ by exactly 0, and every SCC cycle count and converged flag matches. The timer build
(timers on) is bit-identical too. Since main reproduces production exactly, `reference/` is the baseline's energies,
gradients and cycle counts; they are not repeated here.

### v6.7.1: usable for timing only

- **mctc-lib:** v6.7.1's CMake fetches mctc-lib `HEAD` (unpinned), which no longer configures. It is built against
  mctc-lib `77f65c6` (v0.3.1+7), the `HEAD` on the v6.7.1 release date (`build/deps/mctc-lib-v6.7.1`, passed with
  `FETCHCONTENT_SOURCE_DIR_MCTC-LIB`).
- **C23:** GCC 16 compiles C as C23, where `symmetry_i.c`'s `int true(void)` is an error; v6.7.1 is built with
  `-std=gnu17` (symmetry detection only).
- **Gradients are wrong** on these structures, and the error depends on the compiler: on `t2_R`, gradient norm 0.544
  (-O3) or 917 (-O0) Eh/bohr against 0.0486 for production and for Homebrew's v6.7.1; with `-fcheck=all` it is
  correct. Energies and SCC are unaffected (|ΔE| ≤ 8e-8 Eh). The worst atoms sit on symmetry axes (e.g. the C≡C
  carbons of `t2_R`): v6.7.1 builds H from H0/S per orbital pair and drops or divides overlaps that are zero by
  symmetry, which makes the gradient depend on how S rounds near zero. Upstream fixed this in #1208 (2025-03, "Fix
  gradient evaluation when one of coordinates is close to another") and #1214 (H0 without division by S), both in
  main. Zero-initializing locals (`-finit-local-zero`) does not change it.
- **Its C API loses output:** `xtb_releaseOutput` in v6.7.1 never closes the file (inverted test, fixed in #1434,
  2026-09), so the end of the log is lost and `make_reference.py` misreads SCC counts and converged flags for long
  runs (for example "41, not converged" where the CLI shows convergence after 52). Cycle counts for v6.7.1 below come
  from the timer build instead, and match main's for every timed case.

So the v6.7.1 build is good for one thing: timing the per-cycle `dsygvd` against main's cached factor.

## Wall time per single point

Median of 5, seconds. n = basis functions; cycles from the timer builds (identical in main and v6.7.1).

| Case | uhf | n | Cycles | production | main | main + timers | v6.7.1 | v6.7.1 / main |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `t2_R` | 1 | 124 | 22 | 0.045 | 0.045 | 0.046 | 0.046 | 1.04 |
| `facet_closed` | 0 | 412 | 26 | 0.584 | 0.576 | 0.565 | 0.584 | 1.01 |
| `s12_C31` | 1 | 419 | 76 | 1.560 | 1.534 | 1.536 | 1.565 | 1.02 |
| `facet_dbpair_tool` | 1 | 448 | 49 | 1.100 | 1.082 | 1.066 | 1.109 | 1.03 |
| `s3_S42_159` | 1 | 619 | 42 | 2.202 | 2.154 | 2.179 | 2.245 | 1.04 |
| `diamond233` | 0 | 776 | 19 | 2.308 | 2.283 | 2.293 | 2.719 | 1.19 |

- Main and production agree within 1–2% (main slightly faster in every case, probably run order: production ran
  first). The timers cost nothing measurable.
- **Cached Cholesky (v6.7.1 → main):** 1–4% at n ≤ 619, i.e. 0.3–0.7 ms per cycle (2–5% of the eigensolve; table
  below). At n = 776 v6.7.1 is 19% slower, but almost all of that is outside the SCC loop (1.29 s against 0.90 s; the
  #1208/#1214 H0 changes), not the factorization.

## Per-cycle breakdown (main + timers)

Milliseconds per SCC cycle, median over the 5 timed runs. pot = ES / multipole / D4 potential; H = charge-dependent
Hamiltonian; diag = `fact_solve` (`dsygst` + `dsyevd` + `dtrsm`); pop = Mulliken populations, electrostatics,
multipoles (`mpopsh`, `electro`, `mmompop`, `aniso_electro`); mix = Broyden. potrf = the one Cholesky factorization of
S per single point. Outside = wall time outside the SCC loop (integrals, their derivatives, gradient).

| Case | n | Cycles | pot | H | diag | Fermi | dmat | pop | mix | potrf (ms) | SCC loop (ms) | Outside (ms) | diag share of wall | loop share of wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `t2_R` | 124 | 22 | 0.07 | 0.06 | 0.96 | 0.002 | 0.02 | 0.15 | 0.01 | 0.03 | 28 | 18 | 46% | 61% |
| `facet_closed` | 412 | 26 | 0.37 | 0.72 | 12.12 | 0.005 | 0.50 | 1.53 | 0.06 | 0.38 | 398 | 167 | 56% | 70% |
| `s12_C31` | 419 | 76 | 0.46 | 0.76 | 14.04 | 0.010 | 0.58 | 1.63 | 0.18 | 0.34 | 1347 | 189 | 69% | 88% |
| `facet_dbpair_tool` | 448 | 49 | 0.47 | 0.84 | 13.87 | 0.009 | 0.56 | 1.84 | 0.11 | 0.40 | 868 | 199 | 64% | 81% |
| `s3_S42_159` | 619 | 42 | 1.05 | 1.30 | 34.93 | 0.010 | 2.09 | 3.60 | 0.16 | 0.92 | 1814 | 366 | 67% | 83% |
| `diamond233` | 776 | 19 | 3.27 | 2.38 | 57.68 | 0.015 | 3.55 | 6.26 | 0.18 | 1.61 | 1395 | 898 | 48% | 61% |

v6.7.1 (full `dsygvd` per cycle) for comparison, diag in ms per cycle: 0.99, 12.49, 14.33, 14.59, 35.65, 60.13.

Inside the loop the eigensolve is 79–80% of every cycle at n ≥ 412. Fermi smearing and Broyden mixing are negligible;
populations/multipoles (`pop`) are next at 9–11%.

## Time Profiler (main, 1 thread)

`xctrace record --template 'Time Profiler'` of `scripts/bench.py --one` (warm-up + 5 single points), exported and
aggregated with `scripts/callshare.py`: inclusive share of all samples under `xtb_singlepoint` (1 ms sampling; a
routine counts once per sample). Traces: `build/profile/*.trace`.

| Routine | `facet_dbpair_tool`, n = 448 | `diamond233`, n = 776 | BRIEF table (n = 776, older build) |
| --- | --- | --- | --- |
| SCC loop (`scc`) | 82.4% | 61.3% | — |
| eigensolve (`mctc_dsygvd_factorized`) | 64.5% | 48.2% | 74.6% (`dsygvd`) |
| `dsyevd` | 56.7% | 41.9% | — |
| `dsytrd` (tridiagonalization) | 27.3% | 23.5% | 38.7% (sum of its rows) |
| `dlatrd` → `dsymv` | 19.5% | 19.2% | 29.8% |
| `dstedc` (divide and conquer) | 19.6% | 11.5% | 15.5% |
| `dormtr` (back-transformation) | 8.9% | 6.4% | 9.6% |
| `dsygst` | 5.4% | 4.0% | } 8.8% with `dtrsm`, `dpotrf` |
| `dtrsm` (overlaps `dsygst`) | 8.4% | 8.2% | |
| `dpotrf` | 0.0% | 0.1% | |
| integral derivatives (`build_dsdqh0_noreset`) | 11.7% | 20.5% | — |
| multipole gradient (`get_grad_multiint`) | 6.5% | 14.0% | — |
| `dmat` / `mmompop` / H1 build | 2.8 / 5.5 / 4.1% | 3.1 / 3.0 / 2.0% | ~15% together with `setvsdq` |

At n = 776 the eigensolve is 48% of a single point on current main, not 75%: the older profile also refactored S
every cycle and spent less time outside the loop than main does now. Outside the loop, `build_dsdqh0_noreset` and
`get_grad_multiint` (35% together at n = 776) are OpenMP regions (`gomp_parallel` 28–44% of samples, at one thread).

## What the 2x target requires

Amdahl bound from the table above, if only the eigensolve gets faster (everything else unchanged):

| Case | Wall | Eigensolve | Rest | Eigensolve must drop to reach 2x |
| --- | --- | --- | --- | --- |
| `facet_closed` (412) | 0.565 s | 0.315 s | 0.250 s | ≤ 0.033 s (−90%) |
| `s12_C31` (419) | 1.536 s | 1.067 s | 0.469 s | ≤ 0.299 s (−72%) |
| `facet_dbpair_tool` (448) | 1.066 s | 0.680 s | 0.386 s | ≤ 0.147 s (−78%) |
| `s3_S42_159` (619) | 2.179 s | 1.467 s | 0.712 s | ≤ 0.378 s (−74%) |
| `diamond233` (776) | 2.293 s | 1.096 s | 1.197 s | impossible (rest alone > half) |

A 2x at n ≈ 410–450 needs the eigensolve cut by roughly three quarters on the slowly converging production systems
and by 90% on fast-converging ones like `facet_closed`, where the rest of the cycle (2.6 ms) and the work outside the
loop (167 ms) also start to matter. Cutting cycles (Phase 3) and cheaper full solves (Phases 2/4) both count; the
outside-the-loop integral and gradient work is the next target after the eigensolve.

## Cold vs warm starts (separate from the cold baseline)

`scripts/warmstart.py`, main build, 300 K, production uhf: a 21-point linear scan from each structure to its `_rms15`
copy (0.0075 Å rms per step), every point computed cold, chained (one results handle, `xtb_updateMolecule`) and from
one fixed seed (`xtb_copyResults` of the converged t = 0 result, like swift-xtb's `xTB_Seed`). Total SCC cycles over
the 21 points:

| Scan | Cold | Chained | Fixed seed |
| --- | --- | --- | --- |
| `facet_closed` → `_rms15` | 691 | 451 (−35%) | 605 (−12%) |
| `facet_dbpair_tool` u1 → `_rms15` | 1147 | 622 (−46%) | 978 (−15%) |
| `s12_C31` u1 → `_rms15` | 1696 | 770 (−55%) | 1137 (−33%) |

Chained starts need 18–37 cycles per point where cold needs 26–105; the fixed seed's advantage shrinks as the scan
moves away from the seed (from 18 to 38 cycles on `facet_closed`). Every warm point reaches the cold SCF solution:
largest differences 1.2e-7 Eh in energy, 2e-4 eV in gap, 1.1e-3 in occupations (`s12_C31`, gap 0.15–0.18 eV).

**Consequence for the Phase 3 gates.** Those differences are the SCC's own convergence noise (energy criterion
1e-6 Eh at accuracy 1.0), yet five of the 63 warm `s12_C31` points exceed the proposed approximate gates (|ΔE| ≤
1e-7 Eh, occupations ≤ 1e-3). An approximate eigensolver that only changes the SCC trajectory can fail those gates
on small-gap radicals without being wrong. The gates for Phases 3–4 should be set against this noise floor (for
example, compare against several cold restarts) before they are used to reject a method.

## Pseudodiagonalization history

`git log -S startpdiag` and `-S pseudodiag` reach back only to `29ce1a1` (2019-09-30, "initialized repository from
6.2 release branch"), the first commit of the public history. `startpdiag=1000 !large number=never` is already there,
with no recorded reason. `8ad85e4` (#192, 2020-05-11, "Unify interface to LAPACK and BLAS") commented out the legacy
`call solve(fulldiag, …)` that would call `pseudodiag`, so since then pseudodiag is dead code whatever `startpdiag` is;
`7fc982a` (#1248) later swapped `solver%solve` for `fact_solve`. There is no recorded accuracy or convergence failure
behind the decision: it predates the public history.

## Hypotheses

- **H1 (DSYMV bound by L2 bandwidth):** pending the concurrency sweep (below).
- **H2 (H changes little after the first cycles):** not measured directly yet. Indirect: warm starts (chained) need
  35–55% fewer cycles but still 18–37 per point, so the SCC trajectory is long even from a good start.
- **H3 (AMX not the bottleneck):** consistent so far. `dgemm` is 18–20% of samples, but `dsymv`/`dgemv` inside
  `dsytrd` and the scalar `dlaed4` inside `dstedc` dominate the eigensolve.

## Concurrency sweep

Pending: needs a quiet machine (`mediaanalysisd` was using 70% of a core when it was due).

## Not measured / caveats

- Low Power Mode state (not readable from the sandbox).
- E-core placement is inferred from per-process times, not observed directly.
- Absolute times carry about ±2% from background load; ratios between builds measured back to back are tighter.
