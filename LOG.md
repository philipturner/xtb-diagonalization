# Log

## 2026-10-20 — Phase 1 (all but the concurrency sweep)

- **Builds:** main (`68fe777`) and v6.7.1 against Accelerate with the swift-xtb / #1319 recipe (CMake, GCC 16,
  `BLA_VENDOR=Apple`, legacy LAPACK symbols), `scripts/build_xtb.sh`. Production turned out to be upstream main plus
  two GFN-FF file-I/O commits (fork `swift-xtb-2026` `8ea5fcd`), not v6.7.1: it already has the cached Cholesky.
- **Reproduction:** main build is bit-identical to `reference/` on all 174 single points (`scripts/compare.py`).
- **v6.7.1:** needs mctc-lib `77f65c6` and `-std=gnu17`; its gradients are wrong on axis-symmetric structures
  (compiler-dependent; fixed upstream in #1208/#1214) and its C API truncates the log (fixed in #1434). Timing use only.
- **Timers:** `xtb_scc_timer` on `diagonalization` (`db30346`) and `diag/timers-v6.7.1` (`bb9a2e5`), enabled by
  `XTB_SCC_TIMERS=<file>`; bit-identical results.
- **Measured (median of 5, 1 thread):** n = 412–448: 0.58–1.10 s per single point, eigensolve 12–14 ms per cycle =
  79–80% of a cycle and 56–69% of wall; n = 776: eigensolve 48% of wall, integral derivatives + gradient ~35%.
  Cached Cholesky (v6.7.1 → main) saves 1–4%. Profiles: `dsytrd` 27% and `dstedc` 20% of wall at n = 448.
- **Warm starts:** chained −35 to −55% SCC cycles, fixed seed −12 to −33%; all reach the cold solution, but the
  s12_C31 differences (≤ 1.2e-7 Eh, occupations 1.1e-3) already touch the Phase 3 gates: gates need a noise floor.
- **Pseudodiag:** disabled since before the public history (6.2 import, 2019); dead code since #192 (2020-05).
- **Concurrency sweep** (no 10-process point, by the user's call): 8 × 1 thread gives 4.95 jobs/s at n = 448 and
  2.21 at n = 655, +15% / +23% over production's 6 × 2. Per-process efficiency 60–67% at 8 processes, lower for the
  larger matrix: consistent with H1, but AMX sharing per cluster isn't ruled out.
- **Next:** the user reviews `baseline.md` before Phase 2. Suggested BRIEF corrections: production = main, gate noise
  floor, 2x target needs −72–90% eigensolve at n ≈ 410–450.

## 2026-10-20 (later) — BRIEF corrected, gates reset

- **BRIEF.md corrected** from Phase 1: production = upstream main (fork `8ea5fcd`), v6.7.1 for timing only; current
  profile shares next to the old ones; what 2x takes; hypothesis status; Phase 1 marked done; Phase 2 step 1 (cached
  Cholesky) already in production; pseudodiag answer; pushing is the user's step (permissions now deny `git push`,
  including `git -C … push`: both tested and blocked).
- **Gates reset to the measured noise floor.** Warm-vs-cold maxima over 120 runs: 1.2e-7 Eh, 5.4e-5 Eh/bohr, 9.3e-6 Eh
  gap, 2.9e-4 e, 1.1e-3 occupations. New gates, all phases: 5e-7 Eh, 6e-5 Eh/bohr (4.9 pN; first set at 2e-4 = 16 pN, lowered at the user's request), 5e-5 Eh, 1e-3 e, 5e-3; Phase 2
  ≤ 2 extra cycles per run and total within ±1%, Phases 3–4 ≤ 10% (min 2) extra per run. `compare.py --gates
  phase2|approx|identical`; main passes all three.
- **Next:** Phase 2 step 0, the noise floor on the full 174-run grid (warm start from the neighbouring temperature),
  then the GEMM-based transform and the hook point. CLAUDE.md still says pushing `diagonalization` is allowed: the
  user decides whether to change it.
- **Gradient gate lowered** to 6e-5 Eh/bohr (4.9 pN; noise max 5.4e-5 = 4.4 pN), with per-run limits (1.25x each run's
  own noise) once Phase 2 step 0 measures them, plus a second check at accuracy 0.01, where the noise is 150x
  smaller (s12_C31: gradient 2.8e-7 Eh/bohr, energy 1e-11 Eh): method error there ≤ 1e-6 Eh/bohr (0.08 pN).
  `warmstart.py --accuracy`. CLAUDE.md push rule changed: Claude commits, the user pushes.

## 2026-10-20 (later) — Phase 2 step 0: noise floor, two-gate protocol adopted

- **Noise floor, whole grid** (`scripts/noise.py`, 174 runs × 2–3 perturbed starts, accuracy 1.0 and 0.01): at 1.0,
  gradient 7.7e-5 Eh/bohr (6.3 pN), charges 1.2e-3 e, energy 3.1e-7 Eh; at 0.01, gradient 7.2e-7 (0.06 pN), energy
  2.7e-11 Eh. The 1.0 noise exceeded the 6e-5 / 1e-3 gates.
- **Adopted (user's decision): two gate sets.** Method error at accuracy 0.01 against `build/reference-tight/`
  (1e-10 Eh, 2e-6 Eh/bohr = 0.16 pN, 1e-6 Eh gap, 1e-5 e, 5e-5); same solution and convergence at 1.0 against
  `reference/` (5e-7 Eh, 1e-4 Eh/bohr, 5e-5 Eh, 2e-3 e, 5e-3, plus cycle rules). `compare.py --gates
  phase2|approx|tight|identical`; `make_reference.py --accuracy`.
- **Found:** `facet_closed_rms15` u2 300 K has two near-degenerate SCF solutions (gated on same solution only);
  `facet_bare_dimer` u0 300 K and `s12_C00` u3 100 K fail to converge from some warm starts (Phase 3 risk).
- **Next:** Phase 2 step 2 (GEMM-based transform) and step 3 (hook point).

## 2026-10-20 night — Phase 2 steps 2–4 (autonomous; the user is away until 08:00)

- **Kernel study** (`scripts/kernels/eigbench.f90`, dumped GFN2 matrices from `diag/dump`): replacing `dsygst` + `dtrsm`
  by `dtrmm` saves 3–5% of the eigensolve (n = 412: 12.38 → 11.74 ms); `dgemm` doesn't help; `dsyevd` is 88% of it;
  Accelerate's `dsyevr` is 1.6–3x slower (all or lowest eigenpairs). My hook commit: −0–4% per single point, kept on
  `diag/hook-trmm` (`59110aa`), superseded by #1466.
- **PRs #1464/#1465/#1466** (g8j39, open upstream): each passes both gate sets; −0–11%, −10–26%, −6–15% alone.
  Merged into `diagonalization` (`e57877d`; conflicts only with the timer hooks). Combined: **1.21–1.70x faster** than
  main (n = 448: 1.062 → 0.878 s; n = 776: 2.414 → 1.421 s); eigensolve now 67–76% of wall. Both gate sets pass.
- **Phase 3:** offline study on per-cycle dumps (`scripts/kernels/refine.f90`): H2 only in the tail; pseudodiag
  first-order only; Ogita–Aishima ×2 with clusters exact to 1e-9–1e-12 in the last 10–25% of cycles, diverges earlier;
  FP32 robust (density error 1e-6–1e-4). FP32 early cycles implemented, opt-in (`6c6fd43`): −14–17% per single
  point; fails 100 K gates on 5 runs (cycle count, one convergence at accuracy 0.01), passes all at 300/1000 K.
- **Phase 2 drivers / Phase 4:** `dstemr`, `dstebz`+`dstein` 3–9x slower than `dstedc` (`tridiag.f90`); Accelerate
  `dsytrd_2stage` no faster at n ≤ 448, 17–20% at 619–776, no eigenvector support (`trd2stage.c`).
- **Summary for the user:** `FINDINGS.md`.
- **Note:** `scripts/kernels/twostage.c` came from a background agent this session did not start; not committed.
- **Final timing** (round-robin ×3, `build/final/`): `diagonalization` 1.30–1.63x vs the production library, + FP32
  1.55–1.92x. Throughput at 8 × 1 thread vs production's 6 × 2 with the current library: 1.61x (n = 448), 1.75x
  (n = 655). Drop-in through production's `xtb_lib.py`: agrees to 1e-13 Eh.
- **Refinement in xTB** (`diag/refine`, `f349b0e`): passes every gate with unchanged cycle counts, but only 1–4%
  faster at n ≤ 448 and 3–7% slower at n ≥ 619; not merged.
- **Broyden damping** (`diag/broyden`, `5684b47`): 0.25 cuts cycles 13.6% over the grid but changes one SCF solution
  (`facet_bare_dimer` u0 100 K, +3.0 mEh) and slows others; not usable as a default.
- **mmompop by atom pairs** (`6c4b68f`): population stage −40% per cycle, single points −3.5–5.5%; gates pass.
- **Vectorized tridiagonal reduction** (`e891e78`, `src/mctc/lapack/sytrd.c`): LAPACK's dsytrd ported with a NEON
  symmetric matrix-vector product (~50 vs ~30 GB/s); dsytrd −30%, single points −9–13%; gates pass, cycles unchanged.
  An 8-column kernel variant gains little at n ≈ 450 (not used).
