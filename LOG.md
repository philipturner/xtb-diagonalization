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
- **Tip timing** (`build/final2/`, round-robin ×3): 1.51–1.89x per single point vs the production library (gated),
  1.66–2.08x with FP32. **Throughput** at 8 × 1 vs production at 6 × 2: 2.07x (n = 448), ~2.0x (n = 655).
- **FP32 vectorized reduction** (`aa5ecd7`, opt-in path only): FP32 mode 1.89–2.19x vs production, but now
  `facet_bare_dimer` u0 300 K diverges (fragile under any rounding change); FP32 needs a stall safeguard before use.
- **FP32 latch** (stay in FP64 once switched) tried and reverted: same gate failures (`facet_bare_dimer` u0 300 K
  diverges through the API, though the CLI run of the same structure converges in 48 cycles with FP32; `s12_C00` u3
  100 K at accuracy 0.01). A real safeguard would restart the SCC in FP64 after a failure; not done unsupervised.
- **Production candidate** `diag/production` (`05f17ac`) = `diagonalization` + `swift-xtb-2026` (GFN-FF file I/O off):
  GFN-FF bit-identical to production and writes no files (`diagonalization` alone writes `gfnff_topo`); GFN2 identical
  to `diagonalization` (174/174). FP32 `ssyevd_2stage` (new interface only) is accurate (rel. 1.5–3.6e-6), refuses
  eigenvectors; new-interface `dstedc` 5% slower than legacy.
- **2 threads** (production's setting): `diag/production` passes the phase2 gates on all 174 runs, cycles unchanged
  (`make_reference.py --threads 2`).
- **Stopped here.** Remaining safe per-single-point gains are each ≤ 5% (H build, potential, work outside the loop);
  `dstedc`'s scalar root finder can't be sped up from outside Accelerate. Next steps are the user's decisions in
  `FINDINGS.md`: FP32 safeguard, SCC accelerator, handover of `diag/production` (needs a Swift check), pushes.

## 2026-10-21 — Eigensolve share at the tip; Swift port of the tridiagonal reduction

- **Eigensolve share** on `aa5ecd7` (timers, 1 thread, median of 5; `build/share-tip.json`): 68% at n = 412, 74% at
  419, 73% at 448, 77% at 619, 69% at 776 (61% at 124); production had 56–69%. Outside the SCC loop 6–16%, rest of the
  loop 16–20%. 2x over production per single point needs the eigensolve ~25–32% faster at n = 412–448.
- **Phase 4 clarification:** the two-stage checks used Accelerate's own `$NEWLAPACK` routines only; swift-xtb's Swift
  stages were never ported or timed.
- **Swift port of `sytrd.c`** (root `Package.swift`; `Sources/Diagonalization`, `Sources/SytrdBench`): generic over
  FP64/FP32, NEON multiply-add through a C header (`Sources/CSIMD`; Swift's `SIMD.addingProduct` compiled to scalar
  `fmadd`). Bit-identical, FP64 and FP32, to `sytrd.c` built with `-ffp-contract=off` (GCC 16 and clang) on four
  dumped matrices; GCC and clang with contraction on fuse some scalar multiply-adds but not their vectorized in-order
  reductions, so the libxtb build differs at rounding (eigenvalues 2e-13 Eh). Reduction time vs Accelerate
  (`sytrd-bench`, median of 21, round-robin, n = 412/448/619/776): FP64 C 1.38/1.39/1.42/1.34x, Swift
  1.42/1.42/1.41/1.33x; FP32 C 1.78/1.69/1.80/1.67x, Swift 1.77/1.66/1.84/1.68x (Swift/C 0.97–1.02).
- **Library workflow:** xtb `07b6e25` adds `XTB_SYTRD_LIBRARIES` (default: unchanged build, 174/174 bit-identical);
  `scripts/build_xtb_swift.sh` links the Swift static library into `build/xtb-build-swift/libxtb.dylib` (no Swift
  runtime dependency at present; still one file). Gates: phase2 174/174, tight 174/174 (vs the C tip: |dE| ≤ 6.4e-10,
  max|dg| ≤ 4.0e-7). Single points (`bench.py`, two alternating rounds): Swift/C 0.996–1.010.
- **Next:** prototype the two-stage reduction in `Sources/Diagonalization` against `sytrd-bench`; needs its own
  back-transformation entry point (the current hook only replaces `xtb_dsytrd_upper`).
- **FP32 early cycles × fixed-seed warm starts** (`scripts/fp32seed.py`, tip `aa5ecd7`, `XTB_SCC_FP32_RMSQ=1e-3`;
  five scans structure → `_rms15`, 11 points, seed = converged cold FP64 result at t = 0, modes interleaved, median of
  5; `build/fp32seed.json`). Totals over 55 points: cold FP64 2579 cycles 41.0 s; cold FP32 2587 (2028 in FP32) 33.4 s
  (−18.5%); seed FP64 1890, 30.4 s (−25.8%); seed FP32 1892 (1365 in FP32) 25.4 s (**−38.1%**; the product of the two
  alone would be −39.5%). FP32 still saves 16.6% when seeded: seeded runs spend 72% of their cycles above the switch
  (cold: 78%), so the levers are nearly independent. Per scan, both: −26% (`facet_closed`) to −48% (`s12_C31`). Every
  mode passes the production-accuracy gate against cold FP64 at all 55 points (max|dg| 4.8e-5 Eh/bohr). Moderate
  background load (Chrome); ratios are interleaved.
- **Grid-wide seeded starts** (`noise.py`, tip library, FP64 → `build/noise-tip64`, FP32 → `build/noise-tip32`; 336
  starts from neighbouring temperatures and displaced copies): the failing seeded starts are the same with and
  without FP32 (3 not converged: `facet_bare_dimer` u0 300 K from 1000 K, `facet_closed_rms15` u2 100 K ×2; 2 on the
  known second solution of `facet_closed_rms15` u2 300 K). Cold FP32 loses `facet_bare_dimer` u0 300 K (known) and
  converges `s12_C31` u3 100 K (unconverged in production too); seeded from 100 K, FP32 reaches the cold FP64 solution
  of `facet_bare_dimer` 300 K in 14 cycles (dE 1.2e-8 Eh, max|dg| 8.9e-6). FP64 tip noise matches the main build's.
- **Decision (user):** FP64 fallback for FP32 and a caller-side switch to enable FP32 are deferred; next is Phase 4
  (two-stage diagonalizer, to be specified by the user).
- **Unsymmetrized 1-stage baseline, verified** (Phase 4 step 1): Swift `tridiagonalize(symmetric: false)` and a
  separate C implementation (`scripts/kernels/unsym.c`) agree within 5% once buffers are page-aligned. Unsymmetrized /
  symmetric time at n = 412/448/619/776: FP64 1.38/1.39/1.50/1.64, FP32 1.05/0.88/1.07/0.94. Components (C, n = 448,
  median of 201): FP32 full `sgemv` 5.0 µs (~160 GB/s) vs NEON `symv` 8.3 µs vs `ssymv` 19.5 µs; FP64 `dgemv` 29.4 µs
  vs NEON `symv` 16.3 µs. The first Swift numbers (FP64 1.76× at n = 448) were an alignment artifact: Accelerate's
  full-matrix kernels are slow on buffers 32 bytes off a 64-byte boundary, where Swift arrays put their elements;
  `sytrd-bench` now uses page-aligned buffers (Accelerate `dsytrd` also 5% faster there: 6.26 vs 6.60 ms).
- **gemv bandwidth sweep** (`scripts/kernels/gemv_bw.c`, n = 64–3000, page-aligned): the FP32 vs FP64 gap is
  Accelerate's `dgemv 'N'` only (49–66 GB/s at n = 200–776); `dgemv 'T'` reaches 100–155 GB/s like `sgemv`. Strides that
  are multiples of 64 bytes are faster for every kernel. Beyond L2 (n ≥ 2048 FP64) everything falls to 15–55 GB/s.
  With `'T'` the unsymmetrized C reduction is 0.83–1.14× the symmetric one in FP64 (n = 448: 3.78 vs 4.54 ms) and
  0.75–1.18× in FP32. Swift baseline left on `'N'` for now. User's deferral list (b_out = 1 specialization, trailing-
  block updates, symmetry, joint block sizes) recorded in BRIEF Phase 4 step 6: only after first benchmarks.
- **Swift baseline switched to `'T'`.** Unsymmetrized / symmetric (Swift, lda = n) FP64 1.20/0.87/1.15/0.99, FP32
  1.15/0.86/1.27/1.04 at n = 412/448/619/776. With the leading dimension padded to 64 bytes (C, `UNSYM_PAD=1`,
  blocked, median of 21): FP64 0.87/0.85/0.81/0.98, FP32 0.82/0.79/0.95/0.93, faster than the symmetric reduction at
  every size (n = 619 FP64: 8.70 vs 10.71 ms). Candidate for xTB (~4–7% per single point, estimated, before the cost
  of mirroring the upper triangle the congruence writes); not implemented. BRIEF Phase 4 step 6 reordered: block-size
  range first, each optimization behind a runtime Boolean, checked over that range.
- **Full-matrix reduction in xTB** (`26984c0`, default; `XTB_SCC_TRD=sym|copy|full`): the congruence's upper triangle
  is mirrored into a copy with lda padded to 64 bytes (`xtb_dsymmetrize_pad`), reduced by `xtb_dsytrd_full` (gemv 'T',
  gemm, ger), and `dormtr` reads the copy. Gates: `sym` bit-identical to the tip (174/174); `full` phase2 174/174 and
  tight 174/174 (vs `sym`: |dE| ≤ 2.4e-9 Eh, max|dg| ≤ 5.5e-7 Eh/bohr), cycles unchanged. Single points vs `sym`
  (`build/trd-wall`, 1 thread, 3 rounds): −1.5% (n = 124), −7.3% (412), −2.2% (419), −8.3% (448), −11.8% (619), −2.0%
  (776); 2 threads (1 round) −1.1 to −10.3%. Copy alone (`scripts/kernels/symcopy.c`, median of 501): 4/46/43/52/95/150 µs
  at n = 124/412/419/448/619/776, ~45–49 GB/s, 0.26–0.41% of an SCC cycle; whole-cycle timing (`copy` − `sym`) can't
  resolve it (±1 ms noise). The Swift library exports both new functions (`build/xtb-build-swift` links; same bits as
  the C library on `facet_dbpair_tool`, 6 runs). Not yet merged into `diag/production`.
- **8 × 1 throughput, `XTB_SCC_TRD`** (`build/trd-8x1`, sweep.py, 2–4 alternating rounds): `full` loses. Queue rate
  (sum of per-worker rates) vs `sym` at n = 412/448/619/776: −6.7/−6.1/−1.9/−17.4%; by the slowest worker −11/−12/−8/
  −19%. Its median single point is still 5–8% faster at n = 412–619 but 15% slower at 776, and the spread between
  workers grows (21–38% vs 2–23%). `copy` matches `sym` (n = 776 4.42 vs 4.43 jobs/s; n = 448 8.45 vs 8.72, within
  round-to-round noise), so the extra matrix isn't the problem: the full-matrix Accelerate kernels contend, probably
  on the AMX (one per cluster), where our NEON `symv` doesn't. Default switched back to `sym` (`xtb` commit after
  `26984c0`; 174/174 bit-identical to the earlier tip). Lesson for Phase 4: AMX-heavy kernels need the 8 × 1 check.
- **Phase 4 steps 2–3, first 2-stage eigensolve** (Swift, FP64, `Sources/Diagonalization/TwoStage.swift`,
  `Sources/EigenBench`): stage 1 dense → band by panel `geqrf` + `larft` + two `larfb` over the whole matrix; stage 2
  bulge chasing (Schwarz, one reflector per bulge, plain loops, column order); `dstedc`; back-transformation of the
  bulge reflectors blocked b sweeps at a time into (2b − 1) × b WY blocks (`larfb`; order checked: swapped pairs act on
  disjoint rows), then the band panels (`larfb`). Development version per BRIEF: full matrix, no symmetry, whole-matrix
  stage 1 updates. Accurate at every b = 4–64 on all four sizes: eigenvalues ≤ 2.8e-13 Eh from `dsyevd`, residual
  ≤ 1e-15, orthogonality ≤ 5e-15. Time relative to the unsymmetrized 1-stage eigensolve (tridiagonal + `dstedc` +
  `dormtr`), all eigenvectors, n = 412/448/619/776 (`build/eigen-bench-v1.0.txt`, median of 5): best b = 32,
  2.11/2.36/2.12/2.05×; b = 16 2.46–3.09×, b = 64 2.48–3.15×. Half the eigenvectors (`-v0.5`): best b = 24–32,
  1.87–2.13×. Stages at n = 776, b = 32, all vectors (ms): band 13.8 (1-stage tridiagonal: 19.8, despite ~3× redundant
  work), bulge 24.4 (grows ∝ b), `dstedc` 13.3, bulge back-transformation 25.0 (`dormtr`: 7.4; twice the flops from the
  half-empty 2b × b blocks, at ~75 vs ~125 GFLOPS), band back-transformation 6.2. The blocked bulge back-transformation
  cut 18.3 → 5.0 ms at n = 412, b = 32 (unblocked reference kept, `--unblocked-bulges`); the column-order bulge update
  11.5 → 7.1 ms. Rough relevant range so far: b = 16–48. Not yet checked: 8 × 1 throughput.
- **3-stage SBR, correctness first** (`Sources/Diagonalization/SuccessiveBandReduction.swift`): `reduceBandwidth`
  reduces any band b_in → b_out (panel p ≤ b_out): per step a QR of a (p + b_in − b_out) × p block (`geqrf`,
  `larft`) applied to both sides (`larfb`) on a window around it, chasing the bulge's first p columns down by b_in per
  step (Bischof–Lang–Sun); dense → band is b_in = n − 1, the classical chase b_out = p = 1. `successiveBandReduction`
  runs any chain n − 1 → b1 → … → 1; `backTransform` replays each stage's WY blocks in reverse. `eigen-bench --sbr
  64-16,32 [--whole] [--fast-tail]`, `random:N` matrices. Checks: 74 variant runs on random matrices n = 3–131 (chains
  up to 4 stages, bandwidths up to n − 2): eigenvalues ≤ 5.3e-14, residual ≤ 1.1e-15, orthogonality ≤ 4.8e-15; four
  GFN2 matrices, chains [32], [64,16], [96,32], [128,32], [64,8], [128,16], 4-stage [96,32,8], [64,8,2]: eigenvalues
  ≤ 3.0e-13 Eh; every stage leaves exactly zero outside its band (no out-of-band entry escapes the explicit
  annihilations); windowed and whole-matrix updates agree. Time (`build/sbr-fasttail.txt`, last stage by TwoStage's
  chase + blocked back-transformation, n = 776, all vectors, ms): 2-stage [32] 86.4 (TwoStage 83.6); 3-stage [128,32]
  111.8: middle stage 22.2 + its back-transformation 9.7 cost more than they save; [64,16] 143.5: the tail chase drops
  24.4 → 13.1 but its blocked back-transformation rises 25.0 → 56.7 (small blocks). Unoptimized; all-general (no fast
  tail) is 6–10× the 1-stage time, dominated by the per-bulge b → 1 steps.
- **FP32 and throughput of the Phase 4 code** (`eigen-bench --fp32`, `--procs P --jobs K`; until now the Float
  specializations were compiled but never run, and no Phase 4 kernel had a multi-process measurement).
  Accuracy, FP32 (checked in FP64 against `dsyevd` of the FP64 matrix): random n = 3–131, all 74 chain runs: residual
  ≤ 5.4e-7, orthogonality ≤ 2.5e-6, zero out-of-band leftovers, eigenvalues ≤ 7.7e-6 (up to 5.6× Accelerate `ssyevd` on
  the same FP32 matrix at n = 17, where our 1-stage is already 1.9–2.3×: more reduction steps, more roundings). GFN2
  matrices, 36 runs: eigenvalues 8.5e-6–4.9e-5 Eh, 0.1–0.3× `ssyevd`'s (7.4e-5–2.0e-4 Eh; our 1-stage FP32 is as good
  as the band variants, so it's `ssyevd` that is less accurate here), residual ≤ 1.0e-6, orthogonality ≤ 2.5e-6.
  Latency (`build/eigen-fp32.txt` vs `-fp64.txt`, median of 5, all vectors): 1-stage gains 1.6× from FP32, 2-stage
  only 1.2–1.5×, so 2-stage at its best b is 2.17–2.82× the unsymmetrized 1-stage in FP32 (FP64: 2.05–2.36×) and the
  best b moves up to 48 (n = 776 FP32: 56.8 ms vs b = 32 67.8). The blocked bulge back-transformation is no faster in
  FP32 (n = 776, b = 16: 62.4 vs 56.5 ms FP64; b = 32: 28.3 vs 25.0).
  Throughput (`build/eigen-throughput.txt`, eigensolve alone, 1 vs 8 processes × 1 thread, 100/40 jobs at n =
  448/776; short runs, 1–6 s): scaling to 8 processes, FP64 n = 448/776: 1-stage sym 6.4/5.6×, unsym 5.7/3.7×, 2-stage
  b = 32 5.9/5.1×, b = 48 6.1/5.4×; FP32: sym 6.8/6.2×, unsym 6.5/5.2×, 2-stage 6.6–6.7/5.8–6.0×. The current 2-stage
  code scales like the symmetric 1-stage and better than the unsymmetrized one (whose AMX-bound gemv/gemm contend,
  as in xTB with `XTB_SCC_TRD=full`); it isn't (yet) losing extra throughput at 8 × 1. 8-process jobs/s, 2-stage best
  / 1-stage sym: FP64 0.43 (448), 0.47 (776); FP32 0.39, 0.42.
- **Large n** (user's new `sic_n588/1204/1788`; dumps `build/dumps/sic_*/hs_10.bin` from `diag/dump`, uhf 0, 300 K,
  cycle 10). Accuracy: FP64 eigenvalues ≤ 8.2e-13 Eh, residual ≤ 1.9e-15, orthogonality ≤ 6.0e-15; FP32 eigenvalues
  ≤ 2.4e-5 Eh (`ssyevd`: 4.3e-5 / 2.6e-4 / 5.4e-4), residual ≤ 9.5e-7; zero out-of-band leftovers in every SBR stage.
  Latency, 1 process (`build/eigen-large-fp64.txt`, `-fp32.txt`, median of 3, all vectors), best variant / 1-stage
  sym: FP64 n = 588: 2-stage b = 32 48.4 vs 22.2 ms (2.18×); 1204: b = 32 307.9 vs 134.9 (2.28×); 1788: b = 48 912 vs
  446 (2.04×). FP32: 588 b = 48 31.6 vs 13.2; 1204 b = 48 163.2 vs 74.0 (2.21×); 1788 b = 64 462 vs 210 (2.20×). The
  best b grows with n (32 → 48–64). The unsymmetrized 1-stage breaks down once the matrix leaves cache (n = 1788 FP64:
  tridiagonal 449 vs 250 ms symmetric); the whole-matrix band reduction (b = 64: 189 ms) already beats the symmetric
  1-stage reduction (250 ms) at n = 1788 despite its ~3× redundant work. 3-stage still loses everywhere (best: FP32
  n = 1788 [256,64] 557 ms vs 2-stage b = 64 462 ms): its middle stage and back-transformation cost more than the
  smaller tail band saves.
  Throughput, 8 processes × 1 thread (`build/eigen-throughput-large.txt`; 8 / 4 jobs per process at n = 1204 / 1788,
  short runs): scaling 1 → 8 processes, FP64 n = 1204 / 1788: 1-stage sym 3.4× / 2.6×, unsym 1.6× / 1.5×, 2-stage
  b = 64 5.0× / 4.1×; FP32: sym 5.0× / 3.1×, unsym 3.5× / 1.6×, 2-stage b = 64 5.6× / 5.2×. The 1-stage BLAS2
  reduction is memory-bandwidth-bound when 8 matrices of 12–26 MB stream at once; the 2-stage's BLAS3-heavy work isn't.
  8-process jobs/s, 2-stage best / 1-stage sym: FP64 0.63 (1204), 0.78 (1788); FP32 0.47, 0.78 (single process:
  0.43–0.49). With the deferred optimizations (trailing-block stage 1 alone removes ~2/3 of its work) 2-stage may pass
  1-stage in 8 × 1 throughput around n ≈ 1800 first.
- **Structural optimizations, measured one at a time** (`SBROptions`, `eigen-bench --opts`, `scripts/opt_table.py`,
  `build/opt-{small,large}-fp{64,32}.txt`; FP64 n = 776 b = 32 / n = 1788 b = 64, ms): stage 1 13.4 / 200 → trailing
  block only 10.0 / 135 → combined sy2sb update 8.2 / 103 → lower triangle (symm, syr2k) 8.4 / 82 (helps at large n
  only); bulge chase 24.4 / 322 → lower-triangle symmetric chase 15.2 / 151, plus vector-accumulated dot products
  (Swift doesn't vectorize an ordered FP reduction) 13.8 at n = 776; skipping the end-of-stage-1 mirror and zero passes
  when the chase reads only the lower triangle. Not helpful: the parallelogram WY kernel (5 triangular products instead
  of larfb's 7) is slower with Accelerate's trmm (FP32 n = 776: 25.5 → 44.5 ms); precomputing all bulge WY blocks
  (`prep`) costs more than rebuilding each just before use (the table is as large as the matrix); column blocks of Z
  (k128) are slower at every size (the blocks are re-read per column block). Best set `tcs`/`tcls`: 2-stage / 1-stage
  sym, FP64: 1.69× (n = 412, was 2.16×), 1.67× (776, was 1.97×), 1.69× (1204, was 2.44×), 1.51× (1788, was 2.17×);
  FP32 n = 1788: 340 vs 199 ms (1.71×, was 2.27×). Accuracy unchanged (FP64 ≤ 8.2e-13 Eh, FP32 ≤ 4.9e-5 Eh).
- **Code reorganized, one file per step** (user request): `SBROptions` (was TwoStageOptions), `BandForm`, `BandToBand`,
  `BulgeChasing`, `DivideAndConquer`, `BulgeBackTransformation`, `BandToBandBackTransformation`,
  `BandFormBackTransformation`, `WYTransform` (every compact-WY build and apply, with a map of where each step uses
  it), `SuccessiveBandReduction` (driver only: `SBRWorkspace`, `sbrReduce`, `sbrBackTransform`; first stage by
  BandForm, middle stages by BandToBand, last by BulgeChasing; requires an intermediate bandwidth, 1-stage being
  `Tridiagonalization.swift`). 2-stage is the chain [b] through the same driver at no extra cost (SBR[tcs] 32 = 2s[tcs]
  b = 32, same results); 3-stage now gets the optimized first and last stages. Regression: random n = 3–131, 98 runs,
  eigenvalues ≤ 5.3e-14, zero out-of-band leftovers.
- **GEMM compact-WY apply (`SBROptions.gemmWY`, `--opts …g`) and bulge group size (`bulgeGroup`, `vN`)**
  (2026-10-21). Microbenchmark (`scripts/kernels/wy.c`, one bulge block: 2b − 1 rows, b reflectors, m columns, 1
  thread): Accelerate's `larfb` reaches only 26–80 GF/s FP64 / 42–170 GF/s FP32; the same work as gemm (VᵀZ) + trmm or
  gemm (T) + gemm (Z −= VW) runs 64–218 / 89–600 GF/s, 2.2–4× faster (FP64 m = 588: b = 16/32/64 45/87/260 µs vs
  17/32/88 µs). `larft` is minor (1.5–28 µs). T times W is faster as a dense gemm than as trmm (FP32 b = 32: 2.6 vs
  10.6 µs). New `wyApply(gemm:)`: T copied dense, three gemm calls, both sides and both transposes; V is now stored
  explicitly (unit diagonal, zeros above) where needed (BandForm with the option; BandToBand's stored steps). Used in
  every back-transformation and in the reductions that use larfb (BandForm without `c`, BandToBand). Accuracy: random
  n = 3–131, all 2s/SBR/general 3–4-stage chains, FP64 eigenvalues ≤ 5.0e-14, FP32 ≤ 6.8e-6, zero out-of-band
  leftovers.
  Latency (`build/gemmwy-sweep.txt`, median of 5, all vectors, best b, ms, `tcs` → `tcsg`): FP64 n = 412 15.54 →
  14.61, 619 40.99 → 37.80, 776 63.73 → 62.34, 1204 211.3 → 191.8 (`tclsg` b = 64); FP32 412 10.48 → 8.90, 619
  26.61 → 22.85, 776 42.01 → 37.79, 1204 117.7 → 102.1. The bulge back-transformation: FP64 n = 1204 b = 48 78.5 →
  64.7, FP32 b = 32 102.5 → 33.3. 1-stage sym: FP64 9.3 / 25.6 / 41.8 / 135; FP32 5.6 / 15.3 / 25.5 / 72.
  Best 2-stage / 1-stage sym now FP64 1.57 (412), 1.48 (619), 1.49 (776), 1.42 (1204); FP32 1.58, 1.49, 1.48, 1.42.
  Throughput 8 × 1 (`build/gemmwy-throughput.txt`, 1 and 8 processes, jobs/s, b = 32/48 best): **FP32 gains at 8 × 1
  too, FP64 doesn't at n ≥ 776.** FP64 `tcs` → `tcsg`: n = 412 376 → 389, 619 123 → 136, 776 81.1 → 75.6, 1204 20.2
  → 18.4, 1788 5.40 → 5.05 (1-stage sym 640 / 222 / 136 / 27.0 / 6.44). FP32: 412 611 → 677, 619 214 → 252, 776 136
  → 152, 1204 41.0 → 46.2, 1788 13.6 → 14.6 (1-stage sym 1159 / 416 / 234 / 74.3 / 15.6). Best 2-stage / 1-stage at
  8 × 1: FP64 0.61 / 0.61 / 0.60 / 0.75 / 0.84 (n = 412 … 1788), FP32 0.58 / 0.61 / 0.65 / 0.62 / 0.93. Single
  process n = 1788: FP64 `tcsg` b = 48 565 ms vs 1-stage 442; FP32 273 vs 202. The FP64 loss at 8 × 1 is consistent
  with AMX contention (4 processes per AMX unit): larfb's copies don't use the AMX and overlap with other processes'
  AMX work; three gemm calls keep it busy.
  Bulge group size (`build/bulgegroup-latency.txt`, `-throughput.txt`): g < b sweeps per block cuts the arithmetic on
  the block's zeros to (b + g − 1)/b but thins the gemm; at 1 × 1 g = 32 with b = 48/64 is neutral to 6% faster in
  the stage, g = 24/16 lose 5–15% overall, g = 8 1.4–1.9× slower; at 8 × 1 FP64 (b = 48) g = 32/24 don't help either
  (n = 1204: 18.5 / 18.2 vs larfb 19.5 jobs/s). Off by default.
  Code: the stage files moved to `Sources/Diagonalization/Stages/` (found moved in the working tree at session start;
  committed with this step). Next: per-precision defaults (gemmWY on for FP32; FP64 on for latency, off for 8 × 1 at
  n ≥ 776), then the bulge chase (now the largest FP64 stage after the back-transformation) and occupied-only vectors.
- **Bulge chase, occupied-only vectors, recommended configuration** (2026-10-21, continued).
  Band storage for the symmetric chase (`compactBand`, `b`): the lower band copied to leading dimension 2b (no two
  elements within 2b of the diagonal alias; the chase's fill reaches 2b − 1), same code, bit-identical results
  (48/48 random runs). No speed-up (`build/compactband-latency.txt`, within ±2% except two outliers): the chase isn't
  memory-bound. It runs at ~6 GF/s (n = 1204, b = 48: ~0.3 GFLOP in 43–50 ms), ~10% of one core's FP64 NEON peak.
  Explicit vector FMA kernels for every chase loop (`vectorChase`, `x`; new `VectorKernels.swift`: dot, axpy, two-term
  axpy, fused dot + axpy for the symmetric block): chase 42.9 → 39.6 ms (n = 1204, b = 48, after `tcs`), 51.7 → 49.2
  (after `tcls`); accuracy unchanged (random n = 3–131, 120 runs, FP64 ≤ 5.3e-14, FP32 ≤ 7.4e-6). The remaining cost
  is probably per-call overhead of ~200 short loops per bulge (b-element columns); register-blocking 4 columns per
  pass is the next try.
  **Unexplained:** the chase right after the `l` (symm/syr2k) first stage is 15–30% slower than after `tcs` (n = 1204
  b = 48 49–51 vs 40–43 ms; n = 1788 b = 64 124–142 vs 108 ms), cancelling `l`'s stage-1 saving (6 / 28 ms). It isn't
  the data (zero/subnormal counts identical; chase on fresh copies of the `tcls` output runs at the fast speed after
  the first call), and not the skipped final pass (keeping it doesn't help). The slow state sometimes persists for
  several chase calls and is only partly cleared by 5–20 ms of compute without memory traffic: machine state after
  AMX-heavy syr2k (cache writebacks or power state), not measurable without counters. Net: `l` is neutral at large n.
  Occupied-only eigenvectors (`build/vectors-latency.txt`, best b, ms; 1-stage sym / SBR `tcsg`): FP64 m = n/2:
  n = 412 8.49 / 12.38, 619 22.75 / 32.21, 776 38.71 / 51.12, 1204 118.0 / 152.0 (1.29–1.46×; all vectors
  1.45–1.58×); m = n/4 1.20–1.40×. FP32 m = n/2 1.36–1.53×, m = n/4 1.27–1.55×. 1-stage gains less (its ormtr is a
  small share; stedc forms all tridiagonal vectors in both). Single-thread SBR's reduction alone (stage 1 + chase,
  n = 1204: ~76 ms) is already about 1-stage's whole tridiagonalization (~71 ms), so a single-thread win needs a much
  faster chase; its case remains 8 × 1 throughput at large n.
  `SBROptions.recommended(n:fp32:manyProcesses:)`: t c s x, gemmWY for FP32 always and for FP64 alone or n < 700,
  b = 32 / 48 / 64 below 700 / 1500 / above. Regression through the driver: random n = 3–131, FP64 ≤ 4.6e-14, FP32
  ≤ 6.5e-6.
- **Tridiagonal eigenvectors for the occupied orbitals only** (`scripts/kernels/tridiag_subset.c`,
  `build/tridiag-subset.txt`, median of 3, 1 thread). xTB (`e7b0317`) already back-transforms only the occupied
  columns (focc > 1e-16) during the SCC and the rest after convergence, but calls `xstedc('I')` for all n tridiagonal
  vectors. Alternatives that return all eigenvalues and only vectors 1 … m are much slower: FP64 n = 412 / 776 / 1204
  `dstedc` (all n) 3.6 / 13.3 / 29.6 ms, `dsterf` + MRRR `dstemr` for m = n/4 11.2 / 41.8 / 100.4 ms (n/2: 21 / 74 /
  201); bisection + inverse iteration (`dstebz` + `dstein`) 53 ms at n = 412, m = n/4. FP32 likewise (`sstedc` 2.5 /
  8.9 / 19.3 ms, `sstemr` n/4 8.7 / 29.5 / 75.6), and MRRR's FP32 orthogonality is 1e-4 vs 2e-6. Accelerate's divide
  and conquer is fast on these spectra; not pursued.
- **FP64 restart for FP32 SCC runs** (xtb `4d68074`): `scf` keeps the SCC's inputs while `XTB_SCC_FP32_RMSQ` is set
  and, if a run that used FP32 doesn't converge within `XTB_SCC_FP32_MAXITER` cycles (default 160; converged
  production runs need ≤ 130, median 37), reruns it from the same start with every eigensolve in FP64 (a full restart,
  not a hand-off: the latch experiment showed continuing from an FP32-perturbed state doesn't rescue fragile runs).
  `facet_bare_dimer` u0 300 K cold FP32 (didn't converge before): restarts, result bit-identical to FP64 (energy,
  gradient, charges, eigenvalues), 2.05 vs 0.64 s. Grid with the variable unset (`build/ref-restart64`): 174/174
  bit-identical to the `1c9a9c5` grid. Cold FP32 grid (`build/ref-restart32`, rmsq 1e-3): no convergence failures
  beyond production's own two (`s12_C00` u3 and `facet_closed_rms15` u2 at 100 K); approx gates 175/177, both failures
  cycle counts (the restarted run; `s12_C31_rms05` u3 100 K 107 → 123); worst E 1.6e-7 Eh, gradient 5.1e-5 Eh/bohr;
  total cycles +1.8%. Opt-in only; the caller sets the variable around (warm-started) calls. Next: cold perturbation
  study FP32 vs FP64 (rate of different SCF solutions), then rmsq thresholds and an early-abort rule, to decide whether
  FP32 can be the default.
- **Can FP32 be the default? Basin study, error sources, FP64 refinement** (2026-10-21, continued).
  Displaced-copy study (`scripts/basin.py`, `build/basin/`; 20 seeded copies at 0.03 Å rms of `facet_bare_dimer`,
  `s12_C00`, `s12_C31`, `facet_closed_rms15` × 2 spins × 3 temperatures, cold FP64 and FP32 rmsq 1e-3 with the
  restart): 456/480 same solution, **0 different solutions**, 20 unconverged in both, 4 converged only with FP32 (two
  within 2e-5 Eh of where FP64 stalls; two where FP64 diverges, unverified). 8 restarts (all 100 K radicals but one);
  counting each failed 160-cycle attempt, FP32 used 3.2% more cycles than FP64 on the same-solution runs.
  Error sources (`scripts/kernels/fp32_error.c`, 5 dumped matrices, occupied subspace vs FP64): all-FP32 sin(angle)
  1.7–8.6e-6; FP32 in one stage only (reduction, divide and conquer, or back-transformation) 1.0–2.2e-6 each, so no
  stage dominates. One FP64 first-order correction (residual form R = A Z − Z e; the plain projection of A Z made it
  worse, the FP32 vectors' non-orthogonality is as large as the error) + Cholesky QR: 3.8e-12–4.1e-11.
  Cost (`scripts/kernels/refine_cost.c`, n = 776, k = 420): FP64 eigensolve 47.0 ms, FP32 28.8; full refinement 20.9
  (k × k Rayleigh–Ritz dsyevd 11.8) + full-width sormtr 2.6, as much as FP32 saves. Cheaper scheme (xtb `338c72a`,
  `XTB_SCC_FP32_REFINE=1`): first-order step of each z_i against all z_j (j ≠ i), pairs closer than 1e-3 Eh skipped,
  Rayleigh quotients for eigenvalues, Cholesky QR, Rayleigh–Ritz only on the skipped clusters (≤ 2 orbitals): subspace
  ~1e-9, eigenvalues ≤ 5.4e-9 Eh. Cold grid: approx gates 177/177 (plain FP32 175/177), cycles +0.03%; vs the same
  library's FP64 grid 184/192 within tight thresholds, the other 8 open-shell radicals at their SCC noise floor;
  test runs within ~1e-12 Eh of FP64, `facet_bare_dimer` u0 300 K in FP64's 52 cycles.
  Timing (`build/bench-refine3`, `build/sweep-refine`, idle 17:35–17:50, two passes in opposite order), relative to
  FP64: 1 thread FP32 0.80–0.88, refined 0.91–0.98; throughput 1x1 FP32 1.16–1.26x, refined 1.02–1.12x; **8x1 FP32
  1.15–1.25x, refined 0.95–0.98x** (n = 448 / 619 / 776). The refinement's FP64 GEMMs contend for the shared AMX
  units, while FP32's saving is in the reduction. Refinement stays opt-in; plain FP32 + restart is the throughput
  option.
- **Tooling note:** `make_reference.py --help` isn't help: it started the production run and wrote two missing
  sic_n1788 references into `reference/` (stopped; both deleted; nothing overwritten).
- **Decision (user):** plain FP32 + restart will be the default, legacy FP64 selectable; how the default is set and
  checked (results report the mode in effect and any restart, effective settings printed, malformed values rejected,
  a gate run per mode) is to be worked out later.
- **C API for the SCC precision** (xtb `e2e5bee`, user's choice over environment variables, which are process-global
  and unsafe for threads of one Swift process). `xtb_setSCCPrecision(env, calc, fp32_rmsq, fp32_maxiter,
  fp32_refine)` on the calculator (default FP64; invalid values are errors that change nothing; FP32 on GFN-FF is an
  error); `xtb_getSCCPrecision(env, res, ...)` on the results: settings used, FP32 cycles, FP64 rerun, SCC cycles
  (error without an SCC). The library no longer reads `XTB_SCC_FP32_*`; the CLI runs FP64. One diagnostic line at
  minimal verbosity. The restart no longer writes the process-global `set%broydamp`. Driver scripts set the mode
  through the API (`scripts/sccprecision.py`: `--fp32 RMSQ [--fp32-maxiter N] [--fp32-refine]` for make_reference,
  bench, sweep, basin; fp32seed uses the setter) and check every result's report against the request.
  Validation: FP64 via the API 174/174 bit-identical to `1c9a9c5`; FP32 1e-3 via the API 192/192 bit-identical to
  the environment-variable FP32 grid, approx gates vs production 175/177 (the two known cycle-count cases); 384/384
  runs report exactly the requested settings (3 FP64 reruns, all reported). Contract test
  `scripts/test_sccprecision.py`: 16/16, including FP32 and FP64 calculators in two threads of one process, each
  bit-identical to its serial result, and the production library (no API: FP64 runs, FP32 request refused).
  Next: Swift bindings (user's branch): dlsym both symbols; missing → FP64, FP32 request traps; set per calculator,
  check the getter after every single point.
- **Swift bindings** (swift-xtb branch `fp32-diagonalization`, cloned by the user into `swift-xtb/`, commit
  `41ba777`; not tracked here): `calculator.precision` = `.fp64` (default) or `.fp32(switchThreshold: 1e-3,
  maximumIterations: 160, refined: false)` through `xtb_setSCCPrecision`; both API symbols load as optionals
  (`xTB_Library.supportsFP32`; absent → FP64 only and a `.fp32` request traps; only one present traps); values
  validated in Swift; after every singlepoint `xtb_getSCCPrecision` is checked against the calculator (mismatch
  traps) and exposed as `calculator.precisionReport`. Tests: 16/16 with the diagonalization library (1 and 8
  threads); production library 13 pass, 3 FP32 tests skip, `.fp32` traps (checked from a scratch package).
- **Why the chase is slower after `l` (diagnosis; corrects "the chase isn't memory-bound").** Probe (temporary
  eigen-bench code, `build/chase-probe.txt`; chase on a fresh copy of the `tcs` band, n = 1204 b = 48 / 1788 b = 64,
  median of 9): after nothing 39.1 / 108.0 ms, `dgemm` 39.5 / 108.2, stage 1 `tcs` 39.2 / 108.1, 20 ms spin 39.6 /
  108.3; after one `dsyr2k` 49.1 / 127.8, `dsymm` 48.8 / 118.7, stage 1 `tcls` 49.4 / 122.1; 50 ms sleep 43.4 /
  110.9. `scripts/kernels/accel_after.c`: no extra threads and 0.0 ms CPU after the kernels; an AMX `clr` after
  `dsyr2k` raises SIGILL (the kernel already disabled AMX); a pure-L1 compute loop is unaffected (3.46–3.52 ms). The
  chase's memory pattern alone (no real arithmetic, n = 1204) runs at either ~19 or ~32 ms, bimodal: slow after
  `dsymm`, `dsyr2k`, or a plain 64 MB read (no Accelerate), for one to several consecutive runs (~30–150 ms), then
  fast again. So it's a hardware power/cache state after heavy memory streaming that slows L2-bound code up to 1.7x,
  not Accelerate or our code (mechanism unconfirmed without counters). About half of the chase (~19 of ~39 ms at
  n = 1204) is L2 traffic: each of its ~n sweeps streams the whole band window (~1.4 MB) through L2. Band storage
  didn't shrink that footprint, hence no gain. Chase timings depend on what ran just before them. Next lever: keep
  the chase's working set in L1 by passing several consecutive sweeps through the same window (bulge window ~110 KB
  at b = 48 FP64, ~50 KB at b = 32 or FP32; L1 128 KB), confined to `BulgeChasing.swift` behind an option.
- **Hardware counters of the slow state** (Instruments CPU Counters through `xctrace`, user's template
  `build/memory-counters.tracetemplate`: time-sampled every 100 µs; `scripts/kernels/chase_trace.c` marks each run of
  the chase's memory pattern with a signpost, `scripts/chase_counters.py` sums counters per run; n = 1204, b = 48;
  record with `--all-processes` and launch the program separately: `--launch` recorded no samples of it). Per run,
  fast (107 runs, 19.6 ms) vs slow (6 runs, 31.6 ms): INST_ALL 2.15e8 vs 2.14e8 (same work); L1D_CACHE_MISS_LD 3.46e7
  vs 3.93e7; L1D_CACHE_MISS_ST 2.36e7 vs 0.60e7; L1D_CACHE_WRITEBACK 1.31e7 vs 1.30e7; MAP_STALL 2.36e7 vs 5.28e7
  (+29 M cycles, ~9 ms of the 12); L1D_TLB_MISS 3.67e6 vs 2.46e6; L2_TLB_MISS_DATA 1.4e3 both. So the slow state
  costs more per L1 miss (slower L2/fabric response or different prefetching), not more misses; no M1 counter here
  separates L2 from the system-level cache. Supports keeping the chase's working set in L1.
- **L1 blocking of the chase: not worth it** (`scripts/kernels/wavefront.c`, `build/wavefront-1204.txt`; the chase's
  lower-triangle memory pattern, n = 1204, sweeps in groups of g as a wavefront with sweep s + j trailing by 2j bulges,
  the least lag that keeps the order: sweep s + 1's bulge k shares an element with sweep s's bulge k + 2). Speed-up over
  sequential, g = 2 / 3 / 4 / 6: FP64 b = 32 1.12 / 1.08 / 1.05 / 0.97 (10.6 ms sequential), b = 40 1.05 / 1.00 /
  0.96 / 0.92, b = 48 1.00 / 0.94 / 0.91 / 0.90 (14.8 ms); FP32 b = 32 1.05 / 1.03 / 0.99 / 0.96, b = 48 1.00 / 0.96 /
  0.94 / 0.93. The sequential order already streams well (prefetching); the wavefront's live set is ~2g bulge regions
  (2.5 b² elements each, ~+30% line waste: 46 KB at b = 48 FP64 vs 128 KB L1, 128-byte lines), and padding can't align
  segments because every sweep shifts the window by one row. The memory pattern is ~40% of the chase (14.8 of ~39 ms
  at b = 48, fast state). Dropped, as is an aligned scratch window.
- **`l` without the backfire: `lowerGEMM` (`u`).** `scripts/kernels/lower_update.c` (`build/lower-update.txt`): the
  slow state follows the last kernel: the chase's memory pattern right after `dsyr2k` 27.8 / 27.2 / 15.1 ms (runs
  1 / 2 / 3, n = 1204), after 2 x `dgemm` or a lower triangle by `dgemm` blocks ~15, after `dsyr2k` + one tiny `dgemm`
  (256 x 256 x 48) 15.3 (fast again; a 12 MB copy or a 3 ms wait doesn't reset it). The rank-2b lower update by
  `dgemm` per block column of 128 costs 1.02 / 2.82 ms vs `dsyr2k` 0.94 / 2.54 and 2 x `dgemm` 2.12 / 5.77 (n = 1204 /
  1788). New `SBROptions.lowerGEMM` (with `l`): that update in place of `syr2k`. End to end (`build/lowergemm.txt`,
  median of 7; total / stage 1 / chase, ms): FP64 n = 1204 b = 48 `tcsgx` 188.0 / 32.7 / 39.5, `tclsgx` 192.2 / 26.8 /
  44.9, **`tclsugx` 180.8 / 26.1 / 39.2**; n = 1788 b = 64 531.3 / 100.2 / 107.5, 531.8 / 73.3 / 137.8, **504.3 / 72.1
  / 108.2**; n = 776 60.3 / 64.3 / **59.6**. FP32 n = 1204 100.7 / 102.4 / **98.6**, n = 1788 248.8 / 254.5 /
  **239.9**, n = 776 37.3 / 38.6 / **36.7**. 8 x 1 throughput (`build/lowergemm-8x1.txt`, jobs/s, `tcsgx` →
  `tclsugx`): FP64 n = 1204 16.71 → 20.23 (+21%), n = 1788 5.34 → 6.16 (+15%); FP32 n = 1204 50.22 → 50.23, n = 1788
  16.59 → 18.71 (+13%). Against the 1-stage solver's 8 x 1 throughput measured earlier the same day (n = 1788: FP64
  6.44, FP32 15.63 jobs/s): 0.96x FP64 and 1.20x FP32, the first configuration where 2-stage beats 1-stage on
  throughput (to confirm in one run). `recommended()` now uses `l` + `u` from n = 700. Accuracy unchanged.
