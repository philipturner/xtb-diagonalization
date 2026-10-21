# Findings

Answers to the questions in `BRIEF.md`, Phases 1–4. Details and raw numbers: `baseline.md`, `LOG.md`, commit messages.
All times: M1 Max, Accelerate, 1 thread, median of 5 cold single points, 300 K, production settings.

## Headline

`diagonalization` (fork commit `6c6fd43`) is **1.21–1.70x faster** than production (upstream main) and passes every
gate on all 174 runs. With opt-in FP32 early cycles (`XTB_SCC_FP32_RMSQ=1e-3`) it is **1.52–2.0x** faster, but fails
the gates on five 100 K runs.

| Case (n) | production code | `diagonalization` | + FP32 early cycles (opt-in) |
| --- | --- | --- | --- |
| `t2_R` (124) | 0.046 s | 0.030 s (1.54x) | 0.026 s (1.77x) |
| `facet_closed` (412) | 0.585 s | 0.409 s (1.43x) | 0.353 s (1.66x) |
| `s12_C31` (419) | 1.579 s | 1.227 s (1.29x) | 1.014 s (1.56x) |
| `facet_dbpair_tool` (448) | 1.062 s | 0.878 s (1.21x) | 0.699 s (1.52x) |
| `s3_S42_159` (619) | 2.240 s | 1.630 s (1.37x) | 1.413 s (1.59x) |
| `diamond233` (776) | 2.414 s | 1.421 s (1.70x) | 1.209 s (2.00x) |

The 2x goal at n ≈ 410–450 is not reached under the gates. What is left: the eigensolve is still 67–76% of a single
point, and within it the tridiagonal reduction (`dsytrd`, BLAS2) is about half.

## Phase 1

- **Production library:** upstream main (`68fe777`) plus two GFN-FF file-I/O commits, not release 6.7.1; a rebuild
  reproduces `reference/` bit for bit. v6.7.1 has wrong gradients on symmetric structures (fixed upstream in
  #1208/#1214); use it for timing only.
- **Breakdown (production code):** eigensolve 12–14 ms per cycle at n = 412–448, 79–80% of a cycle, 56–69% of a
  single point (48% at n = 776, where integrals and gradient took 35%). Inside it: `dsytrd` 42%, `dstedc` 30%,
  `dormtr` 14%, `dsygst`/`dtrsm` the rest.
- **Cached Cholesky** (v6.7.1 → main): 1–4%.
- **Pseudodiag history:** no recorded reason; off since before the public history (2019) and dead code since 2020.
- **H1 (dsymv L2-bound):** consistent, not proven. 8 × 1 thread: per-process efficiency 60–67%, worse at larger n;
  AMX sharing per cluster not ruled out. **8 × 1 thread gives 15–23% more throughput than production's 6 × 2.**
- **H3 (AMX not the bottleneck):** confirmed: `dgemm` 18–20% of samples; BLAS2 `dsytrd` and scalar `dstedc` dominate.
- **Warm starts** (separate from cold): chained −35 to −55% SCC cycles, fixed seed −12 to −33%, same solutions.

## Accuracy gates (adopted)

SCC convergence noise sets the floor: at accuracy 1.0 a converged result moves up to 7.7e-5 Eh/bohr (6.3 pN) when
only its start changes. So method error is gated at accuracy 0.01 (gradient ≤ 2e-6 Eh/bohr = 0.16 pN, against the
production library at the same accuracy), and production accuracy only checks same solution and convergence.
`facet_closed_rms15` uhf 2 at 300 K has two near-degenerate SCF solutions.

## Phase 2

- **GEMM-based transform:** `dtrmm` instead of `dsygst` + `dtrsm` saves 3–5% of the eigensolve; full `dgemm` doesn't
  help. My version (single hook, `diag/hook-trmm`) gave 0–4% per single point and is superseded by #1466.
- **Drivers:** keep `dsyevd`/`dstedc`. Accelerate's `dsyevr` is 1.6–3x slower (all or only the lowest eigenpairs);
  on the tridiagonal matrix, `dstemr` and `dstebz`+`dstein` for the lowest third are 3–9x slower than `dstedc` for all.
- **Workspace:** allocated once per geometry (`init`), as it should be.
- **PRs by g8j39**, each passing both gate sets, merged into `diagonalization`:
  #1465 integrals −10–26%, #1466 SCC linear algebra (half-flop congruence, `dsytrd`+`dstedc`, back-transformation of
  occupied orbitals only) −6–15%, #1464 three-body dispersion gradient −0–11% (only at n = 776 noticeable). The ATM
  term #1464 changes runs once per single point after the SCC (it isn't self-consistent in GFN2); its extra memory is
  four atom × atom matrices (2.4 MB at 275 atoms), freed on return.
- **Hook point:** #1466's `fact_solve_subspace` / `fact_backtransform` is where every SCC eigensolve now goes.

## Phase 3

- **H2 (H changes little after a few cycles):** true only for the tail. In the previous cycle's orbital basis the
  largest off-diagonal element of H falls from ~1 Eh to ~1e-3 Eh only after 15–35 cycles; on small-gap radicals it
  stays at 1e-3–1e-2 Eh for most of the run.
- **Option A, pseudodiag:** not useful. One occupied–virtual sweep is first-order only (density error ≥ 1e-4 late,
  up to 0.5 early) and costs nearly as much as `dsyevd` at these sizes.
- **Option B, Ogita–Aishima refinement:** two steps with exact treatment of near-degenerate clusters reach 1e-9–1e-12
  density error in the last ~10–25% of cycles, at about 40% of the exact cost, but diverge earlier. Estimated gain if
  restricted to the tail: 5–19% per single point. Not yet implemented in xTB.
- **Option C, FP32 early cycles:** implemented (opt-in). FP32 eigensolve is 30% cheaper; density error 1e-6–1e-4, no
  failures; single points 14–17% faster. Passes every gate at 300 K and 1000 K; at 100 K four slow `s12_C31` radicals
  need 15–16% more cycles (gate 10%) and `s12_C00` uhf 3 stops converging at accuracy 0.01. Even one FP32 cycle
  changes those trajectories; a later switch doesn't help. **Your call** whether 300 K-only use is acceptable.
- **Fragile runs:** `facet_bare_dimer` uhf 0 (300 K) and `s12_C00` uhf 3 (100 K) fail to converge from some perturbed
  starts; the 100 K `s12_C31` radicals change their cycle count by up to 16% under any 1e-6 perturbation.

## Phase 4

- **Two-stage reduction doesn't pay at production sizes.** Accelerate's `dsytrd_2stage` (new LAPACK interface,
  reduction only) is no faster than `dsytrd` at n = 412–448 and 17–20% faster at 619–776, before the second-stage
  eigenvector back-transformation that neither Accelerate nor reference LAPACK provides. The brief's expectation holds.
- FP32 `ssytrd` is 1.4x faster than `dsytrd`, `sstedc` 1.5x faster than `dstedc`: the FP32 route (Phase 3 C) is the
  cheaper way to the same gain.

## Decisions for you

1. Whether to accept FP32 early cycles for 300 K production use (it fails only 100 K gates).
2. Whether to push `diagonalization` and open nothing upstream yet (the three PRs are g8j39's, still open).
3. Whether to run production as 8 processes × 1 thread (+15–23% throughput, measured).
