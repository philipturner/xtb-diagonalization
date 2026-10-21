# Findings

Answers to the questions in `BRIEF.md`, Phases 1–4. Details and raw numbers: `baseline.md`, `LOG.md`, commit messages.
All times: M1 Max, Accelerate, 1 thread, median of 5 cold single points, 300 K, production settings.

## Headline

`diagonalization` (fork commit `aa5ecd7`; default path identical to `e891e78`) is **1.51–1.89x faster** than the
production library per single point and passes every gate on all 174 runs. With opt-in FP32 early cycles
(`XTB_SCC_FP32_RMSQ=1e-3`) it is **1.89–2.19x** (n = 448: 1.90x), but fails the gates (see Phase 3). Timing: three
round-robin rounds of median-of-5, spread ≤ 3.1% (the FP32 column with the vectorized FP32 reduction: one set of 5).

| Case (n) | production library | `diagonalization` | + FP32 early cycles (opt-in) |
| --- | --- | --- | --- |
| `t2_R` (124) | 0.045 s | 0.028 s (1.64x) | 0.024 s (1.89x) |
| `facet_closed` (412) | 0.581 s | 0.349 s (1.66x) | 0.294 s (1.97x) |
| `s12_C31` (419) | 1.553 s | 1.032 s (1.51x) | 0.820 s (1.89x) |
| `facet_dbpair_tool` (448) | 1.101 s | 0.731 s (1.51x) | 0.581 s (1.90x) |
| `s3_S42_159` (619) | 2.224 s | 1.400 s (1.59x) | 1.160 s (1.92x) |
| `diamond233` (776) | 2.343 s | 1.241 s (1.89x) | 1.069 s (2.19x) |

Where it comes from (each step passes both gate sets): #1465 integrals, #1466 SCC linear algebra, #1464 dispersion
gradient (upstream PRs by g8j39, merged), multipole moments summed per atom pair (`6c4b68f`, −3.5–5.5%), and a
tridiagonal reduction with a NEON symmetric matrix-vector product (`e891e78`, −9–13%).

**Throughput, what production sees: 2x reached.** `diagonalization` at 8 processes × 1 thread runs **2.07x** as many
single points as the production library at production's 6 × 2 (n = 448: 4.17 → 8.63 jobs/s, measured back to back)
and about **2.0x** at n = 655 (1.80 → 3.59 jobs/s). About 1.3x of it is the library at 6 × 2 (n = 448: 6.21 jobs/s), the
rest the switch to 8 × 1.

**Drop-in candidate: branch `diag/production` (`05f17ac`)** = `diagonalization` + the fork's `swift-xtb-2026` GFN-FF
commits that the production library carries. Without them, GFN-FF writes `gfnff_topo` into the working directory.
With them, GFN-FF is bit-identical to production and writes nothing; GFN2 is identical to `diagonalization`.
The production Python wrapper (`xtb_lib.py`, unchanged, `LIB` pointed at the build) runs it and agrees
with the production library to 1e-13 Eh and 7e-13 Eh/bohr on three production systems. Swift not checked.

Per single point at n ≈ 410–450, 2x is not reached under the gates (1.51–1.66x). What is left: the eigensolve is
still ~70% of a single point; within it `dstedc` (its scalar secular-equation solver) and the reduction share most.

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
- **Beyond the brief:** Accelerate's `dsymv` reads the matrix at ~30 GB/s; a NEON kernel reaches ~50 GB/s. LAPACK's
  `dsytrd` ported with it (`src/mctc/lapack/sytrd.c`) is 30% faster and, with Accelerate's `dsymv`, reproduces
  Accelerate's `dsytrd` bit for bit. `mmompop` restructured by atom pairs: −40% of the population stage.

## Phase 3

- **H2 (H changes little after a few cycles):** true only for the tail. In the previous cycle's orbital basis the
  largest off-diagonal element of H falls from ~1 Eh to ~1e-3 Eh only after 15–35 cycles; on small-gap radicals it
  stays at 1e-3–1e-2 Eh for most of the run.
- **Option A, pseudodiag:** not useful. One occupied–virtual sweep is first-order only (density error ≥ 1e-4 late,
  up to 0.5 early) and costs nearly as much as `dsyevd` at these sizes.
- **Option B, Ogita–Aishima refinement:** offline, two steps with exact treatment of near-degenerate clusters reach
  1e-9–1e-12 density error in the last ~10–25% of cycles, at about 40% of the exact cost, but diverge earlier.
  Implemented opt-in on `diag/refine` (`f349b0e`, `XTB_SCC_REFINE_RMSQ`): up to three steps, accepted only if the
  last correction is < 1e-5, else exact solve and back-off. **Passes every gate with SCC cycle counts unchanged**, but
  the gain is small: 1–4% faster at n ≤ 448, 3–7% slower at n ≥ 619 (storing all eigenvectors and failed attempts
  cost more than the refined cycles save). Not merged.
- **Option C, FP32 early cycles:** implemented (opt-in). FP32 eigensolve is 30% cheaper (more with the vectorized FP32
  reduction); density error 1e-6–1e-4; single points 14–25% faster. **Not safe as is.** With Accelerate's `ssytrd` it
  passed every gate at 300 K and 1000 K but failed five 100 K runs (cycle count, one convergence at accuracy 0.01);
  with the equally accurate vectorized FP32 reduction, `facet_bare_dimer` uhf 0 at **300 K** stops converging instead.
  Even one FP32 cycle changes those trajectories; a later switch doesn't help. Which fragile run breaks depends on
  rounding details, so FP32 needs a safeguard (for example: fall back to FP64 when the SCC stalls) before production use.
- **Cycle count (side experiment, `diag/broyden`):** Broyden damping 0.2–0.3 instead of xTB's 0.40 cuts SCC cycles on
  the slow open-shell systems by 19–51% and over the grid by 13.6%, but moves `facet_bare_dimer` uhf 0 at 100 K to a
  different SCF solution 3.0 mEh higher (gap 1.54 instead of 0.20 eV) and doubles another run's cycles. Not usable as
  a global default; cycle count is the biggest remaining lever if a safer accelerator can be found.
- **Fragile runs:** `facet_bare_dimer` uhf 0 (300 K) and `s12_C00` uhf 3 (100 K) fail to converge from some perturbed
  starts; the 100 K `s12_C31` radicals change their cycle count by up to 16% under any 1e-6 perturbation. The same
  fragility exists in production: any change of rounding (BLAS version, thread count) could flip such a run.

## Phase 4

- **Two-stage reduction doesn't pay at production sizes.** Accelerate's `dsytrd_2stage` (new LAPACK interface,
  reduction only) is no faster than `dsytrd` at n = 412–448 and 17–20% faster at 619–776, before the second-stage
  eigenvector back-transformation that neither Accelerate nor reference LAPACK provides. The brief's expectation holds.
- FP32 `ssytrd` is 1.4x faster than `dsytrd`, `sstedc` 1.5x faster than `dstedc`: the FP32 route (Phase 3 C) is the
  cheaper way to the same gain.
- **`ssyevd_2stage` inaccuracy:** doesn't reproduce. On this macOS (15.7) it exists only in the new interface
  (`$NEWLAPACK`); its eigenvalues are as accurate as one-stage FP32 (relative error 1.5–3.6e-6 on four GFN2
  matrices), and it refuses eigenvectors (info −1), like `dsyevd_2stage`.
- **New-interface `dstedc`:** 5% slower than the legacy one; nothing to gain from switching interfaces.

## Decisions for you

1. Whether FP32 early cycles are worth a safeguard (restart the SCC in FP64 when it stalls): +12–25% per single point,
   but as is it breaks fragile runs, including one at 300 K. It stays off by default.
2. Whether to push `diagonalization` and `diag/production` (local commits only; the three PRs in them are g8j39's,
   still open upstream), whether to hand `diag/production` over as the production library (after a Swift check), and
   whether to offer `sytrd.c` and the `mmompop` change upstream.
3. Whether to run production as 8 processes × 1 thread: +15–23% throughput with the current library, and with
   `diagonalization` about 2x production's current throughput (measured).
4. Whether to pursue a safer SCC accelerator (the damping experiment shows cycle count is the biggest lever left).
