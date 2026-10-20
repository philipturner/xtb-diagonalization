# xTB Diagonalization Profiling Brief

Oct 19, 2026 · @Philip Turner

## Goal and success criteria

Cut GFN2-xTB single-point wall time on Apple silicon (M1 Max) by at least 2x at n ≈ 410–450 basis functions, where most production work sits (up to n ≈ 650 for the Stage 3 regions), by attacking the per-SCC-cycle eigensolve. The speedup must hold for cold-start single points, which are the default in this project. Every claim of a speedup comes from a measurement, not an estimate.

Measure these, before and after each change:

- **Wall time per single point:** median of 5 runs (after one untimed warm-up run of the process; each single point itself still starts SCC cold) at n = 124, 410–450, 619 and 776 (the test set below).
- **Breakdown:** share of time in `dsygvd` and its children, and per-cycle time for H build, eigensolve, Fermi smearing, density matrix and Broyden mixing.
- **SCC cycle count** per system.
- **Throughput:** jobs per second at n ≈ 450 and n ≈ 750, comparing the current production setup (5–6 processes × 2 threads) against 8–10 processes × 1 thread.
- **Accuracy:** the gates in the Validation section. A change that fails a gate doesn't count, however fast it is.

## What is already known

The eigensolve is 75% of a single point, and its biggest piece is one BLAS2 routine. This Instruments profile is from `Experiments/OptimizingDiagonalization/Diamond233.png` and `DSYEVD.png`: n = 776, 19 SCC cycles, Homebrew `libxtb.6.dylib` linked to Accelerate.

| Routine (call path) | Share of total | Kind of work |
| --- | --- | --- |
| `mctc_dsygvd` → `DSYGVD` (everything below) | 74.6% | — |
| `DSYTRD` → `DLATRD` → `DSYMV` | 29.8% | BLAS2, memory-bound |
| `DSYTRD` → `DLATRD` → `DGEMV` | 5.8% | BLAS2 |
| `DSYTRD` → `DSYR2K` | 3.1% | BLAS3 |
| `DSTEDC` (divide and conquer) | 15.5% | Mostly scalar |
| `DORMTR` (back-transformation) | 9.6% | BLAS3 |
| `DSYGST` + `DTRSM` + `DPOTRF` | 8.8% | BLAS3 |
| `mmompop`, `setvsdq`, `dmat`, H build | \~15% | xTB's own code |

Other facts already established:

- Switching OpenBLAS to Accelerate made one thread per job optimal (down from 8), and running many single-threaded jobs in parallel raised throughput about 20x.
- `DPOTRF` appears under `DSYGVD` in the profile, so S is refactored every cycle in that build. Current xTB master no longer does this (see Code map).
- `OptimizingDiagonalization.md` projects 1.86 s → 0.89 s at n = 776 if diagonalization cost only as much as divide and conquer, and → 0.55 s if the remaining LAPACK work and other kernels also moved to FP32. Neither was implemented.

Hypotheses this work should confirm or kill:

1. **H1:** `DSYMV` is limited by L2 bandwidth. Parallel jobs scale with cores until working sets overflow L2 (12 MB per 4-core P-cluster on M1 Max; an n = 776 FP64 matrix is 4.8 MB, n = 1,000 is 8 MB).
2. **H2:** After the first few SCC cycles, the Hamiltonian changes little, so most cycles don't need a full diagonalization.
3. **H3:** The matrix units (AMX) are not the bottleneck, since only \~20% of the eigensolve is BLAS3.

### Production sizes and the test set

The production matrices are smaller than the n = 776 profile: n ≈ 120–660, mostly 410–450. These counts come from GFN2's basis, verified in `src/xtb/gfn2.f90` (`nShell`, `angShell`): H 1 function (1s only), C and O 4, Si/Ge/Sn 9 (s, p, d).

The test set is ready in `structures/` (30 `.xyz` files, `MANIFEST.tsv` with atoms, n, uhf, source and what each case exercises; provenance in `scripts/provenance/make_testset.py`). All are neutral. Groups:

| Group | Files | Atoms | n | Notes |
| --- | --- | --- | --- | --- |
| T2 cluster: Si₈H₁₄ + Ge(CH₃)₃–C≡C• | `t2_R`, `t2_TS` | 37 | 124 | radical; TS = H shared between C(t) and Si |
| swift-xtb diamond cells | `diamond122/222/233` | 82–275 | 196–776 | closed shell, degenerate levels; 0.97 Å H–H contacts as built in swift-xtb |
| Stage 1/2 common regions | `s12_C00`, `s12_C31` | 94–103 | 396–419 | radical; the slowest-converging production systems |
| 40-Si facet, no tool | `facet_closed`, `facet_bare_dimer`, `facet_dbpair` | 90–92 | 410–412 | closed shell; one bare Si=Si dimer (small gap); bare DB pair (diradical) |
| 40-Si facet + C₂ radical tool | `facet_dbpair_tool`, `facet_capped_tool`, `facet_capsi_product`, `facet_h_abstracted` | 105–107 | 448–450 | Stage 3 production; bare DB pair + tool (3 unpaired, production uhf 1) is the hardest SCF |
| Stage 3 explicit GFN2 regions | `s3_S40_156`, `s3_S42_159`, `s3_S49_163` | 156–163 | 592–655 | tool + facet; S42 just before a bond rupture |
| Off-equilibrium copies | `<id>_rms05`, `<id>_rms15` of six of the above | | | seeded Gaussian displacements, 0.05 Å rms (thermal size) and 0.15 Å rms (max forces 3–19 eV/Å) |

Reference results from the production library are in `reference/` (`scripts/make_reference.py`): for every structure, uhf = the two lowest spin states its electron count allows (0/2 or 1/3) × electronic temperature 100 / 300 / 1000 K, 174 single points. Each JSON holds the energy, full gradient, Mulliken charges, orbital eigenvalues and occupations (C API), HOMO–LUMO gap, SCC iteration count (parsed from xTB's own iteration table), converged flag and wall time; `SUMMARY.tsv` has one line per run. Settings are production's: accuracy 1.0, maxiter 250, fresh results object per single point.

Measured with the production library (4 single-threaded jobs at once, so absolute times are indicative only):

| n | SCC iterations at 300 K | Wall per single point | Per cycle |
| --- | --- | --- | --- |
| 124 | 20–32 | 0.05 s | \~2.5 ms |
| 364–412 | 15–52 | 0.45–1.4 s | \~25 ms |
| 396–419 (Stage 1/2) | 76–116 | 2.4–2.8 s | \~25–30 ms |
| 448–450 | 33–49 | 1.0–1.6 s | \~30 ms |
| 592–655 | 37–57 | 2.4–4.6 s | \~60–80 ms |
| 776 (diamond233) | 19–26 | 2.4–3.4 s | \~125 ms |

What this changes:

- **One eigensolve per cycle, even with `--uhf`.** In xTB's own `scc` loop, `fact_solve` runs once per cycle and `fermismear` applies separate α and β occupations to the same eigenvalues; GFN2 here has no spin-dependent Hamiltonian. The C API's uhf argument to xtb\_newMolecule only sets the unpaired-electron count on this same path.
- **15–116 SCC cycles per single point (measured),** 30–60 typical for production systems and 76–116 for the Stage 1/2 regions, against 13–19 for the diamond cells. Three runs hit the 250-iteration cap (uhf 3 or 2 at 100 K). That makes Phase 3 (eigenvector reuse) and cycle count itself more important, and the diamond cells alone are not representative.
- **Per-cycle cost is \~25–30 ms at n ≈ 410–450** (measured, whole cycle), so fixed overheads, workspace handling and driver choice matter as much as asymptotic speed.
- **Working sets mostly fit in L2:** an n = 450 FP64 matrix is 1.6 MB, n = 650 is 3.4 MB. H1 may only bite for the Stage 3 regions and diamond233.
- **The converged state depends on the settings.** Several systems land on different SCF solutions at different electronic temperatures (for example `facet_dbpair` uhf 0: gap 0.02 eV at 100 K, 1.51 eV at 300 K). Any change that alters which solution is found must be caught by the gates, not mistaken for a speedup.

## Code map

The released xTB still runs a full `dsygvd` every SCC cycle, while master already caches the Cholesky factor of S. Line numbers are from [grimme-lab/xtb](https://github.com/grimme-lab/xtb) master at commit `68fe777` (Oct 8, 2026; `meson.build` says 6.7.1) and may drift.

### xTB (Fortran)

| File | What's there |
| --- | --- |
| `src/scc_core.f90`, `subroutine scc` (\~line 255) | The SCC loop. \~413: factors S once with `mctc_potrf`. \~475: `solver%fact_solve(env, H, S_factorized, emo)` every cycle. Then `fermismear`, `dmat`, `broyden`. |
| `src/mctc/lapack/eigensolve.F90` | `TEigenSolver`. `mctc_dsygvd` = full `dsygvd`. `mctc_dsygvd_factorized` = `dsygst` + `dsyevd` + `dtrsm`. A cuSOLVER path sits under `USE_CUSOLVER`. |
| `src/scc_core.f90`, `solve` / `solve4` (\~lines 802–980) | Legacy solvers, now unused. After the first full solve, they rotate H into the previous MO basis (2 GEMMs), call `pseudodiag`, and rotate back (1 GEMM). `solve4` is an FP32 variant. |
| `src/pseudodiag.f90` | Stewart–Császár–Pulay (1982) pseudodiagonalization: occupied–virtual Jacobi rotations, FP32 scratch, OpenMP. |
| `src/scf_module.F90` (\~line 281) | `startpdiag=1000`, commented "large number=never", so pseudodiag is disabled. |

Release **v6.7.1** calls `solver%solve(env, H, S, emo)` at `scc_core.f90` line 440: a full `dsygvd`, including `dpotrf`, every cycle. That matches the profile. The cached-Cholesky path exists only on master.

### Production entry point: the C API

Production uses ONE library: `../claude-molecular-renderer/libxtb.dylib`, at the root of the mechanosynthesis repository (a sibling of this one) (xtb 6.7.1 built against Accelerate, shipped with the molecular-renderer v1.0.2 binaries). Both that repository's Python wrapper `Sources/Workspace/Data/stage3/scripts/lib/xtb_lib.py` and the Swift code (swift-xtb `main`) load it; Homebrew's xtb is not used in production. A deliverable is a drop-in replacement for this file, checked from both Python and Swift. Python calls the xTB C API (`include/xtb.h`) directly through ctypes; no tblite is involved. `scripts/make_reference.py` is a self-contained copy of the production call sequence (plus charges, orbitals and the SCC iteration count) and is the benchmark driver to start from. The path below matches the top of the n = 776 profile, which starts at `xtb_singlepoint`.

```
Python (ctypes) → libxtb.dylib
  xtb_singlepoint            include/xtb.h ~line 237
  → singlepoint_api          src/api/interface.f90 ~line 40
  → calc%singlepoint → scf → scc   (src/scc_core.f90)
  → solver%solve / fact_solve      (src/mctc/lapack/eigensolve.F90)
```

Three consequences for this work:

- **Every Fortran change ships by rebuilding `libxtb.dylib`.** The Python processes must load the rebuilt library, not Homebrew's. Point the ctypes load path at the build tree and verify it at runtime, for example by logging `xtb_getAPIVersion` and the resolved path.
- **Warm starts depend on how results handles are used.** `singlepoint_api` keeps a restart object (`res%chk`, \~lines 83–99) inside the `xtb_TResults` handle and reuses it as the starting wavefunction, as long as the basis dimensions match. Production Python deliberately creates a fresh `xtb_newResults` per single point (cold start, a project rule since 2026-10-16: results must not depend on what ran before, because GFN2 can land on different SCF solutions at neighbouring geometries). The Swift side has a reproducible alternative, `xTB_Seed` (swift-xtb `main`: `calculator.seed` plus `withSeed(seed) { ... }` warm-starts every single point in the block from one fixed seed, not chained). Treat warm starts as an engineering variable, not the default: they may be usable in about half of practical work (relaxations and scans with a fixed seed), while the rest needs pure cold starts. Report every result cold first; report warm-start numbers separately, as extra insight into where the cycles go.
- **Settings come from API calls.** Production Python (`xtb_lib.GFN2`): `xtb_setAccuracy` 1.0, `xtb_setElectronicTemp` 300 K, `xtb_setMaxIter` 250, verbosity 0, charge 0, uhf per system (`MANIFEST.tsv`). The benchmark must use the same values, and the 100 K and 1000 K references show how sensitive each case is to the electronic temperature.

### swift-xtb (reference only, FP32)

All paths are under `philipturner/nanotech-code-archive/swift-xtb`.

| File | What's there |
| --- | --- |
| `Sources/LinearAlgebra/Diagonalization/Diagonalization.swift` | Driver. FP32 only; default block size 32, small block 8. Stages: band form → bulge chasing → divide and conquer → back-transform (stage 2) → back-transform (stage 1). |
| `.../BandForm.swift` | Panel factorization with WY transforms. The trailing GEMM is sized (blockSize, n, n), the full matrix on every panel instead of the shrinking trailing block, and doesn't exploit symmetry. |
| `.../BulgeChasing.swift` | Stage 2, small `sgemm_` calls. |
| `.../DivideAndConquer.swift` | Accelerate `sstevd_`. |
| `.../BackTransformation.swift` | Both back-transformations. |
| `Sources/LinearAlgebra/BLAS/` | `GEMM.swift` (`sgemm_` wrapper), `WYTransform.swift`, `ReflectorGeneration.swift`. |
| `Tests/AlgorithmTests/DiagonalizationTests.swift` | Tests for divide and conquer, bulge chasing and two-stage tridiagonalization, plus a benchmark. |
| `Experiments/OptimizingDiagonalization/OptimizingDiagonalization.md` | Benchmarks, the projection above, a TODO list (exploit symmetry, skip multiplications by zero, parallelize bulge chasing). |
| `Experiments/DenseDiagonalizationExperiment/README.md` | GFLOPS/k tables; AMX `sgemm_` efficiency by block size (566 GFLOPS at 32, \~1,000 at 256–512); Accelerate `ssyevd_2stage_` gave inaccurate eigenvalues. |
| `Sources/Workspace/DiamondSystems.swift` | The diamond test cells used for the profiles. |

Measured status: the Swift `Diagonalize` was slower than Accelerate `dsyevd` at n ≤ 416 (4.9 vs 5.9 GFLOPS/k for eigenvectors on a continuous spectrum).

## Repository layout and source management

This repository (github.com/philipturner/xtb-diagonalization, private) is the whole project. It was factored out of the mechanosynthesis repository (`../claude-molecular-renderer`), which stays the production user of xTB.

| Path | What it is |
| --- | --- |
| `BRIEF.md` | This plan and its acceptance criteria |
| `xtb/` | Git submodule: the fork github.com/philipturner/xtb, at `68fe777` (= grimme-lab/xtb `main`, 2026-10-08). Remote `upstream` = grimme-lab/xtb. Full history, so `git log -S` works |
| `structures/` | Test set (30 `.xyz`) and `MANIFEST.tsv` |
| `reference/` | Production-library results per structure × uhf × electronic temperature, and `SUMMARY.tsv` |
| `scripts/make_reference.py` | Benchmark/reference driver: production C API call sequence; `--lib` picks the library, `--out` the output folder |
| `scripts/provenance/make_testset.py` | How the structures were made (ran once in the mechanosynthesis repo; doesn't run here) |
| `build/` | Build trees and scratch, git-ignored |

Workflow for xTB changes:

- Work on a branch inside the submodule (`git -C xtb switch -c diag/<step>`), one commit per step with its measurement in the message. Commit the new submodule pointer in this repository at the same time, so every result here names the exact xTB commit it came from.
- Differences from upstream are plain git: `git -C xtb diff upstream/main`, `git -C xtb log upstream/main..HEAD`. Upgrading is `git -C xtb fetch upstream` plus a rebase or merge.
- Pushing the branch to the fork, and anything outward-facing (pull requests upstream), is the user's call; ask first.
- Baseline v6.7.1 (the version of the production library): `git -C xtb fetch upstream --tags`, then `git -C xtb worktree add ../build/xtb-v6.7.1 v6.7.1`. Don't edit it.
- Build trees go under `build/` (for example `build/xtb-build-main/`, `build/xtb-build-v6.7.1/`). Meson subprojects (mctc-lib and others) are downloaded when a build is configured: allow their hosts once, then build offline.
- The mechanosynthesis repository is read-only from here: never write into `../claude-molecular-renderer`, and never replace its `libxtb.dylib`. A faster library is handed over only after the gates pass from both Python and Swift, and the user decides.

## Phase 1: Profiling harness and baseline

The deliverable is a reproducible baseline that every later phase is measured against, built from source rather than taken from a prebuilt binary.

1. **Build two xTB trees on macOS arm64 against Accelerate:** v6.7.1 (worktree `build/xtb-v6.7.1/`, the version of the production library) and main (the `xtb/` submodule). `meson_options.txt` has no Accelerate choice; `-Dlapack=custom` with `custom_libraries` is the likely route, but check how Homebrew's formula links it first. Confirm with `otool -L` that the binary links Accelerate, not OpenBLAS. Check that your v6.7.1 build reproduces `reference/` (energies to 1e-10 Eh, same SCC iteration counts). If it doesn't, find out how the production `libxtb.dylib` was built before going further.
2. **Fix the environment for every run:** `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, on AC power, Low Power Mode off. Record macOS version and chip. Time only when the Mac is otherwise idle: other Claude sessions run long GFN2 jobs here. Check `ps`/load first, and ask the user for a quiet window if needed.
3. **Use the prepared test set** (`structures/`, see "Production sizes and the test set") and `scripts/make_reference.py` as the driver. It loads the libxtb.dylib given by `--lib`, makes the same settings calls as production and creates one fresh results object per single point. It prints the resolved library path; also record `xtb_getAPIVersion`. Write every non-production run to `--out build/...`, never into `reference/`. For warm-start measurements, add a mode that reuses one results handle along a sequence (for example the `_rms05` copies, or a scan you generate), and keep it separate from the cold numbers. Keep the CLI only for quick cross-checks, `xtb mol.xyz --gfn 2 --sp`.
4. **Instrument the SCC loop** in `scc_core.f90`: wall-clock timers around H build, the eigensolve call, `fermismear`, `dmat` and `broyden`, printed per cycle with the cycle count. Keep this as a patch you can apply to both trees.
5. **Profile** each test system with `xctrace record --template 'Time Profiler'` and save the call-tree breakdown under `dsygvd`, so it can be compared against the table above.
6. **Run the concurrency sweep:** the production setup (5–6 processes × 2 threads) against 1-thread processes at 1, 2, 4, 8 and 10 concurrent, at n = 448 and n ≈ 700. Report jobs per second and how many land on E-cores; M1 Max has 8 P-cores and 2 E-cores, so 6 × 2 threads oversubscribes the P-cores. This tests H1.
7. **Write `baseline.md`:** versions, build flags, environment, median times, per-cycle breakdown, cycle counts, final energies and gradients printed to 1e-10 Eh. The v6.7.1-versus-master difference measures the cached-Cholesky change for free.

## Phase 2: Low-risk changes in the xTB tree

These are pure reformulations, so results must match the baseline to 1e-10 Eh. Expect a few percent, not a multiple; their main value is a clean place to hook Phase 3.

1. **Cached Cholesky.** Work on master, which already factors S once per geometry. Confirm the measured saving against v6.7.1 (the profile suggests \~1.5% at n = 776 from `DPOTRF`).
2. **GEMM-based transform.** Replace `dsygst` + `dtrsm` in `mctc_dsygvd_factorized` with a matrix X formed once per geometry, X = U⁻¹ from the Cholesky factor. Each cycle becomes H′ = Xᵀ H X (two GEMMs, or `dsymm` + `dgemm`), `dsyevd` on H′, then C = X C′ (one GEMM). These moves \~7% of runtime onto matrix-unit-friendly GEMMs. Measure; don't assume it's faster at small n.
3. **One hook point.** Route every eigensolve through a single new subroutine that receives the cycle number, the previous eigenvectors and eigenvalues, and the factor X. Phase 3 changes only this routine.
4. The three under-review PRs by https://github.com/g8j39 on the grimme-lab/xtb repository: 1464, 1465, 1466. Small incremental wins worth investigating.

At n ≈ 450–750, also compare drivers on the same transformed H′: `dsyevd`, `dsyevr` for all eigenvectors, and `dsyevr` for only the lowest eigenvectors needed (occupied plus a margin of virtuals for Fermi smearing). Tridiagonalization still dominates all three, so expect differences of tens of percent at most, but at these sizes the cheapest driver may differ from the n = 776 result. Check that workspace is allocated once per geometry, not per call.

Commit each step separately, with its measurement in the commit message.

## Phase 3: Eigenvector reuse across SCC cycles

This is the largest expected win: with 19 cycles at n = 776, most full diagonalizations repeat work on a Hamiltonian that barely changed. Try the options in this order and keep the cheapest one that passes every gate.

Before touching the eigensolver, measure warm starts as a side experiment. Run a 20-step scan through the C API three ways:

1. a fresh `xtb_newResults` per point (production);
2. one handle reused with `xtb_updateMolecule` (chained);
3. one fixed seed for every point, like `xTB_Seed`.

Compare SCC cycles per point, final energies and whether every point reaches the same SCF solution as its cold start. This only says how much warm starts could save where they are allowed (roughly half of practical uses). Every option below must also deliver its speedup on cold starts, which stay the default.

1. **Find out why pseudodiag was disabled.** Clone with full history and run `git log -S startpdiag` and `git log -S pseudodiag`. A recorded accuracy or convergence failure changes the plan; report it before writing code.
2. **Option A: re-enable the existing path.** Wire the legacy `solve` logic into the Phase 2 hook: full solve for the first k cycles (try k = 3, 5, 8), then rotate H into the previous MO basis, call `pseudodiag`, and rotate back. Because the previous C satisfies Cᵀ S C = I, Cᵀ H C works directly in the generalized problem. Always do a full solve on the final cycle, so printed orbitals and gradients come from exact eigenvectors.
3. **Option B: GEMM-only refinement.** If A fails the gates, replace pseudodiag with an Ogita–Aishima-style refinement in the previous basis (a few GEMMs per cycle, quadratically convergent). Near-degenerate orbitals, which the symmetric diamond cells have, need the clustered-eigenvalue variant. Fall back to a full solve whenever the residual ‖H′C − CΛ‖ exceeds a threshold.
4. **Option C: mixed precision.** Independently of A or B, test FP32 `ssyevd` for early cycles and FP64 for the last few (`solve4` is the precedent).

Watch two xTB-specific risks:

- **Fermi smearing** needs accurate eigenvalues near the Fermi level, but pseudodiag only rotates occupied–virtual pairs, so its eigenvalues are approximate. Check radical and small-gap systems especially.
- **Broyden mixing** may converge differently with approximate eigenvectors. Report cycle counts, not just time per cycle.

Report per option: time per cycle, total cycles, total time and every gate's result.

## Phase 4: Two-stage diagonalizer, using swift-xtb as reference

Start this only if, after Phase 3, full diagonalizations are still more than \~25% of runtime. The target is beating Accelerate `dsyevd`, with eigenvectors, at n ≈ 450–750 in FP64. At the production sizes this is the phase least likely to pay off: the Swift FP32 version already lost to dsyevd below n = 416.

1. **Benchmark first.** Time Accelerate `dsyevd` against reference LAPACK's two-stage routines at n = 400, 776 and 1,000, eigenvalues and eigenvectors. Separately check whether the `ssyevd_2stage_` inaccuracy noted in the README reproduces; it may be a bug in Accelerate.
2. **Port the Swift stages to FP64**, either as Fortran/C in the xTB tree or as a Swift dylib exposing a C ABI via `@_cdecl`. Use `DiagonalizationTests.swift` as the correctness spec.
3. **Fix the known inefficiencies** before benchmarking the port:
   - Restrict the band-form trailing GEMM to the shrinking (n − k) block; it currently spans the full n × n every panel.
   - Exploit symmetry in the trailing update (a `syr2k`-style update).
   - Sweep block size. The README's AMX table shows `sgemm_` at 566 GFLOPS for 32-wide blocks versus \~1,000 at 256–512. The best band width trades that against bulge-chasing cost, which grows with band width.
4. **Profile each stage** with the five-way breakdown already in `OptimizingDiagonalization.md`, and compare against `DSYTRD` + `DSTEDC` + `DORMTR` from the baseline.
5. **Plug it into the Phase 2 hook** behind a flag, so it can be used for the full-solve cycles while Phase 3 handles the rest.

## Validation and pitfalls

A speedup counts only if it passes every gate on every system in the test set. These thresholds are proposals; tighten them if downstream use needs it.

| Check | Phase 2 (reformulation) | Phases 3–4 (approximate) |
| --- | --- | --- |
| Total energy, \|ΔE\| | ≤ 1e-10 Eh | ≤ 1e-7 Eh |
| Gradient, max \|Δg\| | ≤ 1e-9 Eh/bohr | ≤ 1e-6 Eh/bohr |
| HOMO–LUMO gap | ≤ 1e-8 Eh | ≤ 1e-5 Eh |
| Atomic charges | ≤ 1e-8 e | ≤ 1e-5 e |
| SCC cycles | Unchanged | At most 1 more |
| Same SCF solution | Same occupations (to 1e-6) and gap as `reference/` | Same occupations (to 1e-3) and gap as `reference/` |
| Convergence | Every run that converges in `reference/` still converges | Same |

Run the gates at all three electronic temperatures and both uhf values in `reference/`, not just production's 300 K. The hard cases are open-shell systems with small gaps (`facet_dbpair_tool`, `facet_bare_dimer`, `facet_dbpair`, `s12_*`, the uhf 2/3 states) and the off-equilibrium `_rms15` copies. Approximate eigenvectors are most likely to fail there, and a change that moves a run onto a different SCF solution fails the gate even if its energy is lower.

Pitfalls to avoid:

- **Don't loosen SCC convergence thresholds** to buy speed; that hides the change being tested.
- **Don't patch any installed dylib in place.** Build from source under `build/`, keep the production `libxtb.dylib` (`../claude-molecular-renderer/`) and Homebrew's install as untouched references, and confirm at runtime which libxtb.dylib each process actually loaded. Replacing the production library is the user's call, after the gates pass from both Python and Swift.
- **Don't time the first run.** Use warm runs, median of at least 5, with the thread environment variables set. M1 Max laptops throttle, so check sustained runs.
- **Don't trust a single geometry.** Degenerate diamond cells, radicals and charged systems fail in different ways.
- **Keep final outputs in FP64,** even if intermediate cycles use FP32.
- **One commit per step** on a branch in `xtb/`, with the measurement in each message, plus the matching submodule-pointer commit here.

## Kickoff prompt for Claude Code

Start Claude Code in this repository's root on the M1 Max and paste:

```
Read CLAUDE.md, then BRIEF.md in full before doing anything. BRIEF.md is the plan
and the acceptance criteria.

Everything you need is here: the xTB fork as the xtb/ submodule (remote upstream =
grimme-lab/xtb), the test set (structures/), production reference results
(reference/) and the driver scripts/make_reference.py. swift-xtb
(philipturner/nanotech-code-archive) is a reference only.

Do Phase 1 only, then stop and report:
- build v6.7.1 (worktree) and main against Accelerate under build/, confirm with
  otool -L, and check the v6.7.1 build reproduces reference/ (energies, SCC
  iterations) via make_reference.py --lib ... --out build/...
- measure SCC cycles per point cold (production) vs warm (chained handle, fixed
  seed) on a scan; cold is the default, warm is a separate insight
- add the per-cycle SCC timers on a branch in xtb/ that applies to both trees
- record the Time Profiler breakdown and the concurrency sweep
- write baseline.md

Also run `git -C xtb log -S startpdiag` and `git -C xtb log -S pseudodiag` and
summarize why pseudodiagonalization was disabled.

Rules: OMP_NUM_THREADS=1 and VECLIB_MAXIMUM_THREADS=1 for every timed run;
median of 5 runs, timed only when the Mac is otherwise idle (other agents run
GFN2 jobs: check load, ask for a quiet window); never modify the Homebrew
install or ../claude-molecular-renderer (including its libxtb.dylib); never
loosen SCC convergence thresholds. Report measured numbers only, and say plainly
when something could not be measured.
```

After Phase 1, review `baseline.md` before letting it start Phase 2: if H1 or H2 fails, the later phases change.
