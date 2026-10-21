# xTB Diagonalization Profiling Brief

Oct 19, 2026 · @Philip Turner · corrected Oct 20, 2026 after Phase 1 (`baseline.md`)

## Goal and success criteria

Cut GFN2-xTB single-point wall time on Apple silicon (M1 Max) by at least 2x at n ≈ 410–450 basis functions, where most production work sits (up to n ≈ 650 for the Stage 3 regions), by attacking the per-SCC-cycle eigensolve. The speedup must hold for cold-start single points, which are the default in this project. Every claim of a speedup comes from a measurement, not an estimate.

Measure these, before and after each change:

- **Wall time per single point:** median of 5 runs (after one untimed warm-up run of the process; each single point itself still starts SCC cold) at n = 124, 410–450, 619 and 776 (the test set below).
- **Breakdown:** share of time in `dsygvd` and its children, and per-cycle time for H build, eigensolve, Fermi smearing, density matrix and Broyden mixing.
- **SCC cycle count** per system.
- **Throughput:** jobs per second at n ≈ 450 and n ≈ 750, comparing the current production setup (5–6 processes × 2 threads) against 8–10 processes × 1 thread.
- **Accuracy:** the gates in the Validation section. A change that fails a gate doesn't count, however fast it is.

What 2x takes (measured in Phase 1, `baseline.md`): the eigensolve is 56–69% of a single point at n ≈ 410–450, and
everything else is already close to half. If only the eigensolve gets faster, 2x needs it 72–78% cheaper on the
slowly converging production systems (`s12_C31`, `facet_dbpair_tool`) and 90% cheaper on fast-converging ones like
`facet_closed` (26 cycles). At n = 776 the eigensolve alone can't reach 2x: the work outside the SCC loop (integral
derivatives and gradient) is more than half. Fewer SCC cycles (Phase 3) and cheaper full solves (Phases 2 and 4) both
count; the outside-loop work is the next target after the eigensolve.

## What is already known

The eigensolve is the largest single cost, and its biggest piece is one BLAS2 routine. An older Instruments profile (`Experiments/OptimizingDiagonalization/Diamond233.png` and `DSYEVD.png`: n = 776, 19 SCC cycles, Homebrew `libxtb.6.dylib` linked to Accelerate, S refactored every cycle) put it at 75% of a single point. On the production code (upstream main) it is 48% at n = 776 and 56–69% at n = 412–448 (Phase 1, `baseline.md`, 1 thread):

| Routine (call path) | Old profile, n = 776 | Production code, n = 776 | Production code, n = 448 | Kind of work |
| --- | --- | --- | --- | --- |
| eigensolve (`DSYGVD` then; `mctc_dsygvd_factorized` now) | 74.6% | 48.2% | 64.5% | — |
| `DSYTRD` → `DLATRD` → `DSYMV` | 29.8% | 19.2% | 19.5% | BLAS2, memory-bound |
| `DSYTRD` → `DLATRD` → `DGEMV` | 5.8% | — | — | BLAS2 |
| `DSYTRD` → `DSYR2K` | 3.1% | 2.4% | 2.5% | BLAS3 |
| `DSTEDC` (divide and conquer) | 15.5% | 11.5% | 19.6% | Mostly scalar |
| `DORMTR` (back-transformation) | 9.6% | 6.4% | 8.9% | BLAS3 |
| `DSYGST` + `DTRSM` + `DPOTRF` | 8.8% | ≤ 12% (overlapping) | ≤ 14% (overlapping) | BLAS3 |
| `mmompop`, `setvsdq`, `dmat`, H build | \~15% | 10% | 14% | xTB's own code |
| integral derivatives + gradient (outside the SCC loop) | — | 35% | 18% | xTB's own code, OpenMP |

Inside the SCC loop the eigensolve is 79–80% of every cycle at n ≥ 412 (12–14 ms per cycle at n = 412–448, 35 ms at 619, 58 ms at 776).

Other facts already established:

- Switching OpenBLAS to Accelerate made one thread per job optimal (down from 8), and running many single-threaded jobs in parallel raised throughput about 20x.
- `DPOTRF` appears under `DSYGVD` in the old profile, so S was refactored every cycle in that build. Current xTB master no longer does this, and neither does production, which is built from master (see Code map). Measured saving of the cached factor against v6.7.1: 1–4% (`baseline.md`).
- `OptimizingDiagonalization.md` projects 1.86 s → 0.89 s at n = 776 if diagonalization cost only as much as divide and conquer, and → 0.55 s if the remaining LAPACK work and other kernels also moved to FP32. Neither was implemented.

Hypotheses this work should confirm or kill:

1. **H1:** `DSYMV` is limited by L2 bandwidth. Parallel jobs scale with cores until working sets overflow L2 (12 MB per 4-core P-cluster on M1 Max; an n = 776 FP64 matrix is 4.8 MB, n = 1,000 is 8 MB). *Phase 1: consistent, not proven.* Two processes scale perfectly; at 8 × 1 thread each job is 32% slower at n = 448 and 58% slower at n = 655. Each P-cluster also shares one AMX unit, so AMX contention isn't ruled out. 8 × 1 thread gives 15–23% more throughput than production's 6 × 2.
2. **H2:** After the first few SCC cycles, the Hamiltonian changes little, so most cycles don't need a full diagonalization. *Phase 1: not measured directly.* Even chained warm starts still need 18–37 cycles per point.
3. **H3:** The matrix units (AMX) are not the bottleneck, since only \~20% of the eigensolve is BLAS3. *Phase 1: consistent:* `dgemm` is 18–20% of samples; `dsymv` and the scalar `dlaed4` dominate.

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
- **Per-cycle cost is \~15–18 ms at n ≈ 410–450 for a process running alone** (Phase 1, whole cycle; the \~25–30 ms above were measured with 4 jobs at once), so fixed overheads, workspace handling and driver choice matter as much as asymptotic speed.
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

Release **v6.7.1** calls `solver%solve(env, H, S, emo)` at `scc_core.f90` line 440: a full `dsygvd`, including `dpotrf`, every cycle. That matches the old profile. The cached-Cholesky path exists only on master, which is what production runs (below). Phase 1 found two more v6.7.1 problems fixed on master: its GFN2 gradients are wrong on structures with atoms on symmetry axes (#1208, #1214; the error depends on the compiler), and its C API never closes the output file (#1434). Use v6.7.1 for timing comparisons only.

### Production entry point: the C API

Production uses ONE library: `../claude-molecular-renderer/libxtb.dylib`, at the root of the mechanosynthesis repository (a sibling of this one) (shipped with the molecular-renderer v1.0.2 binaries). It is **not** release 6.7.1: it was built from the fork's `swift-xtb-2026` branch at `8ea5fcd`, which is upstream main `68fe777` plus two commits that only stop GFN-FF from reading or writing files in the working directory, so its GFN2 code is identical to main (the version string still says 6.7.1). Build: CMake, Homebrew GCC 16, `-DBLA_VENDOR=Apple` (Accelerate, legacy LAPACK symbols), no tblite or CPCM-X, as in swift-xtb's `compile-macos.sh`; `scripts/build_xtb.sh` repeats it, and that build reproduces `reference/` bit for bit. Both that repository's Python wrapper `Sources/Workspace/Data/stage3/scripts/lib/xtb_lib.py` and the Swift code (swift-xtb `main`) load it; Homebrew's xtb is not used in production. A deliverable is a drop-in replacement for this file, checked from both Python and Swift. Python calls the xTB C API (`include/xtb.h`) directly through ctypes; no tblite is involved. `scripts/make_reference.py` is a self-contained copy of the production call sequence (plus charges, orbitals and the SCC iteration count) and is the benchmark driver to start from. The path below matches the top of the n = 776 profile, which starts at `xtb_singlepoint`.

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
| `baseline.md`, `LOG.md` | Phase 1 measurements; one log entry per session |
| `xtb/` | Git submodule: the fork github.com/philipturner/xtb, at `68fe777` (= grimme-lab/xtb `main`, 2026-10-08). Remote `upstream` = grimme-lab/xtb. Full history, so `git log -S` works |
| `structures/` | Test set (30 `.xyz`) and `MANIFEST.tsv` |
| `reference/` | Production-library results per structure × uhf × electronic temperature, and `SUMMARY.tsv` |
| `scripts/make_reference.py` | Benchmark/reference driver: production C API call sequence; `--lib` picks the library, `--out` the output folder |
| `scripts/build_xtb.sh` | Builds a tree the way production is built (out of source, under `build/`) |
| `scripts/compare.py` | Checks a run folder against `reference/` with the Validation gates |
| `scripts/bench.py`, `sweep.py`, `warmstart.py`, `callshare.py`, `latency.py` | Timed single points (with the per-cycle breakdown from a timer build), concurrency sweep, cold vs warm scans, Time Profiler aggregation, Accelerate-vs-OpenBLAS check |
| `scripts/provenance/make_testset.py` | How the structures were made (ran once in the mechanosynthesis repo; doesn't run here) |
| `build/` | Build trees and scratch, git-ignored |

Workflow for xTB changes:

- All xTB work goes on ONE branch of the fork, `diagonalization` (public; already created locally in `xtb/`, tracked by `.gitmodules`). One commit per step with its measurement in the message. Commit the new submodule pointer in this repository at the same time, so every result here names the exact xTB commit it came from. Short experiments may use side branches (`diag/<topic>`) that merge into `diagonalization`.
- The fork's `main` is never committed to: it only mirrors grimme-lab's `main`, so syncing is a fast-forward (`git -C xtb fetch upstream`, then the user runs `git -C xtb push origin upstream/main:main`). Bring upstream changes into the work with `git -C xtb merge upstream/main` on `diagonalization` (merge, not rebase, because the branch is public).
- Differences from upstream are plain git: `git -C xtb diff upstream/main...diagonalization`, `git -C xtb log upstream/main..diagonalization`.
- Pushing is the user's step: Claude commits to `diagonalization` and `diag/*` (and to this repository), and the user pushes; Claude's permission rules deny `git push`. The fork's `main` only ever gets a fast-forward to upstream, `diagonalization` is never force-pushed, and pull requests upstream need the user's go-ahead.
- Release v6.7.1, for timing comparisons only (production is main, see Code map): worktree `build/xtb-v6.7.1` (`git -C xtb worktree add ../build/xtb-v6.7.1 v6.7.1`); don't edit it. It needs mctc-lib `77f65c6` (its CMake fetches an unpinned `HEAD` that no longer configures) and `-std=gnu17` under GCC 16; see `baseline.md`. Side branch `diag/timers-v6.7.1` carries the SCC timers for it.
- Build trees go under `build/` (`build/xtb-build-main/`, `build/xtb-build-v6.7.1/`, …) via `scripts/build_xtb.sh`. CMake fetches mctc-lib with git at configure time; the sandbox blocks git from writing hook files, so configure runs outside it once, then builds run inside it, offline.
- The mechanosynthesis repository is read-only from here: never write into `../claude-molecular-renderer`, and never replace its `libxtb.dylib`. A faster library is handed over only after the gates pass from both Python and Swift, and the user decides.

## Phase 1: Profiling harness and baseline

The deliverable is a reproducible baseline that every later phase is measured against, built from source rather than taken from a prebuilt binary.

**Status: done 2026-10-20; results in `baseline.md`.** Corrections that came out of it: production is upstream main, not v6.7.1, so the main build (not v6.7.1) is the one that reproduces `reference/`, and does so bit for bit on all 174 runs. The concurrency sweep left out 10 concurrent processes by the user's decision: background work such as Photos' analysis can't be controlled in production, so 10 single-threaded jobs on 8 P + 2 E cores isn't worth recommending. The steps below are kept as they were run.

1. **Build two xTB trees on macOS arm64 against Accelerate:** v6.7.1 (worktree `build/xtb-v6.7.1/`, the version of the production library) and main (the `xtb/` submodule). `meson_options.txt` has no Accelerate choice; `-Dlapack=custom` with `custom_libraries` is the likely route, but check how Homebrew's formula links it first. Confirm with `otool -L` that the binary links Accelerate, not OpenBLAS, and cross-check at run time against `reference/LATENCY.md` (`scripts/latency.py`: production Accelerate vs Homebrew OpenBLAS, OpenBLAS 1.5–1.7x slower single-threaded at n ≥ 400). Check that your v6.7.1 build reproduces `reference/` (energies to 1e-10 Eh, same SCC iteration counts). If it doesn't, find out how the production `libxtb.dylib` was built before going further.
2. **Fix the environment for every run:** `OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, on AC power, Low Power Mode off. Record macOS version and chip. Time only when the Mac is otherwise idle: other Claude sessions run long GFN2 jobs here. Check `ps`/load first, and ask the user for a quiet window if needed.
3. **Use the prepared test set** (`structures/`, see "Production sizes and the test set") and `scripts/make_reference.py` as the driver. It loads the libxtb.dylib given by `--lib`, makes the same settings calls as production and creates one fresh results object per single point. It prints the resolved library path; also record `xtb_getAPIVersion`. Write every non-production run to `--out build/...`, never into `reference/`. For warm-start measurements, add a mode that reuses one results handle along a sequence (for example the `_rms05` copies, or a scan you generate), and keep it separate from the cold numbers. Keep the CLI only for quick cross-checks, `xtb mol.xyz --gfn 2 --sp`.
4. **Instrument the SCC loop** in `scc_core.f90`: wall-clock timers around H build, the eigensolve call, `fermismear`, `dmat` and `broyden`, printed per cycle with the cycle count. Keep this as a patch you can apply to both trees.
5. **Profile** each test system with `xctrace record --template 'Time Profiler'` and save the call-tree breakdown under `dsygvd`, so it can be compared against the table above.
6. **Run the concurrency sweep:** the production setup (5–6 processes × 2 threads) against 1-thread processes at 1, 2, 4, 8 and 10 concurrent, at n = 448 and n ≈ 700. Report jobs per second and how many land on E-cores; M1 Max has 8 P-cores and 2 E-cores, so 6 × 2 threads oversubscribes the P-cores. This tests H1.
7. **Write `baseline.md`:** versions, build flags, environment, median times, per-cycle breakdown, cycle counts, final energies and gradients printed to 1e-10 Eh. The v6.7.1-versus-master difference measures the cached-Cholesky change for free.

## Phase 2: Low-risk changes in the xTB tree

These are reformulations: exact in exact arithmetic, but they change rounding, and the SCC carries rounding differences up to the size of its own convergence noise. Phase 1 measured this on an upstream reformulation (v6.7.1's full `dsygvd` against main's cached factor): energies moved by up to 7.9e-8 Eh, 124 of 174 runs by more than 1e-10. So they are gated like every phase, by the two gate sets in Validation (method error at accuracy 0.01, same solution at production accuracy), with a tighter cycle-count rule. Expect a few percent, not a multiple; their main value is a clean place to hook Phase 3.

0. **Noise floor on the whole grid.** *Done 2026-10-20 (`scripts/noise.py`, `build/noise/acc1/` and `acc0.01/`, tight
   production results in `build/reference-tight/`).* Every `reference/` run was restarted from 2–3 converged neighbours
   (neighbouring electronic temperatures, nearest `_rms05` copy) at accuracy 1.0 and 0.01. The production-accuracy
   noise exceeded the first gates (gradient 7.7e-5 > 6e-5 Eh/bohr; charges 1.2e-3 > 1e-3 e), which led to the
   two-gate protocol in Validation: method error at accuracy 0.01, same solution at accuracy 1.0.
1. **Cached Cholesky.** Already in production (master factors S once per geometry). Measured saving against v6.7.1: 1–4% at n ≤ 619 (`baseline.md`). Nothing left to do.
2. **GEMM-based transform.** Replace `dsygst` + `dtrsm` in `mctc_dsygvd_factorized` with a matrix X formed once per geometry, X = U⁻¹ from the Cholesky factor. Each cycle becomes H′ = Xᵀ H X (two GEMMs, or `dsymm` + `dgemm`), `dsyevd` on H′, then C = X C′ (one GEMM). These moves \~7% of runtime onto matrix-unit-friendly GEMMs. Measure; don't assume it's faster at small n.
3. **One hook point.** Route every eigensolve through a single new subroutine that receives the cycle number, the previous eigenvectors and eigenvalues, and the factor X. Phase 3 changes only this routine.
4. The three under-review PRs by https://github.com/g8j39 on the grimme-lab/xtb repository: 1464, 1465, 1466. Small incremental wins worth investigating.

At n ≈ 450–750, also compare drivers on the same transformed H′: `dsyevd`, `dsyevr` for all eigenvectors, and `dsyevr` for only the lowest eigenvectors needed (occupied plus a margin of virtuals for Fermi smearing). Tridiagonalization still dominates all three, so expect differences of tens of percent at most, but at these sizes the cheapest driver may differ from the n = 776 result. Check that workspace is allocated once per geometry, not per call.

Commit each step separately, with its measurement in the commit message.

## Phase 3: Eigenvector reuse across SCC cycles

This is the largest expected win: with 19 cycles at n = 776, most full diagonalizations repeat work on a Hamiltonian that barely changed. Try the options in this order and keep the cheapest one that passes every gate.

Before touching the eigensolver, measure warm starts as a side experiment (*done in Phase 1, `baseline.md`: over 21-point scans, chained starts need 35–55% fewer SCC cycles, a fixed seed 12–33% fewer, and every point reaches the cold SCF solution*). Run a 20-step scan through the C API three ways:

1. a fresh `xtb_newResults` per point (production);
2. one handle reused with `xtb_updateMolecule` (chained);
3. one fixed seed for every point, like `xTB_Seed`.

Compare SCC cycles per point, final energies and whether every point reaches the same SCF solution as its cold start. This only says how much warm starts could save where they are allowed (roughly half of practical uses). Every option below must also deliver its speedup on cold starts, which stay the default.

1. **Find out why pseudodiag was disabled.** *Done in Phase 1: no recorded reason.* `startpdiag=1000 !large number=never` is already in the 6.2 import (`29ce1a1`, 2019-09-30), the first commit of the public history, and since #192 (2020-05) the legacy `solve` that calls `pseudodiag` is commented out, so it is dead code. Nothing recorded changes the plan.
2. **Option A: re-enable the existing path.** Wire the legacy `solve` logic into the Phase 2 hook: full solve for the first k cycles (try k = 3, 5, 8), then rotate H into the previous MO basis, call `pseudodiag`, and rotate back. Because the previous C satisfies Cᵀ S C = I, Cᵀ H C works directly in the generalized problem. Always do a full solve on the final cycle, so printed orbitals and gradients come from exact eigenvectors.
3. **Option B: GEMM-only refinement.** If A fails the gates, replace pseudodiag with an Ogita–Aishima-style refinement in the previous basis (a few GEMMs per cycle, quadratically convergent). Near-degenerate orbitals, which the symmetric diamond cells have, need the clustered-eigenvalue variant. Fall back to a full solve whenever the residual ‖H′C − CΛ‖ exceeds a threshold.
4. **Option C: mixed precision.** Independently of A or B, test FP32 `ssyevd` for early cycles and FP64 for the last few (`solve4` is the precedent).

Watch two xTB-specific risks:

- **Fermi smearing** needs accurate eigenvalues near the Fermi level, but pseudodiag only rotates occupied–virtual pairs, so its eigenvalues are approximate. Check radical and small-gap systems especially.
- **Broyden mixing** may converge differently with approximate eigenvectors. Report cycle counts, not just time per cycle.

Report per option: time per cycle, total cycles, total time and every gate's result.

## Phase 4: Successive band reduction (revised 2026-10-21)

The first version of this phase (benchmark Accelerate's two-stage routines, port the Swift stages) is done as far as
Accelerate goes: `dsytrd_2stage` doesn't pay at n ≤ 448 and gives no eigenvectors; `ssyevd_2stage` is accurate
(`FINDINGS.md`). The user's direction for the rest of the phase:

- **Successive band reduction (SBR)** generalizes two-stage tridiagonalization. 1-stage (best below n ≈ 100): dense →
  tridiagonal. 2-stage (best at n ≈ 100–800): dense → band b → tridiagonal; the best b moved with n in swift-xtb
  (8, 16, 32, larger at larger n). 3-stage (expected best above n ≈ 800, perhaps a general answer to the block-size
  problem): dense → b1 (large, outer) → b2 (small, inner) → tridiagonal, at the cost of one more back-transformation.
- **One code base:** 3-stage reduces to 2-stage at no runtime cost by a setting in a configuration descriptor (from C or
  Fortran, one exported symbol per variant if needed). 1-stage may stand alone; ideally it is what 2-stage does inside a
  single b × b block.
- **Don't exploit symmetry** during development (it halves the work but is unrelated to the stage count and adds
  complexity). The fair baseline is our 1-stage reduction without symmetry (`tridiagonalize(..., symmetric: false)`).
- **Watch multi-process throughput.** The extra back-transformation work is many small GEMMs (8×8×32, 8×32×32) on the
  AMX; with 8 processes they may contend and lose throughput even when one process gets faster. Single-thread latency
  still matters for work that can't run as parallel SCF invocations.
- **Pipelining** (bulge chasing starting halfway through the band reduction, D&C during the last sweeps) is a later
  optimization: it helps single-run latency, not parallel throughput.

Reference: the draft in `philipturner/nanotech-code-archive` `swift-xtb/Sources/LinearAlgebra` (cloned read-only into
`build/nanotech-code-archive`): FP32, row-major; band reduction with one-sided panel factorization + compact WY
(T built in small blocks of `smallBlockSize`); bulge chasing one reflector per bulge with `sgemm_`/`sger_` calls on
windows of ~3b × b; back-transformation of the bulge reflectors grouped b sweeps at a time into 2b × b WY blocks; band
back-transformation one GEMM chain per panel. Inefficiencies to avoid: every panel updates the full n × n matrix
(left and right) instead of the shrinking trailing block (about 3× the flops, before symmetry); the band
back-transformation also spans all n rows; per-call array allocations; BLAS call overhead on the ≤ 32-wide bulge
kernels (NEON beats Accelerate below 8 × 8, `DenseDiagonalizationExperiment/README.md`).

Design for this repository (`Sources/Diagonalization`, generic FP64/FP32, column-major, FP64 first):
- **A stage is a band reduction b_in → b_out.** The configuration is the list of intermediate bandwidths: `[]` is
  1-stage (the `sytrd` path), `[b]` 2-stage, `[b1, b2]` 3-stage; the driver loops over stages, so 2-stage is 3-stage
  with one stage fewer. Dense → band: panel QR (valid one-sided while the panel width ≤ b_out) plus a two-sided
  trailing update by GEMM on the trailing block only. Band → band, b_out ≥ 1: Bischof–Lang–Sun bulge chasing; b_out = 1
  with one-column panels is the classical bulge chase of the draft.
- **Storage:** the full n × n matrix during development (no band storage, no symmetry).
- **Back-transformation** per stage in reverse order, reflectors grouped into WY blocks. Only the eigenvectors xTB uses
  need back-transforming (#1466 back-transforms the occupied block only), so its cost scales with that column count.
- **D&C:** Accelerate's `dstedc`, unchanged.

Steps:
1. **Baseline (done):** `tridiagonalize(symmetric: false)`, gemv/gemm/ger on the full matrix. FP64: 1.39–1.76× the
   time of the symmetric Swift reduction at n = 412–776 (n = 448: 8.16 vs 4.64 ms; Accelerate `dsytrd` 6.63 ms).
   FP32: 0.95–1.10× (Accelerate's full `sgemv`/`sgemm` are as fast as the symmetric kernels on half the data).
2. **End-to-end harness:** reduce → `dstedc` → back-transform m eigenvectors on the dumped GFN2 matrices, per-stage
   times (the draft's breakdown); baselines: unsymmetrized and symmetric 1-stage + `dormtr`. Correctness: eigenvalues
   against `dsyevd`, residual ‖H Z − Z Λ‖, orthogonality ‖Zᵀ Z − I‖.
3. **2-stage:** dense → band (trailing block only) and band → tridiagonal, both back-transformations; sweep b = 8–64.
4. **3-stage:** general band → band stage, configuration `[b1, b2]`; sweep (b1, b2).
5. **Throughput:** jobs per second with 8 processes × 1 thread against the 1-stage reduction.
6. **Later:** symmetry, band storage, NEON kernels for the small bulge blocks, pipelining.
7. **Into xTB:** a C entry point for the whole eigensolve (the current `XTB_SYTRD_LIBRARIES` hook only replaces the
   reduction), then both gate sets.

## Validation and pitfalls

A speedup counts only if it passes every gate on every system in the test set. Adopted 2026-10-20 after the
whole-grid noise measurement (Phase 2 step 0, `baseline.md`).

**Why two gate sets.** A converged GFN2 result moves when only its starting wavefunction changes: that is the SCC's own
convergence noise, and any change to the SCC trajectory (an exact reformulation that only changes rounding, a warm
start, an approximate eigensolver) moves results by that much. At production accuracy (1.0) it reaches 7.7e-5 Eh/bohr
(6.3 pN) in gradients, so no gradient gate at the pN level can be reliable there. At accuracy 0.01 it is about 100x
smaller everywhere (gradients ≤ 7.2e-7 Eh/bohr, 0.06 pN). So the method's own error is measured at accuracy 0.01, where
the noise is small, and production accuracy only has to show that results stay on the same SCF solution and converge
as before. Accuracy 0.01 is a validation setting only; production keeps 1.0 and its thresholds are never changed.

Measured noise floor (production library; every run of the grid restarted from 2–3 converged neighbours: the
neighbouring electronic temperatures and the nearest `_rms05` copy; maxima over starts that reach the same solution):

| Quantity | Accuracy 1.0: max (95th percentile) | Accuracy 0.01: max (95th percentile) |
| --- | --- | --- |
| Total energy, \|ΔE\| | 3.1e-7 (1.3e-7) Eh | 2.7e-11 (1.3e-11) Eh |
| Gradient, max \|Δg\| | 7.7e-5 (4.4e-5) Eh/bohr = 6.3 (3.6) pN | 7.2e-7 (3.6e-7) Eh/bohr = 0.06 (0.03) pN |
| HOMO–LUMO gap | 2.8e-5 (9.7e-6) Eh | 3.6e-7 (8.1e-8) Eh |
| Atomic charges, max \|Δq\| | 1.2e-3 (1.1e-4) e | 4.9e-6 (9.2e-7) e |
| Occupations, max \|Δf\| | 1.4e-3 (8.7e-4) | 1.5e-5 (8.8e-6) |

**Gate 1, method error (tight accuracy).** Run the candidate with `make_reference.py --accuracy 0.01` and compare with
the production library's results at the same accuracy (`build/reference-tight/`): `compare.py --gates tight`.

| Quantity | Gate | Gate / max noise |
| --- | --- | --- |
| Total energy | ≤ 1e-10 Eh | 3.7x |
| Gradient | ≤ 2e-6 Eh/bohr (0.16 pN) | 2.8x |
| HOMO–LUMO gap | ≤ 1e-6 Eh | 2.8x |
| Atomic charges | ≤ 1e-5 e | 2.0x |
| Occupations | ≤ 5e-5 | 3.3x |

SCC cycles are reported here, not gated (accuracy 0.01 needs 32% more cycles than 1.0 with the production library).

**Gate 2, same solution and convergence (production accuracy).** Run the candidate exactly as production
(`make_reference.py`, accuracy 1.0) and compare with `reference/`: `compare.py --gates phase2` or `--gates approx`.

| Check | Phase 2 (reformulation) | Phases 3–4 (approximate) |
| --- | --- | --- |
| Total energy | ≤ 5e-7 Eh | same |
| Gradient | ≤ 1e-4 Eh/bohr (8.2 pN) | same |
| HOMO–LUMO gap | ≤ 5e-5 Eh | same |
| Atomic charges | ≤ 2e-3 e | same |
| Occupations | ≤ 5e-3 | same |
| SCC cycles per run | At most 2 more than `reference/` | At most 10% (and at least 2) more |
| SCC cycles, total over the grid | Within ±1% of `reference/` | Reported; the wall-time gain is what counts |
| Convergence | Every run that converges in `reference/` still converges | Same |

These sit at 1.3–3.6x the accuracy-1.0 noise maxima. Different SCF solutions differ by far more (occupations by O(0.1–1),
gaps by 0.01–1 eV), so this gate still catches a move to another solution.

**Multi-solution runs.** `facet_closed_rms15` uhf 2 at 300 K has two SCF solutions 1.3e-6 Eh apart whose gradients differ
by 2e-3 Eh/bohr (165 pN) and charges by 2.6e-2 e, at both accuracies: perturbed starts land on the other one. It is
gated only on reaching the reference's solution (max |Δf| < 0.05, |Δgap| < 1e-3 Eh, |ΔE| < 1e-5 Eh, max |Δq| < 5e-3 e),
in both gate sets. Add any run that the noise measurement finds in the same state.

**Warm-start fragility, for Phase 3.** Two runs converge cold but not from a converged neighbour: `facet_bare_dimer`
uhf 0 at 300 K started from its 1000 K result (both accuracies), and `s12_C00` uhf 3 at 100 K from its 300 K result
(accuracy 0.01). Methods that reuse earlier wavefunctions or eigenvectors are most likely to fail there.

`scripts/compare.py` implements both gate sets (`--gates phase2 | approx | tight`, plus `identical` for bit-for-bit
checks); `scripts/noise.py` measures the noise floor.

Run the gates at all three electronic temperatures and both uhf values in `reference/`, not just production's 300 K. Runs that do not converge in `reference/` (three hit the 250-cycle cap) are checked only for convergence status. The hard cases are open-shell systems with small gaps (`facet_dbpair_tool`, `facet_bare_dimer`, `facet_dbpair`, `s12_*`, the uhf 2/3 states) and the off-equilibrium `_rms15` copies. Approximate eigenvectors are most likely to fail there, and a change that moves a run onto a different SCF solution fails the gate even if its energy is lower.

Pitfalls to avoid:

- **Don't loosen SCC convergence thresholds** to buy speed; that hides the change being tested.
- **Don't patch any installed dylib in place.** Build from source under `build/`, keep the production `libxtb.dylib` (`../claude-molecular-renderer/`) and Homebrew's install as untouched references, and confirm at runtime which libxtb.dylib each process actually loaded. Replacing the production library is the user's call, after the gates pass from both Python and Swift.
- **Don't time the first run.** Use warm runs, median of at least 5, with the thread environment variables set. M1 Max laptops throttle, so check sustained runs.
- **Don't trust a single geometry.** Degenerate diamond cells, radicals and charged systems fail in different ways.
- **Keep final outputs in FP64,** even if intermediate cycles use FP32.
- **One commit per step** on a branch in `xtb/`, with the measurement in each message, plus the matching submodule-pointer commit here.

## Kickoff prompt for Claude Code

Phase 1 is done (2026-10-20). This is the prompt it was run with; later sessions start from `LOG.md` and `baseline.md` instead. Note that its v6.7.1 reproduction step was wrong: production is main, and the main build is the one that reproduces `reference/`.

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
