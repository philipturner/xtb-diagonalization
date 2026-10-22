# Handoff: test the `diagonalization` branch of xTB on Windows (Intel, 2014)

For a Claude session on the user's Windows PC: a 4-core Intel desktop CPU from 2014 (Haswell generation:
AVX2/FMA3, no AVX-512; check the exact model and whether it has Hyper-Threading). Goal: **build, validate and time** the
`diagonalization` branch of the user's xTB fork against the branch the user ships on Windows today, and report measured
numbers. Written 2026-10-22 on the M1 Max where the work was done (repo `xtb-diagonalization`, notes in `LOG.md`).

## Rules (from the user; they apply on this machine too)
- **Ask before anything outward-facing.** Never push (the user pushes), never force-push, no pull requests, and don't
  commit to the fork's `main`. If you need to change xTB source, ask first. Changes then go on `diagonalization`, one
  commit per step with its measurement in the message, ending with
  `Co-Authored-By: Claude <noreply@anthropic.com>` (use your own model name).
- **Report measured numbers only.** Say plainly what you could not measure. Never loosen SCC thresholds. Cold SCF
  (a fresh results object per single point) is the default.
- **Don't add unneeded files.** No `.claude/` folders, no stray scripts in the repos. Put scratch and builds in a
  git-ignored `build/` or a temp folder.
- **Timed runs only on an idle machine.** Use the median of 5, and set thread variables at process launch
  (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`), never by changing the environment mid-run. libgomp reads
  `OMP_NUM_THREADS` when the library loads.
- **Delete stray `.sccnotconverged` files.** xtb writes one into the working directory when an SCF fails.

## What the branch changes, and what of it applies on x86-64
`diagonalization` (github.com/philipturner/xtb) is upstream `main` plus the per-SCC-cycle eigensolve work. The core is
`src/mctc/lapack/sytrd.c` (C, OpenMP), called from `src/mctc/lapack/eigensolve.F90`. Other files carry the FP32 SCC
precision API (`xtb_setSCCPrecision`) and timers.

| Piece | Apple silicon | x86-64 Windows |
|---|---|---|
| Tridiagonal reduction: blocked dsytrd port | yes | **yes**: the symv is OpenBLAS `dsymv`, the panel products are `gemv` |
| NEON symv (8 columns per pass), fused panel products | yes | no (`__aarch64__` only) |
| **Reduction on a team of OpenMP threads** (newest, `c04600e`) | yes | **no** (`__aarch64__` only: it is built on the NEON kernels) |
| Divide and conquer on the caller's OpenMP threads (`xtb_dstedc`/`xtb_sstedc`, OpenMP tasks) | yes | **yes** (if built with OpenMP and n ≥ 256) |
| Back-transformation by gemm (`xtb_dormtr_upper`), two column blocks on two threads | yes | **yes** |
| FP32 SCC eigensolves (`xtb_setSCCPrecision`) | yes | **yes** (`ssytrd`-style port, OpenBLAS `ssymv`) |

So on this PC, expect gains from the divide and conquer and the back-transformation (with `OMP_NUM_THREADS` > 1) and
from FP32 mode, but **not** from the team reduction. One thread runs the serial path. `sytrd.c` was only
syntax-checked for x86-64 (clang, with and without OpenMP: clean) and has **never been compiled or run on Windows,
MinGW or OpenBLAS**. Finding out whether it does is the first job.

Things that might break on MinGW + OpenBLAS (none tested):
- **Linking.** The parallel divide and conquer calls LAPACK auxiliaries directly: `dlaed2/3/4`, `dlamrg`, `dlacpy`
  and `dlaset` (and the `s` versions), plus `dgemm`, `dsteqr` and `dstedc` (see `STEDC_PARALLEL` and `XTB_TS` in
  sytrd.c). On non-Apple systems they are the plain `name_` symbols. OpenBLAS normally bundles reference LAPACK,
  including these; a link error would show it doesn't.
- **OpenBLAS threading inside OpenMP tasks.** OpenBLAS is called from several OpenMP threads at once. If MSYS2's
  OpenBLAS uses its own thread pool, that can oversubscribe the 4 cores (or, in old versions, be unsafe). Time with
  `OPENBLAS_NUM_THREADS=1`, and check what the MSYS2 package was built with.
- **Workspace sizes.** LAPACK's `xlaed2` writes past its documented workspace size. AddressSanitizer found this on
  the Mac (intermittent crash, FP32 n = 1788), and sytrd.c already sizes Q2 as n². Any crash in the divide and
  conquer is worth reporting with the case and thread count.

## How the user builds xTB on Windows today
In swift-xtb: `compile-windows-msys2.sh` (https://github.com/philipturner/swift-xtb), run with Git Bash. It:
- installs MSYS2 base (`msys2-base-x86_64-20250622.sfx.exe`) into `.build/msys64`;
- installs the pacman packages `mingw-w64-x86_64-gcc-fortran`, `-cmake`, `-ninja` and `-openblas`;
- clones `--single-branch --branch swift-xtb-2026 https://github.com/philipturner/xtb`;
- configures with `PATH="$MINGW64_DIR/bin:$PATH" cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc
  -DCMAKE_Fortran_COMPILER=gfortran -DWITH_TBLITE=OFF -DWITH_CPCMX=OFF -DMCTCLIB_FIND_METHOD=fetch`, then runs
  `ninja -C build -j4` and `ninja -C build test`;
- copies `libxtb.dll` (as `xtb.dll` for Swift) plus `libgcc_s_seh-1.dll`, `libgfortran-5.dll`, `libgomp-1.dll`,
  `libopenblas.dll`, `libquadmath-0.dll` and `libwinpthread-1.dll` from `msys64/mingw64/bin`.

`test.bat` runs `test.sh` through `C:\Program Files\Git\bin\bash.exe`. The renderer downloads a prebuilt
`xtb-windows.zip` (the same DLL set) from molecular-renderer-simulator-binaries.

## Plan
1. **Two builds, same toolchain:**
   - Baseline: `swift-xtb-2026`, exactly as the script does.
   - Test: `diagonalization`, same CMake line, separate folder.

   CMake must find OpenMP for both Fortran and C (`xtb/CMakeLists.txt` links `OpenMP::OpenMP_C` for sytrd.c). Check
   the configure output. Without it, sytrd.c builds serial, silently. First compare the two branches
   (`git log --oneline swift-xtb-2026..diagonalization` and the reverse). If `swift-xtb-2026` carries Windows or API
   fixes that `diagonalization` lacks, report that rather than merging on your own.
2. **Smoke test:** `xtb.exe --version`, `ninja -C build test`, and swift-xtb's tests against the new DLL set. If
   swift-xtb tests fail without explanation, check which DLLs sit in the folder first. Last time, an unexplained
   Windows failure went away with a fresh clone, not with deleting `.build`.
3. **Accuracy:** single points with both DLLs on the same structures, FP64 and FP32. FP32 means
   `xtb_setSCCPrecision`, with rmsq 1e-3 as on the Mac. Compare energies, gradients, charges, HOMO-LUMO gap and SCC
   cycle counts.
   - **FP64:** expect agreement to about 1e-9 Eh or better, and the same cycle counts.
   - **FP32:** expect at most about +10% cycles per run. The s12_C31 structures are known to exceed that in FP32 on
     any rounding change; the user accepts that, but report it.

   Run with 1 and with 4 OpenMP threads. The Mac tools for this (`scripts/make_reference.py`, `compare.py`,
   `bench.py`, `sweep.py`, `structures/`) are in the `xtb-diagonalization` repo, if the user can give you a copy.
   They load the library through Python ctypes, and `--lib PATH` should accept a DLL (untested on Windows).
   - **Always pass `--out build/...`:** `make_reference.py` has no `--help`, and unknown arguments start a run that
     writes to `reference/` with its Mac default library path.
   - **Never overwrite `reference/`.** It holds the Mac production results, and Windows (OpenBLAS) results will differ
     from them in rounding, so compare Windows-to-Windows with `compare.py DIR --gates phase2 --ref BASELINEDIR`.
4. **Timing**, idle machine, median of 5, cold single points, sizes in the n ≈ 410–650 range (the user's target),
   plus one larger case if feasible:
   - **Latency:** one process, `OMP_NUM_THREADS=1` and `=4` (and `=8` if it has Hyper-Threading).
   - **Throughput:** 4 processes × 1 thread, jobs/s.

   Run each with `OPENBLAS_NUM_THREADS=1` first, then OpenBLAS's default, and report both. If possible, also report
   where the time goes per SCC cycle (eigensolve vs the rest). The Mac branch has timer builds (`scc_timer.f90`); see
   what the Windows build supports.
5. **Report** in one place: the CPU model, the build and gate results, and tables of baseline vs `diagonalization`,
   FP64 and FP32, per thread count. List what you could not run, and why.

## Mac numbers for orientation (M1 Max, Accelerate; not comparable in absolute terms)
- **Throughput:** 8 processes × 1 thread vs production FP64 (xtb 6.7.1 + Accelerate): FP64 1.76–1.93×, FP32
  2.20–2.56× (n = 412–1204).
- **Latency:** one SCF, 8 OpenMP threads, newest build, vs production with 1 thread: FP64 n = 412 0.205 s (2.76×),
  619 0.781 s (2.78×), 776 0.614 s (3.82×). FP32 is 3.0–4.9×.
- **Where the time went** (1 thread, n = 412 FP64 eigensolve ≈ 7.7 ms): reduction 3.0 ms, divide and conquer 3.8 ms,
  back-transformation 0.9 ms.
- **What threads did:** with 8 threads, the divide and conquer ran 2.2–2.8× faster and the back-transformation
  1.25–1.9×. The team reduction ran 1.9–2.8× faster in FP64 at 4 threads (Apple only, as above).

On x86 the reduction is OpenBLAS's `dsymv`, which may be much slower or faster than the NEON kernel. Before proposing
an x86 port of the team reduction (an AVX2 symv plus the same team structure), measure how large the reduction's
share is on this CPU.
