# xtb-diagonalization

Faster GFN2-xTB single points on Apple silicon, from faster per-SCC-cycle eigensolves. The xTB code is on the
`diagonalization` branch of [philipturner/xtb](https://github.com/philipturner/xtb) (the `xtb/` submodule). This
repository has the test set, the reference results, the benchmark and validation scripts, and Swift prototypes of the
eigensolver.

```
git clone --recurse-submodules https://github.com/philipturner/xtb-diagonalization
git -C xtb remote add upstream https://github.com/grimme-lab/xtb
```

## Results (M1 Max)
Measured against the production library (xtb 6.7.1 + Accelerate). FP32 means FP32 eigensolves in the early SCC
cycles, then FP64 to convergence (`xtb_setSCCPrecision`, rmsq 1e-3).

| Setup | FP64 | FP32 |
|---|---|---|
| Throughput: 8 processes × 1 thread, n = 412–1204 | 1.76–1.93× | 2.20–2.56× |
| Latency: one SCF, 8 OpenMP threads (vs production on 1 thread), n = 412–1788 | 2.6–3.8× | 2.9–4.9× |

Every accuracy gate passes at 1 thread (bit-identical to the previous step) and in FP64 at 4 threads. In FP32 at 4
threads, the SCC cycle-count gate fails on a few s12_C31 runs (+10–23% cycles), as it does for any change in FP32
rounding; energies, gradients and charges stay within the gates. Details, with every measurement: `LOG.md`.

What changed in xTB (`src/mctc/lapack/sytrd.c`, C with OpenMP):
- **Serial kernels:** the reduction to tridiagonal form with NEON kernels (symmetric matrix-vector product, fused panel
  products), the back-transformation by gemm, and FP32 versions of both.
- **On the caller's OpenMP threads:** the reduction (one team of at most one cluster of the fastest cores), divide and
  conquer (all threads), and the back-transformation (two threads).

The parallel reduction and the NEON kernels are arm64 only. Elsewhere, the reduction uses BLAS calls, while the
parallel divide and conquer and back-transformation still apply.
swift-xtb's `fp32-diagonalization` branch builds this xTB on macOS.

## Layout
- `structures/`: 33 test geometries (n = 124–1788 basis functions: closed and open shell, small gap,
  off-equilibrium, and a large-n hexagonal SiC family), listed in `structures/MANIFEST.tsv`. `scripts/provenance/`
  shows how they were made.
- `reference/`: GFN2 results from the production library for every structure × the two lowest spin states ×
  electronic temperature 100 / 300 / 1000 K. Each run records the energy, gradient, charges, orbital eigenvalues and
  occupations, HOMO–LUMO gap and SCC iteration count. `reference/SUMMARY.tsv` has one line per run. Never overwrite
  it: results from other libraries go to `--out build/...`.
- `scripts/make_reference.py`: runs any libxtb through the xTB C API with the production call sequence (cold SCF,
  accuracy 1.0, `--lib`, `--out`, `--threads`, `--fp32`). It has no `--help`, and unknown arguments start a
  production run.
- `scripts/compare.py`: the accuracy gates (`identical`, `phase2`, `approx`, `tight`).
- `scripts/bench.py`, `sweep.py`, `bench_cases.sh`: latency and throughput.
- `scripts/kernels/`: C studies of single eigensolver steps.
- `scripts/build_xtb.sh`: builds libxtb as production does (CMake, GCC 16, Accelerate).
- `Package.swift`, `Sources/`: the Swift prototypes (`Diagonalization`), covering the 1-stage and 2-stage
  eigensolvers, parallel versions, and `eigen-bench` / `sytrd-bench`. `scripts/build_xtb_swift.sh` links them into
  libxtb in place of `sytrd.c`. Inside Claude Code, `swift build` needs `--disable-sandbox`.
- `BRIEF.md`: the plan and acceptance criteria. `FINDINGS.md`: answers to the brief's questions. `baseline.md`: the
  first measurements. `LOG.md`: the session log.

## Reproducing the reference energies
`reference/` comes from xTB's C API, which takes coordinates in bohr and returns Hartree.
`scripts/make_reference.py` converts the `.xyz` files from Å with 0.52917721067 Å/bohr (CODATA 2014). Use the same
value to reproduce the energies to the last digit.

swift-xtb (on `main` and its branches) uses its own constants, which differ from CODATA in the 7th digit:

| Constant | swift-xtb | CODATA 2018 | Relative difference |
|---|---|---|---|
| Hartree (`xTB_ZJPerHartree`) | 4359.7482 zJ | 4359.7447 zJ | +8.0e-7 |
| Bohr radius (`xTB_NmPerBohr`) | 0.0529177 nm | 0.0529177211 nm | −4.0e-7 |

The two differences have different effects:
- **The energy unit:** divide swift-xtb energies by 4359.7482, not by the CODATA value, to get back xTB's Hartree.
  Otherwise there's a 0.8 ppm offset, 3.3e-4 Eh at −412 Eh.
- **The bohr radius changes the geometry itself:** a structure given in nm reaches xTB scaled by 4.0e-7, and the
  energy moves with it. Measured with the production library, cold SCF, 300 K:

  | Structure | Energy change |
  |---|---|
  | facet_closed | −6.3e-7 Eh |
  | s3_S42_159 | +1.7e-7 Eh |
  | diamond233 | +1.6e-6 Eh |
  | t2_R | −5.2e-8 Eh |

  That is larger than the accuracy gates here (5e-7 Eh). Compare swift-xtb against `reference/` only with positions
  converted using 0.052917721067 nm/bohr.
