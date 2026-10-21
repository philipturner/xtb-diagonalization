# Latency check: is a build linked to Accelerate or OpenBLAS?

`otool -L libxtb.dylib` is the primary check: it must list `/System/Library/Frameworks/Accelerate.framework` and no
`libopenblas`. These timings are the runtime cross-check. Measured 2026-10-20 on the M1 Max with
`scripts/latency.py LIB OUT.json`:
- one process, `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1`;
- cold single points (fresh results object each), production settings (accuracy 1.0, 300 K, maxiter 250);
- one untimed warm-up, then the median of 5.

The Mac had a load average of about 1.8 from other work, so treat the absolute numbers as indicative to ±5%.
Run-to-run spread within a set of 5 was ≤ 2%.

| Structure | uhf | n | SCC iterations | Accelerate (production `libxtb.dylib`, 6.7.1) | OpenBLAS (Homebrew `libxtb.6.dylib`) | OpenBLAS / Accelerate |
| --- | --- | --- | --- | --- | --- | --- |
| `t2_R` | 1 | 124 | 22 | 0.045 s | 0.057 s | 1.26 |
| `facet_closed` | 0 | 412 | 26 | 0.565 s | 0.854 s | 1.51 |
| `facet_dbpair_tool` | 1 | 448 | 49 | 1.077 s | 1.774 s | 1.65 |
| `s3_S42_159` | 1 | 619 | 42 | 2.198 s | 3.553 s | 1.62 |
| `diamond233` | 0 | 776 | 19 | 2.342 s | 3.906 s | 1.67 |

How to read it:
- **Energies:** both libraries agree to ≤ 1e-8 Eh on all five.
- **Accelerate per SCC cycle:** about 2 ms at n = 124, 22 ms at n = 412–448, 52 ms at n = 619, 123 ms at n = 776.
- **A correctly linked Accelerate build** of the same xTB version should land within ~5–10% of the Accelerate column.
- **OpenBLAS** is 1.5–1.7x slower single-threaded at n ≥ 400. A build that matches the OpenBLAS column, or is 1.4x or
  more slower than Accelerate at n ≥ 400, is probably linked to OpenBLAS (or reference BLAS/LAPACK).
- **n = 124 doesn't discriminate** (1.26x): use the n ≥ 400 rows.
- **Thread check:** with multiple threads allowed, OpenBLAS responds to `OPENBLAS_NUM_THREADS` and Accelerate to
  `VECLIB_MAXIMUM_THREADS`. Comparing a 1-thread and a 4-thread run under each variable is a second runtime check.
- **What the process loaded:** `DYLD_PRINT_LIBRARIES=1` shows which BLAS dylibs were actually loaded at run time.

Raw results: `reference/latency/*.json`.
