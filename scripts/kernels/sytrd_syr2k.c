// Does the symmetric 1-stage reduction (sytrd.c) pay for Accelerate's dsyr2k twice: once for the update itself and
// again in the next panel's L2-bound symv, which may run in the slow state dsyr2k leaves behind (LOG.md: after
// dsyr2k / dsymm, L2-bound code runs 15-30% slower; after dgemm it doesn't; lower_update.c)? sytrd.c's dsyr2k ('U',
// rank 2 x 32) three ways:
//   0  dsyr2k (sytrd.c as is)
//   1  per block column of 128, rows above the diagonal block by two dgemm, the diagonal block by dsyr2k (all the
//      small dsyr2k calls first, so the update ends with dgemm)
//   2  per block column of 128 by two dgemm including the diagonal block in full (also writes the strictly lower part
//      of the diagonal blocks: fine here, a copy of the matrix)
// Each variant: total time and the time in the symv-dominated rest (as sytrd_profile.c), median of 21; eigenvalues
// of the tridiagonal matrix (dsterf) against variant 0.
//   clang -O3 -mcpu=apple-m1 sytrd_syr2k.c -framework Accelerate -o sytrd_syr2k
//   VECLIB_MAXIMUM_THREADS=1 ./sytrd_syr2k 412 448 619 776
#include <mach/mach_time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define dsyr2k_ var_dsyr2k
#define ssyr2k_ var_ssyr2k
#include "../../xtb/src/mctc/lapack/sytrd.c"
#undef dsyr2k_
#undef ssyr2k_

extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *,
                    const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void ssyr2k_(const char *, const char *, const lint *, const lint *, const float *, const float *,
                    const lint *, const float *, const lint *, const float *, float *, const lint *);
extern void dsterf_(const lint *, double *, double *, lint *);
extern void ssterf_(const lint *, float *, float *, lint *);

static int variant;
static uint64_t t_update;

#define VAR_SYR2K(NAME, T, SYR2K, GEMM)                                                                               \
void NAME(const char *u, const char *t, const lint *np, const lint *kp, const T *al, const T *a, const lint *lda,      \
          const T *b, const lint *ldb, const T *be, T *c, const lint *ldc) {                                          \
  uint64_t t0 = mach_absolute_time();                                                                                 \
  lint n = *np, k = *kp, bs = 128;                                                                                    \
  if (variant == 0) SYR2K(u, t, np, kp, al, a, lda, b, ldb, be, c, ldc);                                             \
  else {                                                                                                              \
    if (variant == 1)                                                                                                 \
      for (lint j = 0; j < n; j += bs) {                                                                              \
        lint w = n - j < bs ? n - j : bs;                                                                             \
        SYR2K(u, t, &w, kp, al, a + j, lda, b + j, ldb, be, c + j + (size_t)j * *ldc, ldc);                          \
      }                                                                                                               \
    for (lint j = 0; j < n; j += bs) {                                                                                \
      lint w = n - j < bs ? n - j : bs, h = variant == 1 ? j : j + w;                                                 \
      if (h == 0) continue;                                                                                           \
      GEMM("N", "T", &h, &w, &k, al, a, lda, b + j, ldb, be, c + (size_t)j * *ldc, ldc);                             \
      GEMM("N", "T", &h, &w, &k, al, b, ldb, a + j, lda, be, c + (size_t)j * *ldc, ldc);                             \
    }                                                                                                                 \
  }                                                                                                                   \
  t_update += mach_absolute_time() - t0;                                                                              \
}
VAR_SYR2K(var_dsyr2k, double, dsyr2k_, dgemm_)
extern void sgemm_(const char *, const char *, const lint *, const lint *, const lint *, const float *, const float *,
                   const lint *, const float *, const lint *, const float *, float *, const lint *);
VAR_SYR2K(var_ssyr2k, float, ssyr2k_, sgemm_)

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb);
  double tick = 1e-6 * tb.numer / tb.denom;
  for (int ai = 1; ai < argc; ai++) {
    lint n = atoi(argv[ai]), lda = n, info;
    size_t nn = (size_t)n * n;
    for (int prec = 0; prec < 2; prec++) {
      size_t sz = prec ? 4 : 8;
      void *A0, *A, *work; posix_memalign(&A0, 16384, nn * sz); posix_memalign(&A, 16384, nn * sz);
      posix_memalign(&work, 16384, (size_t)n * 32 * sz);
      double *d = malloc(n * 8), *e = malloc(n * 8), *tau = malloc(n * 8), *ev0 = malloc(n * 8);
      srand(n);
      for (lint j = 0; j < n; j++)
        for (lint i = 0; i <= j; i++) {
          double v = (double)rand() / RAND_MAX - 0.5;
          if (prec) { ((float *)A0)[i + j * n] = v; ((float *)A0)[j + i * n] = v; }
          else { ((double *)A0)[i + j * n] = v; ((double *)A0)[j + i * n] = v; }
        }
      printf("n = %4d %s:", n, prec ? "FP32" : "FP64");
      double base = 0;
      for (variant = 0; variant < 3; variant++) {
        double tot[21], upd[21];
        for (int r = -2; r < 21; r++) {
          memcpy(A, A0, nn * sz); t_update = 0;
          uint64_t t0 = mach_absolute_time();
          if (prec) xtb_ssytrd_upper(&n, A, &lda, (float *)d, (float *)e, (float *)tau, work);
          else xtb_dsytrd_upper(&n, A, &lda, d, e, tau, work);
          uint64_t t1 = mach_absolute_time();
          if (r >= 0) { tot[r] = (t1 - t0) * tick; upd[r] = t_update * tick; }
        }
        // eigenvalues of the tridiagonal matrix
        double err = 0;
        if (prec) { lint nn1 = n; ssterf_(&nn1, (float *)d, (float *)e, &info); for (lint i = 0; i < n; i++) { double v = ((float *)d)[i]; if (variant == 0) ev0[i] = v; else err = fmax(err, fabs(v - ev0[i])); } }
        else { lint nn1 = n; dsterf_(&nn1, d, e, &info); for (lint i = 0; i < n; i++) { if (variant == 0) ev0[i] = d[i]; else err = fmax(err, fabs(d[i] - ev0[i])); } }
        double s[21]; memcpy(s, tot, sizeof s); qsort(s, 21, 8, cmp);
        int med = 0; for (int r = 0; r < 21; r++) if (tot[r] == s[10]) med = r;
        if (variant == 0) base = tot[med];
        printf("  v%d %.3f ms (%.3fx; update %.3f, rest %.3f; dEig %.0e)", variant, tot[med], base / tot[med], upd[med],
               tot[med] - upd[med], err);
      }
      printf("\n");
      free(A0); free(A); free(work); free(d); free(e); free(tau); free(ev0);
    }
  }
  return 0;
}
