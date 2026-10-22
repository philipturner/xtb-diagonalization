// Where does the symmetric 1-stage reduction (xtb_dsytrd_upper / xtb_ssytrd_upper, xtb/src/mctc/lapack/sytrd.c, what
// xTB ships) spend its time? Every Accelerate call is wrapped with a timer, by kind (gemv 'N', gemv 'T', syr2k, syr2,
// larfg, dot/axpy/scal); the rest (the NEON symv and the loops) is the remainder. Random symmetric matrices, lda = n,
// page-aligned; median of 21 calls (each a fresh copy), per-kind totals from the median call.
//   clang -O3 -mcpu=apple-m1 sytrd_profile.c -framework Accelerate -o sytrd_profile
//   VECLIB_MAXIMUM_THREADS=1 ./sytrd_profile 412 448 619 776
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define dgemv_ prof_dgemv
#define sgemv_ prof_sgemv
#define dsyr2k_ prof_dsyr2k
#define ssyr2k_ prof_ssyr2k
#define dsyr2_ prof_dsyr2
#define ssyr2_ prof_ssyr2
#define dlarfg_ prof_dlarfg
#define slarfg_ prof_slarfg
#include "../../xtb/src/mctc/lapack/sytrd.c"
#undef dgemv_
#undef sgemv_
#undef dsyr2k_
#undef ssyr2k_
#undef dsyr2_
#undef ssyr2_
#undef dlarfg_
#undef slarfg_

extern void dgemv_(const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, const lint *, const double *, double *, const lint *);
extern void sgemv_(const char *, const lint *, const lint *, const float *, const float *, const lint *,
                   const float *, const lint *, const float *, float *, const lint *);
extern void dsyr2k_(const char *, const char *, const lint *, const lint *, const double *, const double *,
                    const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void ssyr2k_(const char *, const char *, const lint *, const lint *, const float *, const float *,
                    const lint *, const float *, const lint *, const float *, float *, const lint *);
extern void dsyr2_(const char *, const lint *, const double *, const double *, const lint *, const double *,
                   const lint *, double *, const lint *);
extern void ssyr2_(const char *, const lint *, const float *, const float *, const lint *, const float *,
                   const lint *, float *, const lint *);
extern void dlarfg_(const lint *, double *, double *, const lint *, double *);
extern void slarfg_(const lint *, float *, float *, const lint *, float *);

enum { GEMV_N, GEMV_T, SYR2K, SYR2, LARFG, KINDS };
static const char *kind_name[KINDS] = {"gemv N", "gemv T", "syr2k", "syr2", "larfg"};
static uint64_t acc[KINDS];
#define TIMED(k, call) do { uint64_t t0 = mach_absolute_time(); call; acc[k] += mach_absolute_time() - t0; } while (0)

void prof_dgemv(const char *t, const lint *m, const lint *n, const double *al, const double *a, const lint *lda,
                const double *x, const lint *ix, const double *be, double *y, const lint *iy) {
  TIMED(t[0] == 'N' ? GEMV_N : GEMV_T, dgemv_(t, m, n, al, a, lda, x, ix, be, y, iy));
}
void prof_sgemv(const char *t, const lint *m, const lint *n, const float *al, const float *a, const lint *lda,
                const float *x, const lint *ix, const float *be, float *y, const lint *iy) {
  TIMED(t[0] == 'N' ? GEMV_N : GEMV_T, sgemv_(t, m, n, al, a, lda, x, ix, be, y, iy));
}
void prof_dsyr2k(const char *u, const char *t, const lint *n, const lint *k, const double *al, const double *a,
                 const lint *lda, const double *b, const lint *ldb, const double *be, double *c, const lint *ldc) {
  TIMED(SYR2K, dsyr2k_(u, t, n, k, al, a, lda, b, ldb, be, c, ldc));
}
void prof_ssyr2k(const char *u, const char *t, const lint *n, const lint *k, const float *al, const float *a,
                 const lint *lda, const float *b, const lint *ldb, const float *be, float *c, const lint *ldc) {
  TIMED(SYR2K, ssyr2k_(u, t, n, k, al, a, lda, b, ldb, be, c, ldc));
}
void prof_dsyr2(const char *u, const lint *n, const double *al, const double *x, const lint *ix, const double *y,
                const lint *iy, double *a, const lint *lda) {
  TIMED(SYR2, dsyr2_(u, n, al, x, ix, y, iy, a, lda));
}
void prof_ssyr2(const char *u, const lint *n, const float *al, const float *x, const lint *ix, const float *y,
                const lint *iy, float *a, const lint *lda) {
  TIMED(SYR2, ssyr2_(u, n, al, x, ix, y, iy, a, lda));
}
void prof_dlarfg(const lint *n, double *al, double *x, const lint *ix, double *tau) {
  TIMED(LARFG, dlarfg_(n, al, x, ix, tau));
}
void prof_slarfg(const lint *n, float *al, float *x, const lint *ix, float *tau) {
  TIMED(LARFG, slarfg_(n, al, x, ix, tau));
}

static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  mach_timebase_info_data_t tb; mach_timebase_info(&tb);
  double tick = 1e-6 * tb.numer / tb.denom;  // ms per tick
  for (int ai = 1; ai < argc; ai++) {
    lint n = atoi(argv[ai]), lda = n;
    size_t nn = (size_t)n * n;
    for (int prec = 0; prec < 2; prec++) {
      size_t sz = prec ? 4 : 8;
      void *A0, *A, *work, *d, *e, *tau;
      posix_memalign(&A0, 16384, nn * sz); posix_memalign(&A, 16384, nn * sz);
      posix_memalign(&work, 16384, (size_t)n * 32 * sz);
      d = malloc(n * sz); e = malloc(n * sz); tau = malloc(n * sz);
      srand(1);
      for (lint j = 0; j < n; j++)
        for (lint i = 0; i <= j; i++) {
          double v = (double)rand() / RAND_MAX - 0.5;
          if (prec) { ((float *)A0)[i + j * n] = v; ((float *)A0)[j + i * n] = v; }
          else { ((double *)A0)[i + j * n] = v; ((double *)A0)[j + i * n] = v; }
        }
      double tot[21], parts[21][KINDS];
      for (int r = -2; r < 21; r++) {
        memcpy(A, A0, nn * sz);
        memset(acc, 0, sizeof acc);
        uint64_t t0 = mach_absolute_time();
        if (prec) xtb_ssytrd_upper(&n, A, &lda, d, e, tau, work);
        else xtb_dsytrd_upper(&n, A, &lda, d, e, tau, work);
        uint64_t t1 = mach_absolute_time();
        if (r < 0) continue;
        tot[r] = (t1 - t0) * tick;
        for (int k = 0; k < KINDS; k++) parts[r][k] = acc[k] * tick;
      }
      double sorted[21]; memcpy(sorted, tot, sizeof tot); qsort(sorted, 21, 8, cmp);
      int med = 0; for (int r = 0; r < 21; r++) if (tot[r] == sorted[10]) med = r;
      printf("n = %4d %s total %6.3f ms:", n, prec ? "FP32" : "FP64", tot[med]);
      double rest = tot[med];
      for (int k = 0; k < KINDS; k++) { printf("  %s %.3f (%.0f%%)", kind_name[k], parts[med][k], 100 * parts[med][k] / tot[med]); rest -= parts[med][k]; }
      printf("  rest %.3f (%.0f%%)\n", rest, 100 * rest / tot[med]);
      free(A0); free(A); free(work); free(d); free(e); free(tau);
    }
  }
  return 0;
}
