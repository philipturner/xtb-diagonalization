// sytrd.c's xtb_dstedc / xtb_sstedc (divide and conquer on the caller's OpenMP threads) and the OpenMP column blocks
// of xtb_dormtr_upper / xtb_sormtr_upper, against Accelerate's xstedc and the serial back-transformation, on the
// tridiagonal matrices of dumped GFN2 SCC cycles (H' = U^-T H U^-1 reduced by sytrd.c). Per thread count: median
// time of 11, eigenvalues vs xstedc, residual max|T z - lambda z| / max|lambda| and orthogonality max|Z^T Z - I|
// (FP64 accumulation), and max|dZ| of the parallel back-transformation of m = n / 2 columns vs the serial one.
//   gcc-16 -O3 -fopenmp stedc_omp.c -framework Accelerate -o stedc_omp
//   VECLIB_MAXIMUM_THREADS=1 ./stedc_omp build/dumps/facet_closed/hs_1.bin ...
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dsygst_(const lint *, const char *, const lint *, double *, const lint *, const double *, const lint *,
                    lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

#define CHECK(P, T)                                                                                                   \
static void check_##P(lint n, const double *H) {                                                                     \
  T *A = malloc(sizeof(T) * n * n), *d0 = malloc(sizeof(T) * n), *e0 = malloc(sizeof(T) * n), *tau = malloc(sizeof(T) * n), \
    *wk = malloc(sizeof(T) * n * 32), *d = malloc(sizeof(T) * n), *e = malloc(sizeof(T) * n),                        \
    *Z = malloc(sizeof(T) * n * n), *dref = malloc(sizeof(T) * n);                                                    \
  for (size_t i = 0; i < (size_t)n * n; i++) A[i] = (T)H[i];                                                          \
  xtb_##P##sytrd_upper(&n, A, &n, d0, e0, tau, wk);                                                                   \
  lint lwork = 1 + 4 * n + n * n, liwork = 3 + 5 * n, info;                                                           \
  T *work = malloc(sizeof(T) * lwork); lint *iwork = malloc(sizeof(lint) * liwork);                                   \
  memcpy(dref, d0, sizeof(T) * n); memcpy(e, e0, sizeof(T) * n);                                                      \
  P##stedc_("I", &n, dref, e, Z, &n, work, &lwork, iwork, &liwork, &info);                                            \
  lint m = n / 2;                                                                                                     \
  T *Zs = malloc(sizeof(T) * n * m), *Zp = malloc(sizeof(T) * n * m), *ow = malloc(sizeof(T) * 32 * (4 * n + 32));    \
  int counts[4] = {1, 2, 4, 8};                                                                                       \
  for (int c = 0; c < 4; c++) {                                                                                       \
    omp_set_num_threads(counts[c]);                                                                                   \
    double t[11];                                                                                                     \
    for (int r = -1; r < 11; r++) {                                                                                   \
      memcpy(d, d0, sizeof(T) * n); memcpy(e, e0, sizeof(T) * n);                                                     \
      double t0 = now();                                                                                              \
      xtb_##P##stedc(&n, d, e, Z, &n, work, &lwork, iwork, &liwork, &info);                                           \
      if (r >= 0) t[r] = 1e3 * (now() - t0);                                                                          \
      if (info) { printf("info %d\n", (int)info); exit(1); }                                                          \
    }                                                                                                                 \
    qsort(t, 11, 8, cmp);                                                                                             \
    double ev = 0, res = 0, orth = 0, scale = 0;                                                                      \
    for (lint i = 0; i < n; i++) { ev = fmax(ev, fabs((double)d[i] - dref[i])); scale = fmax(scale, fabs((double)dref[i])); } \
    for (lint a = 0; a < n; a += 7) {                                                                                 \
      const T *za = Z + (size_t)a * n;                                                                                \
      for (lint i = 0; i < n; i++) {                                                                                  \
        double tz = (double)d0[i] * za[i] + (i > 0 ? (double)e0[i - 1] * za[i - 1] : 0) + (i < n - 1 ? (double)e0[i] * za[i + 1] : 0); \
        res = fmax(res, fabs(tz - (double)d[a] * za[i]));                                                             \
      }                                                                                                               \
      for (lint b2 = 0; b2 < n; b2 += 5) {                                                                            \
        const T *zb = Z + (size_t)b2 * n; double g = 0;                                                              \
        for (lint i = 0; i < n; i++) g += (double)za[i] * zb[i];                                                      \
        orth = fmax(orth, fabs(g - (a == b2)));                                                                       \
      }                                                                                                               \
    }                                                                                                                 \
    /* back-transformation of the first m columns: serial (1 thread) vs this count */                                 \
    omp_set_num_threads(1); memcpy(Zs, Z, sizeof(T) * n * m); xtb_##P##ormtr_upper(&n, &m, A, &n, tau, Zs, &n, ow);   \
    omp_set_num_threads(counts[c]); memcpy(Zp, Z, sizeof(T) * n * m);                                                 \
    double tb[11];                                                                                                    \
    for (int r = -1; r < 11; r++) {                                                                                   \
      memcpy(Zp, Z, sizeof(T) * n * m); double t0 = now();                                                            \
      xtb_##P##ormtr_upper(&n, &m, A, &n, tau, Zp, &n, ow);                                                           \
      if (r >= 0) tb[r] = 1e3 * (now() - t0);                                                                         \
    }                                                                                                                 \
    qsort(tb, 11, 8, cmp);                                                                                            \
    double dz = 0; for (size_t i = 0; i < (size_t)n * m; i++) dz = fmax(dz, fabs((double)Zp[i] - Zs[i]));             \
    printf("  %s %d threads: stedc %8.3f ms, eigenvalues %.1e, residual %.1e, orthogonality %.1e | ormtr %7.3f ms, max|dZ| %.1e\n", \
           sizeof(T) == 8 ? "FP64" : "FP32", counts[c], t[5], ev, res / scale, orth, tb[5], dz);                          \
  }                                                                                                                   \
  free(A); free(d0); free(e0); free(tau); free(wk); free(d); free(e); free(Z); free(dref); free(work); free(iwork);   \
  free(Zs); free(Zp); free(ow);                                                                                       \
}
CHECK(d, double)
CHECK(s, float)

int main(int argc, char **argv) {
  for (int ai = 1; ai < argc; ai++) {
    FILE *f = fopen(argv[ai], "rb"); int n32; fread(&n32, 4, 1, f);
    lint n = n32, info, one = 1;
    double *H = malloc((size_t)n * n * 8), *S = malloc((size_t)n * n * 8);
    fread(H, 8, (size_t)n * n, f); fread(S, 8, (size_t)n * n, f); fclose(f);
    dpotrf_("U", &n, S, &n, &info); dsygst_(&one, "U", &n, H, &n, S, &n, &info);
    for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) H[i + j * n] = H[j + i * n];
    printf("%s n = %d\n", argv[ai], n);
    check_d(n, H);
    check_s(n, H);
    free(H); free(S);
  }
  return 0;
}
