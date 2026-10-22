// sytrd.c's xtb_dsytrd_upper / xtb_ssytrd_upper on the caller's OpenMP threads (one team per call, spinning
// barrier) against the serial path (1 thread), on the H' = U^-T H U^-1 of dumped GFN2 SCC cycles. Per thread count:
// median time of 11; max |eigenvalue - serial's| / max|lambda| (xsterf of the two tridiagonal matrices); residual
// max|H' z - lambda z| / max|lambda| of every 7th eigenvector back-transformed by xtb_xormtr_upper (FP64
// accumulation); a hash of the last call's d and e (the same for every call with the same thread count).
//   gcc-16 -O3 -fopenmp sytrd_team.c -framework Accelerate -o sytrd_team
//   VECLIB_MAXIMUM_THREADS=1 ./sytrd_team build/dumps/facet_closed/hs_1.bin ...   (COUNTS=1,2,4: thread counts)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../xtb/src/mctc/lapack/sytrd.c"

extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dsygst_(const lint *, const char *, const lint *, double *, const lint *, const double *, const lint *,
                    lint *);
extern void dsterf_(const lint *, double *, double *, lint *);
extern void ssterf_(const lint *, float *, float *, lint *);
extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void sstedc_(const char *, const lint *, float *, float *, float *, const lint *, float *, const lint *,
                    lint *, const lint *, lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

#define CHECK(P, T)                                                                                                   \
static void check_##P(lint n, const double *H, int ncounts, const int *counts) {                                     \
  T *A0 = malloc(sizeof(T) * n * n), *A = malloc(sizeof(T) * n * n), *d = malloc(sizeof(T) * n),                     \
    *e = malloc(sizeof(T) * n), *tau = malloc(sizeof(T) * n), *wk = malloc(sizeof(T) * n * 32),                      \
    *ref = malloc(sizeof(T) * n), *ev = malloc(sizeof(T) * n), *et = malloc(sizeof(T) * n);                          \
  for (size_t i = 0; i < (size_t)n * n; i++) A0[i] = (T)H[i];                                                         \
  lint info;                                                                                                          \
  for (int c = 0; c < ncounts; c++) {                                                                                 \
    omp_set_num_threads(counts[c]);                                                                                   \
    double t[11];                                                                                                     \
    for (int r = -1; r < 11; r++) {                                                                                   \
      memcpy(A, A0, sizeof(T) * n * n);                                                                               \
      double t0 = now();                                                                                              \
      xtb_##P##sytrd_upper(&n, A, &n, d, e, tau, wk);                                                                 \
      if (r >= 0) t[r] = 1e3 * (now() - t0);                                                                          \
    }                                                                                                                 \
    qsort(t, 11, 8, cmp);                                                                                             \
    memcpy(ev, d, sizeof(T) * n); memcpy(et, e, sizeof(T) * n);                                                       \
    P##sterf_(&n, ev, et, &info);                                                                                     \
    if (c == 0) memcpy(ref, ev, sizeof(T) * n);                                                                       \
    double dev = 0, scale = 0;                                                                                        \
    for (lint i = 0; i < n; i++) { dev = fmax(dev, fabs((double)ev[i] - ref[i])); scale = fmax(scale, fabs((double)ref[i])); } \
    /* eigenvectors of the tridiagonal matrix, back-transformed with this call's reflectors */                       \
    lint lwork = 1 + 4 * n + n * n, liwork = 3 + 5 * n;                                                               \
    T *Z = malloc(sizeof(T) * n * n), *work = malloc(sizeof(T) * lwork), *ow = malloc(sizeof(T) * 32 * (4 * n + 32)); \
    lint *iwork = malloc(sizeof(lint) * liwork);                                                                      \
    memcpy(ev, d, sizeof(T) * n); memcpy(et, e, sizeof(T) * n);                                                       \
    P##stedc_("I", &n, ev, et, Z, &n, work, &lwork, iwork, &liwork, &info);                                           \
    omp_set_num_threads(1); xtb_##P##ormtr_upper(&n, &n, A, &n, tau, Z, &n, ow);                                      \
    double res = 0;                                                                                                   \
    for (lint k = 0; k < n; k += 7) {                                                                                 \
      const T *z = Z + (size_t)k * n;                                                                                 \
      for (lint i = 0; i < n; i++) {                                                                                  \
        double s = 0; for (lint j = 0; j < n; j++) s += H[i + (size_t)j * n] * z[j];                                 \
        res = fmax(res, fabs(s - (double)ev[k] * z[i]));                                                              \
      }                                                                                                               \
    }                                                                                                                 \
    free(Z); free(work); free(ow); free(iwork);                                                                       \
    unsigned long long hash = 1469598103934665603ULL;                                                                 \
    for (size_t i = 0; i < sizeof(T) * n; i++) hash = (hash ^ ((unsigned char *)d)[i]) * 1099511628211ULL;           \
    for (size_t i = 0; i < sizeof(T) * (n - 1); i++) hash = (hash ^ ((unsigned char *)e)[i]) * 1099511628211ULL;     \
    printf("  %s %d threads: sytrd %8.3f ms (min %8.3f), eigenvalues vs serial %.1e, residual %.1e, d e %016llx\n",  \
           sizeof(T) == 8 ? "FP64" : "FP32", counts[c], t[5], t[0], dev / scale, res / scale, hash);                  \
  }                                                                                                                   \
  free(A0); free(A); free(d); free(e); free(tau); free(wk); free(ref); free(ev); free(et);                           \
}
CHECK(d, double)
CHECK(s, float)

int main(int argc, char **argv) {
  int counts[8] = {1, 2, 3, 4, 6, 8}, ncounts = 6;
  if (getenv("COUNTS")) { ncounts = 0; for (char *p = strtok(getenv("COUNTS"), ","); p; p = strtok(NULL, ",")) counts[ncounts++] = atoi(p); }
  for (int ai = 1; ai < argc; ai++) {
    FILE *f = fopen(argv[ai], "rb"); int n32; fread(&n32, 4, 1, f);
    lint n = n32, info, one = 1;
    double *H = malloc((size_t)n * n * 8), *S = malloc((size_t)n * n * 8);
    fread(H, 8, (size_t)n * n, f); fread(S, 8, (size_t)n * n, f); fclose(f);
    dpotrf_("U", &n, S, &n, &info); dsygst_(&one, "U", &n, H, &n, S, &n, &info);
    for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) H[i + j * n] = H[j + i * n];
    printf("%s n = %d\n", argv[ai], n);
    check_d(n, H, ncounts, counts);
    check_s(n, H, ncounts, counts);
    free(H); free(S);
  }
  return 0;
}
