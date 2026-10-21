// Can the SCC's tridiagonal eigenproblem compute only the eigenvectors it back-transforms (the lowest m) faster than
// divide and conquer computes all n? xTB (e7b0317) calls xstedc('I') every cycle and back-transforms m_occ columns.
//   tridiag_subset HS.bin [REPS]   (HS.bin from diag/dump; tridiagonal form from dsytrd_ of H' = X^T H X)
//   clang -O2 tridiag_subset.c -framework Accelerate -o tridiag_subset
//   VECLIB_MAXIMUM_THREADS=1 ./tridiag_subset HS.bin
// Every variant returns all n eigenvalues (xTB's Fermi occupations need them) and the eigenvectors 1 ... m:
//   stedc        xstedc('I'): all vectors (today)
//   sterf+stemr  all eigenvalues by xsterf, then MRRR (xstemr, RANGE = 'I') for vectors 1 ... m
//   stebz+stein  all eigenvalues by bisection (xstebz, RANGE = 'A', ORDER = 'B'), inverse iteration (xstein) for 1 ... m
// m = n / 4, n / 2, n. Median of REPS calls (default 9). Checks, in FP64: max |w - w(dstedc)| over all n, residual
// max |T z - w z| and orthogonality max |Z^T Z - I| over the m vectors.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef int lint;
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dsytrd_(const char *, const lint *, double *, const lint *, double *, double *, double *, double *,
                    const lint *, lint *);
#define DECL(T, P) \
extern void P##stedc_(const char *, const lint *, T *, T *, T *, const lint *, T *, const lint *, lint *, const lint *, \
                      lint *); \
extern void P##sterf_(const lint *, T *, T *, lint *); \
extern void P##stemr_(const char *, const char *, const lint *, T *, T *, const T *, const T *, const lint *, \
                      const lint *, lint *, T *, T *, const lint *, const lint *, lint *, lint *, T *, const lint *, \
                      lint *, const lint *, lint *); \
extern void P##stebz_(const char *, const char *, const lint *, const T *, const T *, const lint *, const lint *, \
                      const T *, const T *, const T *, lint *, lint *, T *, lint *, lint *, T *, lint *, lint *); \
extern void P##stein_(const lint *, const T *, const T *, const lint *, const T *, const lint *, const lint *, T *, \
                      const lint *, T *, lint *, lint *, lint *);
DECL(double, d)
DECL(float, s)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static lint n;
static double *D0, *E0, *Wref;

// Residual and orthogonality of the m vectors z (ldz = n) with eigenvalues w, against T(D0, E0), in FP64.
static void check(lint m, const double *w, const double *z, double *res, double *orth) {
  double r = 0, o = 0;
  for (lint k = 0; k < m; k++) {
    const double *x = z + (size_t)k * n;
    for (lint i = 0; i < n; i++) {
      double t = D0[i] * x[i] - w[k] * x[i];
      if (i > 0) t += E0[i - 1] * x[i - 1];
      if (i < n - 1) t += E0[i] * x[i + 1];
      r = fmax(r, fabs(t));
    }
  }
  for (lint k = 0; k < m; k++)
    for (lint l = k; l < m; l++) {
      double s = 0;
      for (lint i = 0; i < n; i++) s += z[i + (size_t)k * n] * z[i + (size_t)l * n];
      o = fmax(o, fabs(s - (k == l)));
    }
  *res = r; *orth = o;
}

#define RUN(T, P, NAME) \
static void run_##P(int reps) { \
  T *d = malloc(n * sizeof(T)), *e = malloc(n * sizeof(T)), *w = malloc(n * sizeof(T)); \
  T *z; posix_memalign((void **)&z, 16384, (size_t)n * n * sizeof(T)); \
  lint lwork = 1 + 4 * n + n * n + 18 * n, liwork = 3 + 10 * n; \
  T *work = malloc(lwork * sizeof(T)); lint *iwork = malloc(liwork * sizeof(lint)); \
  lint *isuppz = malloc(2 * n * sizeof(lint)), *iblock = malloc(n * sizeof(lint)), *isplit = malloc(n * sizeof(lint)); \
  lint *ifail = malloc(n * sizeof(lint)), *idx = malloc(n * sizeof(lint)), *keep = malloc(n * sizeof(lint)); \
  lint *bs = malloc(n * sizeof(lint)); T *ws = malloc(n * sizeof(T)), *wall = malloc(n * sizeof(T)); \
  double *wd = malloc(n * 8), *zd = malloc((size_t)n * n * 8), *t = malloc(reps * 8); \
  lint ms[3] = {n / 4, n / 2, n}; \
  for (int v = 0; v < 3; v++) for (int mi = (v == 0 ? 2 : 0); mi < 3; mi++) { \
    lint m = ms[mi], info = 0, mfound = 0, nsplit = 0; \
    for (int r = -1; r < reps; r++) { \
      for (lint i = 0; i < n; i++) { d[i] = (T)D0[i]; e[i] = i < n - 1 ? (T)E0[i] : 0; } \
      double t0 = now(); \
      if (v == 0) { \
        P##stedc_("I", &n, d, e, z, &n, work, &lwork, iwork, &liwork, &info); \
        memcpy(w, d, n * sizeof(T)); mfound = n; \
      } else if (v == 1) { \
        T *dc = work, *ec = work + n, *wk = work + 2 * n; lint lw = lwork - 2 * n, one = 1, tryrac = 1; T zero = 0; \
        memcpy(dc, d, n * sizeof(T)); memcpy(ec, e, n * sizeof(T)); \
        P##sterf_(&n, dc, ec, &info); \
        memcpy(w, dc, n * sizeof(T)); \
        P##stemr_("V", "I", &n, d, e, &zero, &zero, &one, &m, &mfound, dc, z, &n, &n, isuppz, &tryrac, wk, &lw, \
                  iwork, &liwork, &info); \
        memcpy(w, dc, m * sizeof(T)); \
      } else { \
        T zero = 0, abstol = 0; lint one = 1; \
        P##stebz_("A", "B", &n, &zero, &zero, &one, &one, &abstol, d, e, &mfound, &nsplit, w, iblock, isplit, work, \
                  iwork, &info); \
        /* ORDER = 'B': by split block, ascending within each; xstein wants that order. Vectors for the m lowest: */ \
        /* pick them by rank, then pack (w, iblock) for those. */ \
        for (lint i = 0; i < n; i++) idx[i] = i; \
        for (lint i = 1; i < n; i++) { lint k = idx[i], j = i; while (j > 0 && w[idx[j - 1]] > w[k]) { idx[j] = idx[j - 1]; j--; } idx[j] = k; } \
        for (lint i = 0; i < n; i++) keep[i] = 0; \
        for (lint i = 0; i < m; i++) keep[idx[i]] = 1; \
        lint c = 0; for (lint i = 0; i < n; i++) if (keep[i]) { ws[c] = w[i]; bs[c] = iblock[i]; c++; } \
        P##stein_(&n, d, e, &m, ws, bs, isplit, z, &n, work, iwork, ifail, &info); \
        for (lint i = 0; i < n; i++) wall[i] = w[idx[i]]; \
        memcpy(w, wall, n * sizeof(T)); \
      } \
      if (r >= 0) t[r] = 1e3 * (now() - t0); \
      if (info != 0) { printf("%s variant %d info %d\n", NAME, v, info); break; } \
    } \
    qsort(t, reps, 8, cmp); \
    double dw = 0; for (lint i = 0; i < n; i++) dw = fmax(dw, fabs((double)w[i] - Wref[i])); \
    for (lint k = 0; k < m; k++) wd[k] = v == 2 ? (double)ws[k] : (double)w[k]; \
    for (size_t i = 0; i < (size_t)n * m; i++) zd[i] = (double)z[i]; \
    double res, orth; check(m, wd, zd, &res, &orth); \
    static const char *vn[3] = {"stedc      ", "sterf+stemr", "stebz+stein"}; \
    printf("%s %s m = %4d: median %8.3f ms (min %8.3f) | eigenvalues %.1e  residual %.1e  orthogonality %.1e\n", \
           NAME, vn[v], m, t[reps / 2], t[0], dw, res, orth); \
  } \
  free(d); free(e); free(w); free(z); free(work); free(iwork); free(isuppz); free(iblock); free(isplit); free(ifail); \
  free(wd); free(zd); free(t); \
}
RUN(double, d, "FP64")
RUN(float, s, "FP32")

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); int reps = argc > 2 ? atoi(argv[2]) : 9;
  int n4; fread(&n4, 4, 1, f); n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  D0 = malloc(n * 8); E0 = malloc(n * 8); Wref = malloc(n * 8);
  double *tau = malloc(n * 8), *e = malloc(n * 8); lint lwork = 64 * n; double *work = malloc(lwork * 8);
  dsytrd_("U", &n, H, &n, D0, E0, tau, work, &lwork, &info);
  memcpy(Wref, D0, n * 8); memcpy(e, E0, n * 8);
  extern void dsterf_(const lint *, double *, double *, lint *);
  dsterf_(&n, Wref, e, &info);
  printf("%s  n = %d, median of %d\n", argv[1], n, reps);
  run_d(reps);
  run_s(reps);
  return 0;
}
