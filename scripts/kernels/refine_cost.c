// Cost of the FP64 refinement of an FP32 eigensolve (xtb eigensolve.F90, refine_block), piece by piece, next to the
// FP64 and FP32 eigensolves it competes with. One SCC cycle's eigensolve of H' = X^T H X (congruence not timed).
//   refine_cost HS.bin K [REPS]   (K: refined block, xtb uses last occupied + 32)
//   clang -O2 refine_cost.c -framework Accelerate -o refine_cost
//   VECLIB_MAXIMUM_THREADS=1 ./refine_cost HS.bin 420
// Accelerate's legacy LAPACK for every piece (xtb's own vectorized sytrd isn't used here). Median of REPS (default 9).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef int lint;
#define LAPACK(T, P) \
extern void P##sytrd_(const char *, const lint *, T *, const lint *, T *, T *, T *, T *, const lint *, lint *); \
extern void P##stedc_(const char *, const lint *, T *, T *, T *, const lint *, T *, const lint *, lint *, \
                      const lint *, lint *); \
extern void P##ormtr_(const char *, const char *, const char *, const lint *, const lint *, const T *, const lint *, \
                      const T *, T *, const lint *, T *, const lint *, lint *);
LAPACK(double, d)
LAPACK(float, s)
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dtrsm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dgemm_(const char *, const char *, const lint *, const lint *, const lint *, const double *, const double *,
                   const lint *, const double *, const lint *, const double *, double *, const lint *);
extern void dsymm_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, const lint *, const double *, double *, const lint *);
extern void dsyrk_(const char *, const char *, const lint *, const lint *, const double *, const double *, const lint *,
                   const double *, double *, const lint *);
extern void dsyevd_(const char *, const char *, const lint *, double *, const lint *, double *, double *, const lint *,
                    lint *, const lint *, lint *);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

#define TIME(label, setup, body) do { \
  double ts[64]; for (int r = -1; r < reps; r++) { setup; double t0 = now(); body; if (r >= 0) ts[r] = now() - t0; } \
  qsort(ts, reps, 8, cmp); printf("  %-44s %8.3f ms\n", label, 1e3 * ts[reps / 2]); total += ts[reps / 2]; } while (0)

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); lint k = atoi(argv[2]); int reps = argc > 3 ? atoi(argv[3]) : 9;
  int n4; fread(&n4, 4, 1, f); lint n = n4, nv; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1, zero = 0, mone = -1, total;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) H[i + j * n] = H[j + i * n];
  if (k > n) k = n;
  nv = n - k;
  lint lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
  double *A = malloc(nn * 8), *w = malloc(n * 8), *e = malloc(n * 8), *tau = malloc(n * 8), *Z = malloc(nn * 8);
  double *work = malloc(lwork * 8), *R = malloc(nn * 8), *Y = malloc(nn * 8), *G = malloc(nn * 8);
  float *As = malloc(nn * 4), *ws = malloc(n * 4), *es = malloc(n * 4), *taus = malloc(n * 4), *Zs = malloc(nn * 4);
  float *works = malloc(lwork * 4);
  lint *iwork = malloc(liwork * sizeof(lint));
  printf("%s  n = %d, refined block k = %d, median of %d\n", argv[1], n, k, reps);

  printf(" FP64 eigensolve (xTB's FP64 cycle; back-transformation of K vectors)\n"); total = 0;
  TIME("dsytrd", memcpy(A, H, nn * 8), dsytrd_("U", &n, A, &n, w, e, tau, work, &lwork, &info));
  double *w0 = malloc(n * 8), *e0 = malloc(n * 8), *A0 = malloc(nn * 8), *tau0 = malloc(n * 8);
  memcpy(w0, w, n * 8); memcpy(e0, e, n * 8); memcpy(A0, A, nn * 8); memcpy(tau0, tau, n * 8);
  TIME("dstedc", (memcpy(w, w0, n * 8), memcpy(e, e0, n * 8)),
       dstedc_("I", &n, w, e, Z, &n, work, &lwork, iwork, &liwork, &info));
  double *Zt = malloc(nn * 8); memcpy(Zt, Z, nn * 8);
  TIME("dormtr, K columns", memcpy(Z, Zt, nn * 8), dormtr_("L", "U", "N", &n, &k, A0, &n, tau0, Z, &n, work, &lwork, &info));
  printf("  %-44s %8.3f ms\n", "total", 1e3 * total);

  printf(" FP32 eigensolve\n"); total = 0;
  TIME("ssytrd", for (size_t i = 0; i < nn; i++) As[i] = (float)H[i], ssytrd_("U", &n, As, &n, ws, es, taus, works, &lwork, &info));
  float *ws0 = malloc(n * 4), *es0 = malloc(n * 4); memcpy(ws0, ws, n * 4); memcpy(es0, es, n * 4);
  TIME("sstedc", (memcpy(ws, ws0, n * 4), memcpy(es, es0, n * 4)),
       sstedc_("I", &n, ws, es, Zs, &n, works, &lwork, iwork, &liwork, &info));
  float *Zs0 = malloc(nn * 4); memcpy(Zs0, Zs, nn * 4);
  double t32 = total;
  TIME("sormtr, K columns (unrefined FP32 cycle)", memcpy(Zs, Zs0, nn * 4), sormtr_("L", "U", "N", &n, &k, As, &n, taus, Zs, &n, works, &lwork, &info));
  printf("  %-44s %8.3f ms\n", "total, unrefined", 1e3 * total);
  total = t32;
  TIME("sormtr, all n columns (refined)", memcpy(Zs, Zs0, nn * 4), sormtr_("L", "U", "N", &n, &n, As, &n, taus, Zs, &n, works, &lwork, &info));
  printf("  %-44s %8.3f ms\n", "total before refinement", 1e3 * total);

  printf(" FP64 refinement (refine_block)\n"); total = 0;
  for (size_t i = 0; i < nn; i++) Z[i] = Zs[i];
  for (lint i = 0; i < n; i++) w[i] = ws[i];
  double *Z0 = malloc(nn * 8); memcpy(Z0, Z, nn * 8);
  TIME("promote Z to FP64", , for (size_t i = 0; i < nn; i++) Z[i] = Zs[i]);
  TIME("residual: dsymm (A Z_k) - Z_k e", ,
       (dsymm_("L", "U", &n, &k, &one, H, &n, Z, &n, &zero, R, &n),
        ({ for (lint i = 0; i < k; i++) for (lint r = 0; r < n; r++) R[r + i * n] -= w[i] * Z[r + i * n]; })));
  TIME("Y = Z_v^T R (dgemm)", , dgemm_("T", "N", &nv, &k, &n, &one, Z + (size_t)k * n, &n, R, &n, &zero, Y, &nv));
  TIME("Z_k -= Z_v Y (dgemm)", memcpy(Z, Z0, (size_t)n * k * 8),
       dgemm_("N", "N", &n, &k, &nv, &mone, Z + (size_t)k * n, &n, Y, &nv, &one, Z, &n));
  TIME("Cholesky QR: dsyrk + dpotrf", , (dsyrk_("U", "T", &k, &n, &one, Z, &n, &zero, G, &n), dpotrf_("U", &k, G, &n, &info)));
  TIME("Cholesky QR: dtrsm", memcpy(Z, Z0, (size_t)n * k * 8), dtrsm_("R", "U", "N", "N", &n, &k, &one, G, &n, Z, &n));
  TIME("Rayleigh-Ritz: dsymm (A Z_k)", , dsymm_("L", "U", &n, &k, &one, H, &n, Z, &n, &zero, R, &n));
  TIME("Rayleigh-Ritz: G = Z_k^T (A Z_k) (dgemm)", , dgemm_("T", "N", &k, &k, &n, &one, Z, &n, R, &n, &zero, G, &n));
  double *G0 = malloc(nn * 8); memcpy(G0, G, nn * 8);
  TIME("Rayleigh-Ritz: dsyevd (k x k)", memcpy(G, G0, nn * 8), dsyevd_("V", "U", &k, G, &n, w, work, &lwork, iwork, &liwork, &info));
  TIME("Rayleigh-Ritz: Z_k Q (dgemm)", , dgemm_("N", "N", &n, &k, &k, &one, Z, &n, G, &n, &zero, R, &n));
  printf("  %-44s %8.3f ms\n", "total refinement", 1e3 * total);
  return 0;
}
