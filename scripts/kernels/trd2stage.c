// Phase 4, step 1: is a two-stage tridiagonal reduction worth porting at production sizes?
//   trd2stage HS.bin [REPS]
// HS.bin: dump from diag/dump (int32 n, H(n,n), S(n,n), ...). Forms H' = X^T H X (X = U^-1, S = U^T U) and times
// the reduction to tridiagonal form only (median of REPS, ms):
//   legacy dsytrd_ (what xTB links), new-interface dsytrd ($NEWLAPACK), new dsytrd_2stage (VECT = 'N': Accelerate,
//   like reference LAPACK, has no back-transformation for the second stage), and FP32 ssytrd_.
// Also checks that the 2-stage tridiagonal matrix has the same eigenvalues (dsterf) as the legacy one.
// Build: clang -O2 trd2stage.c -framework Accelerate -o trd2stage
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef int lint;
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *,
                   const double *, const double *, const lint *, double *, const lint *);
extern void dsytrd_(const char *, const lint *, double *, const lint *, double *, double *, double *, double *,
                    const lint *, lint *);
extern void ssytrd_(const char *, const lint *, float *, const lint *, float *, float *, float *, float *,
                    const lint *, lint *);
extern void dsterf_(const lint *, double *, double *, lint *);
extern void dsytrd_new(const char *, const lint *, double *, const lint *, double *, double *, double *, double *,
                       const lint *, lint *) __asm__("_dsytrd$NEWLAPACK");
extern void dsytrd_2stage(const char *, const char *, const lint *, double *, const lint *, double *, double *,
                          double *, double *, const lint *, double *, const lint *, lint *)
                          __asm__("_dsytrd_2stage$NEWLAPACK");

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8), *Hp = malloc(nn * 8), *A = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  memcpy(Hp, H, nn * 8);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, Hp, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, Hp, &n);

  double *d = malloc(n * 8), *e = malloc(n * 8), *tau = malloc(n * 8), *d2 = malloc(n * 8), *e2 = malloc(n * 8);
  double *t = malloc(reps * 8);
  float *As = malloc(nn * 4), *ds = malloc(n * 4), *es = malloc(n * 4), *taus = malloc(n * 4);
  lint lwork = 64 * n + 1000000; double *work = malloc(lwork * 8); float *works = malloc(lwork * 4);
  // 2-stage workspace: query
  double hq, wq; lint m1 = -1, lhous, lw2;
  dsytrd_2stage("N", "U", &n, A, &n, d2, e2, tau, &hq, &m1, &wq, &m1, &info);
  lhous = (lint)hq; lw2 = (lint)wq;
  double *hous = malloc((size_t)lhous * 8), *work2 = malloc((size_t)lw2 * 8);
  printf("n = %d (2-stage workspace: hous %d, work %d)\n", n, lhous, lw2);

  const char *name[4] = {"legacy dsytrd_      ", "new dsytrd          ", "new dsytrd_2stage N ", "FP32 ssytrd_        "};
  for (int k = 0; k < 4; k++) {
    for (int r = -1; r < reps; r++) {
      memcpy(A, Hp, nn * 8);
      if (k == 3) for (size_t i = 0; i < nn; i++) As[i] = (float)Hp[i];
      double t0 = now();
      if (k == 0) dsytrd_("U", &n, A, &n, d, e, tau, work, &lwork, &info);
      if (k == 1) dsytrd_new("U", &n, A, &n, d, e, tau, work, &lwork, &info);
      if (k == 2) dsytrd_2stage("N", "U", &n, A, &n, d2, e2, tau, hous, &lhous, work2, &lw2, &info);
      if (k == 3) ssytrd_("U", &n, As, &n, ds, es, taus, works, &lwork, &info);
      if (r >= 0) t[r] = 1e3 * (now() - t0);
      if (info) { printf("info %d in %s\n", info, name[k]); return 1; }
    }
    qsort(t, reps, 8, cmp);
    if (k == 0) { memcpy(d2, d, n * 8); }  // keep legacy for the check below
    printf("%s median %8.3f ms  (min %8.3f)\n", name[k], t[reps / 2], t[0]);
  }
  // eigenvalue check: legacy vs 2-stage tridiagonal
  memcpy(A, Hp, nn * 8); dsytrd_("U", &n, A, &n, d, e, tau, work, &lwork, &info); dsterf_(&n, d, e, &info);
  memcpy(A, Hp, nn * 8); dsytrd_2stage("N", "U", &n, A, &n, d2, e2, tau, hous, &lhous, work2, &lw2, &info);
  dsterf_(&n, d2, e2, &info);
  double mx = 0; for (lint i = 0; i < n; i++) mx = fmax(mx, fabs(d[i] - d2[i]));
  printf("max |eigenvalue(legacy) - eigenvalue(2-stage)| = %.2e\n", mx);
  return 0;
}
