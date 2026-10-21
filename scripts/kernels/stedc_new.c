// Is Accelerate's new-interface dstedc ($NEWLAPACK) faster than the legacy dstedc_ that xTB links?
//   stedc_new HS.bin [REPS]   (HS.bin from diag/dump; tridiagonal form from legacy dsytrd_ of H' = X^T H X)
//   clang -O2 stedc_new.c -framework Accelerate -o stedc_new
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
extern void dstedc_(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                    lint *, const lint *, lint *);
extern void dstedc_new(const char *, const lint *, double *, double *, double *, const lint *, double *, const lint *,
                       lint *, const lint *, lint *) __asm__("_dstedc$NEWLAPACK");

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); int reps = argc > 2 ? atoi(argv[2]) : 20;
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8), *Z = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  double *d0 = malloc(n * 8), *e0 = malloc(n * 8), *tau = malloc(n * 8), *d = malloc(n * 8), *e = malloc(n * 8);
  double *dref = malloc(n * 8), *t = malloc(reps * 8);
  lint lwork = 1 + 4 * n + n * n + 64 * n; double *work = malloc(lwork * 8);
  lint liwork = 3 + 5 * n; lint *iwork = malloc(liwork * sizeof(lint));
  dsytrd_("U", &n, H, &n, d0, e0, tau, work, &lwork, &info);
  printf("n = %d\n", n);
  for (int k = 0; k < 2; k++) {
    for (int r = -1; r < reps; r++) {
      memcpy(d, d0, n * 8); memcpy(e, e0, n * 8);
      double t0 = now();
      if (k == 0) dstedc_("I", &n, d, e, Z, &n, work, &lwork, iwork, &liwork, &info);
      else dstedc_new("I", &n, d, e, Z, &n, work, &lwork, iwork, &liwork, &info);
      if (r >= 0) t[r] = 1e3 * (now() - t0);
    }
    qsort(t, reps, 8, cmp);
    if (k == 0) memcpy(dref, d, n * 8);
    double dw = 0; for (lint i = 0; i < n; i++) dw = fmax(dw, fabs(d[i] - dref[i]));
    printf("%s median %7.3f ms (min %7.3f)  max|dw| vs legacy %.1e\n", k ? "new dstedc   " : "legacy dstedc_", t[reps / 2],
           t[0], dw);
  }
  return 0;
}
