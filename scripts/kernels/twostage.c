// Does Accelerate's new LAPACK ($NEWLAPACK) have a working 2-stage symmetric eigensolver?
//   twostage HS.bin [REPS]
// HS.bin: dump from diag/dump (int32 n, H(n,n), S(n,n), ...). Forms H' = X^T H X with X = U^-1 (S = U^T U), then
// compares eigenvalues and timings of: legacy dsyevd_ (what xTB links), new dsyevd, new dsyevd_2stage (JOBZ N and V),
// and FP32 ssyevd_2stage (JOBZ N) against legacy dsyevd.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int lint;
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *,
                   const double *, const double *, const lint *, double *, const lint *);
extern void dsyevd_(const char *, const char *, const lint *, double *, const lint *, double *, double *,
                    const lint *, lint *, const lint *, lint *);
extern void dsyevd_new(const char *, const char *, const lint *, double *, const lint *, double *, double *,
                       const lint *, lint *, const lint *, lint *) __asm__("_dsyevd$NEWLAPACK");
extern void dsyevd_2stage(const char *, const char *, const lint *, double *, const lint *, double *, double *,
                          const lint *, lint *, const lint *, lint *) __asm__("_dsyevd_2stage$NEWLAPACK");
extern void ssyevd_2stage(const char *, const char *, const lint *, float *, const lint *, float *, float *,
                          const lint *, lint *, const lint *, lint *) __asm__("_ssyevd_2stage$NEWLAPACK");

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb"); int reps = argc > 2 ? atoi(argv[2]) : 10;
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8), *Hp = malloc(nn * 8), *A = malloc(nn * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);  // S now holds X = U^-1
  memcpy(Hp, H, nn * 8);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, Hp, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, Hp, &n);

  double *eref = malloc(n * 8), *e = malloc(n * 8), *t = malloc(reps * 8);
  float *As = malloc(nn * 4), *es = malloc(n * 4);
  lint lwork = -1, liwork = -1, iq; double wq; float wqs;
  // generous workspace for every driver; query the 2-stage ones, whose needs differ
  lint lw = 1 + 6 * n + 2 * n * n, liw = 3 + 5 * n;
  memcpy(A, Hp, nn * 8);
  dsyevd_2stage("V", "U", &n, A, &n, e, &wq, &lwork, &iq, &liwork, &info);
  printf("n = %d; dsyevd_2stage JOBZ=V workspace query: info %d, lwork %.0f, liwork %d\n", n, info, wq, iq);
  if (info == 0 && wq > lw) lw = (lint)wq;
  dsyevd_2stage("N", "U", &n, A, &n, e, &wq, &lwork, &iq, &liwork, &info);
  printf("dsyevd_2stage JOBZ=N workspace query: info %d, lwork %.0f, liwork %d\n", info, wq, iq);
  if (info == 0 && wq > lw) lw = (lint)wq;
  if (info == 0 && iq > liw) liw = iq;
  ssyevd_2stage("N", "U", &n, As, &n, es, &wqs, &lwork, &iq, &liwork, &info);
  lint lws = (lint)wqs > lw ? (lint)wqs : lw, liws = iq > liw ? iq : liw;
  double *work = malloc((size_t)lw * 8); lint *iwork = malloc(liws * sizeof(lint));
  float *works = malloc((size_t)lws * 4);

  const char *names[] = {"legacy dsyevd_ (V)", "new dsyevd (V)", "new dsyevd_2stage (N)", "new dsyevd_2stage (V)",
                         "new ssyevd_2stage (N), FP32", "legacy dsyevd_ (N)"};
  for (int k = 0; k < 6; k++) {
    for (int r = 0; r <= reps; r++) {
      memcpy(A, Hp, nn * 8);
      if (k == 4) for (size_t i = 0; i < nn; i++) As[i] = (float)Hp[i];
      double t0 = now();
      switch (k) {
        case 0: dsyevd_("V", "U", &n, A, &n, e, work, &lw, iwork, &liw, &info); break;
        case 1: dsyevd_new("V", "U", &n, A, &n, e, work, &lw, iwork, &liw, &info); break;
        case 2: dsyevd_2stage("N", "U", &n, A, &n, e, work, &lw, iwork, &liw, &info); break;
        case 3: dsyevd_2stage("V", "U", &n, A, &n, e, work, &lw, iwork, &liw, &info); break;
        case 4: ssyevd_2stage("N", "U", &n, As, &n, es, works, &lws, iwork, &liws, &info); break;
        case 5: dsyevd_("N", "U", &n, A, &n, e, work, &lw, iwork, &liw, &info); break;
      }
      if (r) t[r - 1] = 1e3 * (now() - t0);
      if (info != 0) break;
    }
    if (info != 0) { printf("%-30s info %d (not available / failed)\n", names[k], info); continue; }
    if (k == 4) for (lint i = 0; i < n; i++) e[i] = es[i];
    if (k == 0) memcpy(eref, e, n * 8);
    double de = 0, emax = 0;
    for (lint i = 0; i < n; i++) { de = fmax(de, fabs(e[i] - eref[i])); emax = fmax(emax, fabs(eref[i])); }
    double res = 0;
    if (k == 1 || k == 3 || k == 0) {  // eigenvector residual max |H' v - e v| over all pairs
      for (lint j = 0; j < n; j++) {
        for (lint i = 0; i < n; i++) {
          double s = 0;
          for (lint l = 0; l < n; l++) s += Hp[i + l * n] * A[l + j * n];
          res = fmax(res, fabs(s - e[j] * A[i + j * n]));
        }
      }
    }
    qsort(t, reps, 8, cmp);
    printf("%-30s median %8.3f ms (min %8.3f)  max|de| vs legacy %.2e (rel %.1e)%s", names[k], t[reps / 2], t[0],
           de, de / emax, (k == 1 || k == 3 || k == 0) ? "" : "\n");
    if (k == 1 || k == 3 || k == 0) printf("  residual %.2e\n", res);
  }
  return 0;
}
