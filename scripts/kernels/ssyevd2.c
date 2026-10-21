// Phase 4, step 1: does Accelerate's FP32 two-stage eigensolver give inaccurate eigenvalues (swift-xtb README)?
//   ssyevd2 HS.bin   (HS.bin from diag/dump)
// On H' = X^T H X: eigenvalues from legacy dsyevd_ (reference), FP32 ssyevd_ (legacy), and FP32 ssyevd_2stage
// ($NEWLAPACK, JOBZ = 'N'; JOBZ = 'V' is tried too). Max absolute and relative eigenvalue errors.
//   clang -O2 ssyevd2.c -framework Accelerate -o ssyevd2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef int lint;
extern void dpotrf_(const char *, const lint *, double *, const lint *, lint *);
extern void dtrtri_(const char *, const char *, const lint *, double *, const lint *, lint *);
extern void dtrmm_(const char *, const char *, const char *, const char *, const lint *, const lint *, const double *,
                   const double *, const lint *, double *, const lint *);
extern void dsyevd_(const char *, const char *, const lint *, double *, const lint *, double *, double *,
                    const lint *, lint *, const lint *, lint *);
extern void ssyevd_(const char *, const char *, const lint *, float *, const lint *, float *, float *,
                    const lint *, lint *, const lint *, lint *);
extern void ssyevd_2stage(const char *, const char *, const lint *, float *, const lint *, float *, float *,
                          const lint *, lint *, const lint *, lint *) __asm__("_ssyevd_2stage$NEWLAPACK");

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  int n4; fread(&n4, 4, 1, f); lint n = n4; size_t nn = (size_t)n * n;
  double *H = malloc(nn * 8), *S = malloc(nn * 8), *A = malloc(nn * 8), *w = malloc(n * 8);
  fread(H, 8, nn, f); fread(S, 8, nn, f); fclose(f);
  lint info; double one = 1.0;
  dpotrf_("U", &n, S, &n, &info);
  for (lint j = 0; j < n; j++) for (lint i = j + 1; i < n; i++) S[i + j * n] = 0.0;
  dtrtri_("U", "N", &n, S, &n, &info);
  dtrmm_("R", "U", "N", "N", &n, &n, &one, S, &n, H, &n);
  dtrmm_("L", "U", "T", "N", &n, &n, &one, S, &n, H, &n);
  lint lwork = 1 + 6 * n + 2 * n * n + 100000, liwork = 3 + 5 * n;
  double *work = malloc(lwork * 8); float *works = malloc(lwork * 4); lint *iwork = malloc(liwork * sizeof(lint));
  memcpy(A, H, nn * 8);
  dsyevd_("N", "U", &n, A, &n, w, work, &lwork, iwork, &liwork, &info);
  float *As = malloc(nn * 4), *ws = malloc(n * 4);
  const char *name[2] = {"FP32 ssyevd_ (legacy)        ", "FP32 ssyevd_2stage (new, N)  "};
  for (int k = 0; k < 2; k++) {
    for (size_t i = 0; i < nn; i++) As[i] = (float)H[i];
    if (k == 0) ssyevd_("N", "U", &n, As, &n, ws, works, &lwork, iwork, &liwork, &info);
    else ssyevd_2stage("N", "U", &n, As, &n, ws, works, &lwork, iwork, &liwork, &info);
    double ea = 0, er = 0, scale = fmax(fabs(w[0]), fabs(w[n - 1]));
    for (lint i = 0; i < n; i++) { ea = fmax(ea, fabs(ws[i] - w[i])); er = fmax(er, fabs(ws[i] - w[i]) / scale); }
    printf("n = %d  %s info %d  max|de| %.2e Eh  max|de|/max|e| %.2e\n", n, name[k], info, ea, er);
  }
  for (size_t i = 0; i < nn; i++) As[i] = (float)H[i];
  ssyevd_2stage("V", "U", &n, As, &n, ws, works, &lwork, iwork, &liwork, &info);
  printf("n = %d  FP32 ssyevd_2stage with eigenvectors (V): info %d%s\n", n, info, info < 0 ? " (refused)" : "");
  return 0;
}
