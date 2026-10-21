// Cost of xtb_dsymmetrize_pad (sytrd.c): copy the upper triangle of an n x n matrix into a full symmetric matrix with
// the leading dimension padded to 8 doubles, as the SCC eigensolve does every cycle with XTB_SCC_TRD=full.
//   clang -O2 symcopy.c -o symcopy && ./symcopy build/xtb-build-diag/libxtb.dylib
// Median of 501 calls per size, page-aligned buffers.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef void (*symfn)(const int *, const double *, const int *, double *, const int *);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
int main(int argc, char **argv) {
  void *h = dlopen(argv[1], RTLD_NOW); if (!h) { printf("%s\n", dlerror()); return 1; }
  symfn f = (symfn)dlsym(h, "xtb_dsymmetrize_pad"); if (!f) { printf("no xtb_dsymmetrize_pad\n"); return 1; }
  int sizes[] = {124, 412, 419, 448, 619, 776};
  printf("    n    ldp   copy us   GB/s (bytes read + written)\n");
  for (int s = 0; s < 6; s++) {
    int n = sizes[s], ldp = (n + 7) / 8 * 8, reps = 501;
    double *src, *dst, *t = malloc(reps * 8);
    posix_memalign((void **)&src, 16384, (size_t)n * n * 8); posix_memalign((void **)&dst, 16384, (size_t)ldp * n * 8);
    for (size_t k = 0; k < (size_t)n * n; k++) src[k] = k;
    for (int r = -50; r < reps; r++) { double t0 = now(); f(&n, src, &n, dst, &ldp); if (r >= 0) t[r] = now() - t0; }
    qsort(t, reps, 8, cmp);
    double bytes = 8.0 * n * (n + 1) / 2 + 8.0 * n * n;
    printf("%5d %6d %9.1f %8.1f\n", n, ldp, 1e6 * t[reps / 2], bytes / t[reps / 2] / 1e9);
    free(src); free(dst); free(t);
  }
  return 0;
}
