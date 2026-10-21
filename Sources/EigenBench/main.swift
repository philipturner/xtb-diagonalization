// End-to-end symmetric eigensolve of dumped GFN2 matrices, FP64, by stage: reduction to tridiagonal form, divide and
// conquer (Accelerate dstedc), back-transformation of the first m eigenvectors.
//   1-stage sym     Swift tridiagonalize (symmetric) + dstedc + dormtr: the SCC's eigensolve today
//   1-stage unsym   the same without exploiting symmetry: the baseline for the band-reduction work (BRIEF Phase 4)
//   2-stage b       reduceToBand + chaseBulges + dstedc + backTransformBulges + backTransformBand
//
//   swift build -c release --disable-sandbox
//   VECLIB_MAXIMUM_THREADS=1 .build/release/eigen-bench [--b 8,16,32] [--vectors 1.0] [--reps 5] HS.bin ...
//
// --unblocked-bulges: stage 2's back-transformation one reflector at a time (the reference) instead of blocked.
// HS.bin: dump from the xtb branch diag/dump (int32 n, H(n,n), S(n,n), ...); the eigenproblem is H' = X^T H X
// (S = U^T U, X = U^-1). --vectors: the fraction of eigenvectors (lowest first) to back-transform. Variants run
// round-robin, one call each per round after one untimed round; medians per stage. Checks against dsyevd
// eigenvalues, the residual max|H' X - X diag(w)| and the orthogonality max|X^T X - I| of the back-transformed X.
// Buffers are page-aligned (Accelerate's full-matrix kernels are slow on misaligned ones).
import CLAPACK
import Darwin
import Diagonalization
import Foundation

func now() -> Double {
  var t = timespec()
  clock_gettime(CLOCK_MONOTONIC_RAW, &t)
  return Double(t.tv_sec) + 1e-9 * Double(t.tv_nsec)
}

/// A page-aligned, zero-initialized buffer that lives for the whole run.
func buffer(_ count: Int) -> UnsafeMutablePointer<Double> {
  let p = UnsafeMutableRawPointer.allocate(byteCount: max(count, 1) * 8, alignment: 16384)
    .bindMemory(to: Double.self, capacity: max(count, 1))
  p.initialize(repeating: 0, count: max(count, 1))
  return p
}

var bandwidths = [8, 16, 32]
var fraction = 1.0
var reps = 5
var blockedBulges = true
var files: [String] = []
var argi = 1
let args = CommandLine.arguments
while argi < args.count {
  switch args[argi] {
  case "--b": bandwidths = args[argi + 1].split(separator: ",").map { Int($0)! }; argi += 2
  case "--vectors": fraction = Double(args[argi + 1])!; argi += 2
  case "--reps": reps = Int(args[argi + 1])!; argi += 2
  case "--unblocked-bulges": blockedBulges = false; argi += 1
  default: files.append(args[argi]); argi += 1
  }
}

/// H' = X^T H X from a dump, column-major n x n.
func readMatrix(_ path: String) -> (Int, UnsafeMutablePointer<Double>) {
  guard let f = fopen(path, "rb") else { fatalError("cannot open \(path)") }
  defer { fclose(f) }
  var n32: Int32 = 0
  fread(&n32, 4, 1, f)
  let n = Int(n32)
  let H = buffer(n * n), S = buffer(n * n)
  fread(H, 8, n * n, f)
  fread(S, 8, n * n, f)
  var nn = n32, info: Int32 = 0, one = 1.0
  dpotrf_("U", &nn, S, &nn, &info)
  for j in 0..<n { for i in (j + 1)..<n { S[i + j * n] = 0 } }
  dtrtri_("U", "N", &nn, S, &nn, &info)
  dtrmm_("R", "U", "N", "N", &nn, &nn, &one, S, &nn, H, &nn)
  dtrmm_("L", "U", "T", "N", &nn, &nn, &one, S, &nn, H, &nn)
  S.deallocate()
  return (n, H)
}

/// One eigensolver variant: run() fills w (eigenvalues) and the first m columns of x, recording stage times (s).
struct Variant {
  var name: String
  var stages: [String]
  var run: (inout [Double]) -> Void
}

for path in files {
  let (n, H) = readMatrix(path)
  let m = max(1, Int((fraction * Double(n)).rounded()))
  print("\(path)  n = \(n), back-transforming m = \(m) eigenvectors, median of \(reps)")

  // Reference eigenvalues.
  let wRef = buffer(n)
  do {
    let A = buffer(n * n)
    A.update(from: H, count: n * n)
    var nn = Int32(n), lwork = Int32(1 + 6 * n + 2 * n * n), liwork = Int32(3 + 5 * n), info: Int32 = 0
    let work = buffer(Int(lwork))
    let iwork = UnsafeMutablePointer<Int32>.allocate(capacity: Int(liwork))
    dsyevd_("N", "U", &nn, A, &nn, wRef, work, &lwork, iwork, &liwork, &info)
    A.deallocate(); work.deallocate(); iwork.deallocate()
  }

  // Shared buffers.
  let A = buffer(n * n), Z = buffer(n * n), w = buffer(n), e = buffer(n), tau = buffer(n)
  let bmax = bandwidths.max() ?? 1
  let lwork = max(1 + 4 * n + n * n, 64 * n, bandReductionWorkspace(n, bmax), bulgeBackTransformWorkspace(bmax, n),
                  64 * n + 100_000)
  let work = buffer(lwork)
  let liwork = 3 + 5 * n
  let iwork = UnsafeMutablePointer<Int32>.allocate(capacity: liwork)
  let v1 = buffer(n * n), t1 = buffer(bmax * n), v2 = buffer(n * n)

  func stedc() {
    var nn = Int32(n), ldz = Int32(n), lw = Int32(1 + 4 * n + n * n), liw = Int32(liwork), info: Int32 = 0
    dstedc_("I", &nn, w, e, Z, &ldz, work, &lw, iwork, &liw, &info)
    if info != 0 { fatalError("dstedc info \(info)") }
  }
  func ormtr() {
    var nn = Int32(n), mm = Int32(m), lda = Int32(n), ldz = Int32(n), lw = Int32(lwork), info: Int32 = 0
    dormtr_("L", "U", "N", &nn, &mm, A, &lda, tau, Z, &ldz, work, &lw, &info)
    if info != 0 { fatalError("dormtr info \(info)") }
  }
  func timed(_ t: inout [Double], _ body: () -> Void) {
    let t0 = now(); body(); t.append(now() - t0)
  }

  var variants: [Variant] = []
  for symmetric in [true, false] {
    variants.append(Variant(
      name: symmetric ? "1-stage sym" : "1-stage unsym", stages: ["tridiag", "dstedc", "ormtr"],
      run: { t in
        A.update(from: H, count: n * n)
        timed(&t) { tridiagonalize(n, A, n, w, e, tau, work, symmetric: symmetric) }
        timed(&t) { stedc() }
        timed(&t) { ormtr() }
      }))
  }
  for b in bandwidths {
    variants.append(Variant(
      name: "2-stage b=\(b)", stages: ["band", "bulge", "dstedc", "bt-bulge", "bt-band"],
      run: { t in
        A.update(from: H, count: n * n)
        timed(&t) { reduceToBand(n, b, A, n, v1, t1, work) }
        timed(&t) { chaseBulges(n, b, A, n, v2, w, e, work) }
        timed(&t) { stedc() }
        timed(&t) {
          if blockedBulges { backTransformBulgesBlocked(n, b, v2, Z, n, m, work) }
          else { backTransformBulges(n, b, v2, Z, n, m) }
        }
        timed(&t) { backTransformBand(n, b, v1, t1, Z, n, m, work) }
      }))
  }

  var times = Array(repeating: [[Double]](), count: variants.count)
  var checks = Array(repeating: "", count: variants.count)
  for r in -1..<reps {
    for (k, v) in variants.enumerated() {
      var t: [Double] = []
      v.run(&t)
      if r >= 0 { times[k].append(t) }
      if r == -1 {
        // Accuracy of this variant: eigenvalues, residual and orthogonality of X = Z[:, 0 ..< m].
        var dw = 0.0, scale = 0.0
        for i in 0..<n { dw = max(dw, abs(w[i] - wRef[i])); scale = max(scale, abs(wRef[i])) }
        let R = buffer(n * m), G = buffer(m * m)
        var nn = Int32(n), mm = Int32(m), one = 1.0, zero = 0.0
        dgemm_("N", "N", &nn, &mm, &nn, &one, H, &nn, Z, &nn, &zero, R, &nn)
        var res = 0.0
        for j in 0..<m { for i in 0..<n { res = max(res, abs(R[i + j * n] - w[j] * Z[i + j * n])) } }
        dgemm_("T", "N", &mm, &mm, &nn, &one, Z, &nn, Z, &nn, &zero, G, &mm)
        var orth = 0.0
        for j in 0..<m { for i in 0..<m { orth = max(orth, abs(G[i + j * m] - (i == j ? 1 : 0))) } }
        checks[k] = String(format: "eigenvalues %.1e Eh, residual %.1e, orthogonality %.1e",
                           dw, res / scale, orth)
        R.deallocate(); G.deallocate()
      }
    }
  }
  let base = times[1].map { $0.reduce(0, +) }.sorted()[reps / 2]
  for (k, v) in variants.enumerated() {
    let per = (0..<v.stages.count).map { s in times[k].map { $0[s] }.sorted()[reps / 2] }
    let total = times[k].map { $0.reduce(0, +) }.sorted()[reps / 2]
    var line = "  " + v.name.padding(toLength: 14, withPad: " ", startingAt: 0)
    line += String(format: " total %8.2f ms (%.2fx unsym) |", 1e3 * total, total / base)
    for (s, name) in v.stages.enumerated() { line += " \(name) " + String(format: "%.2f", 1e3 * per[s]) }
    print(line)
    print("                 \(checks[k])")
  }
  for p in [A, Z, w, e, tau, work, v1, t1, v2, wRef, H] { p.deallocate() }
  iwork.deallocate()
}
