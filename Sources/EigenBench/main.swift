// End-to-end symmetric eigensolve of dumped GFN2 matrices, FP64 or FP32, by stage: reduction to tridiagonal form,
// divide and conquer (Accelerate xstedc), back-transformation of the first m eigenvectors.
//   1-stage sym     Swift tridiagonalize (symmetric) + xstedc + xormtr: the SCC's eigensolve today
//   1-stage unsym   the same without exploiting symmetry: the baseline for the band-reduction work (BRIEF Phase 4)
//   2-stage b       reduceToBand + chaseBulges + xstedc + backTransformBulges + backTransformBand
//
//   swift build -c release --disable-sandbox
//   VECLIB_MAXIMUM_THREADS=1 .build/release/eigen-bench [--fp32] [--b 8,16,32] [--vectors 1.0] [--reps 5] HS.bin ...
//
// --fp32: every variant in single precision (the matrix is rounded to FP32 first). Accuracy is still measured in FP64
// against dsyevd of the FP64 matrix; the line "Accelerate ssyevd" gives FP32's own floor on the same matrix.
// --unblocked-bulges: stage 2's back-transformation one reflector at a time (the reference) instead of blocked.
// --sbr 64-16,32: successive band reduction variants (SuccessiveBandReduction.swift), here 3-stage n-1 -> 64 -> 16 -> 1
// and 2-stage n-1 -> 32 -> 1; each stage and back-transformation timed. --whole: SBR updates span the whole matrix
// instead of a window (cross-check). The check line adds each SBR stage's largest entry outside its target band.
// --fast-tail: the SBR variants' last stage (b -> 1) by TwoStage.swift's bulge chase and blocked back-transformation.
// --procs P [--jobs K]: throughput instead of latency. For each variant, P copies of this program (one file only)
// start together and each runs the variant K times (default 20) after a warm-up; reports jobs per second as a work
// queue sees it (sum over processes of K / elapsed) and the median job time. Run with VECLIB_MAXIMUM_THREADS=1.
// "random:N" in place of HS.bin: a random symmetric N x N matrix.
// HS.bin: dump from the xtb branch diag/dump (int32 n, H(n,n), S(n,n), ...); the eigenproblem is H' = X^T H X
// (S = U^T U, X = U^-1). --vectors: the fraction of eigenvectors (lowest first) to back-transform. Variants run
// round-robin, one call each per round after one untimed round; medians per stage. Checks against dsyevd
// eigenvalues, the residual max|H' X - X diag(w)| / max|w| and the orthogonality max|X^T X - I| of the
// back-transformed X, all in FP64. Buffers are page-aligned (Accelerate's full-matrix kernels are slow on misaligned
// ones).
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
func buffer<R: BinaryFloatingPoint>(_ count: Int, _: R.Type = R.self) -> UnsafeMutablePointer<R> {
  let p = UnsafeMutableRawPointer.allocate(byteCount: max(count, 1) * MemoryLayout<R>.stride, alignment: 16384)
    .bindMemory(to: R.self, capacity: max(count, 1))
  p.initialize(repeating: 0, count: max(count, 1))
  return p
}

/// The LAPACK calls of the 1-stage path and the divide and conquer, per precision.
protocol EigenScalar: BandReductionScalar {
  static func stedc(_ n: Int, _ d: UnsafeMutablePointer<Self>, _ e: UnsafeMutablePointer<Self>,
                    _ z: UnsafeMutablePointer<Self>, _ ldz: Int, _ work: UnsafeMutablePointer<Self>, _ lwork: Int,
                    _ iwork: UnsafeMutablePointer<Int32>, _ liwork: Int) -> Int32
  static func ormtr(_ n: Int, _ m: Int, _ a: UnsafePointer<Self>, _ lda: Int, _ tau: UnsafePointer<Self>,
                    _ z: UnsafeMutablePointer<Self>, _ ldz: Int, _ work: UnsafeMutablePointer<Self>,
                    _ lwork: Int) -> Int32
  /// Eigenvalues only.
  static func syevd(_ n: Int, _ a: UnsafeMutablePointer<Self>, _ lda: Int, _ w: UnsafeMutablePointer<Self>,
                    _ work: UnsafeMutablePointer<Self>, _ lwork: Int, _ iwork: UnsafeMutablePointer<Int32>,
                    _ liwork: Int) -> Int32
}

extension Double: EigenScalar {
  static func stedc(_ n: Int, _ d: UnsafeMutablePointer<Double>, _ e: UnsafeMutablePointer<Double>,
                    _ z: UnsafeMutablePointer<Double>, _ ldz: Int, _ work: UnsafeMutablePointer<Double>, _ lwork: Int,
                    _ iwork: UnsafeMutablePointer<Int32>, _ liwork: Int) -> Int32 {
    var nn = Int32(n), ldz = Int32(ldz), lw = Int32(lwork), liw = Int32(liwork), info: Int32 = 0
    dstedc_("I", &nn, d, e, z, &ldz, work, &lw, iwork, &liw, &info)
    return info
  }
  static func ormtr(_ n: Int, _ m: Int, _ a: UnsafePointer<Double>, _ lda: Int, _ tau: UnsafePointer<Double>,
                    _ z: UnsafeMutablePointer<Double>, _ ldz: Int, _ work: UnsafeMutablePointer<Double>,
                    _ lwork: Int) -> Int32 {
    var nn = Int32(n), mm = Int32(m), lda = Int32(lda), ldz = Int32(ldz), lw = Int32(lwork), info: Int32 = 0
    dormtr_("L", "U", "N", &nn, &mm, a, &lda, tau, z, &ldz, work, &lw, &info)
    return info
  }
  static func syevd(_ n: Int, _ a: UnsafeMutablePointer<Double>, _ lda: Int, _ w: UnsafeMutablePointer<Double>,
                    _ work: UnsafeMutablePointer<Double>, _ lwork: Int, _ iwork: UnsafeMutablePointer<Int32>,
                    _ liwork: Int) -> Int32 {
    var nn = Int32(n), lda = Int32(lda), lw = Int32(lwork), liw = Int32(liwork), info: Int32 = 0
    dsyevd_("N", "U", &nn, a, &lda, w, work, &lw, iwork, &liw, &info)
    return info
  }
}

extension Float: EigenScalar {
  static func stedc(_ n: Int, _ d: UnsafeMutablePointer<Float>, _ e: UnsafeMutablePointer<Float>,
                    _ z: UnsafeMutablePointer<Float>, _ ldz: Int, _ work: UnsafeMutablePointer<Float>, _ lwork: Int,
                    _ iwork: UnsafeMutablePointer<Int32>, _ liwork: Int) -> Int32 {
    var nn = Int32(n), ldz = Int32(ldz), lw = Int32(lwork), liw = Int32(liwork), info: Int32 = 0
    sstedc_("I", &nn, d, e, z, &ldz, work, &lw, iwork, &liw, &info)
    return info
  }
  static func ormtr(_ n: Int, _ m: Int, _ a: UnsafePointer<Float>, _ lda: Int, _ tau: UnsafePointer<Float>,
                    _ z: UnsafeMutablePointer<Float>, _ ldz: Int, _ work: UnsafeMutablePointer<Float>,
                    _ lwork: Int) -> Int32 {
    var nn = Int32(n), mm = Int32(m), lda = Int32(lda), ldz = Int32(ldz), lw = Int32(lwork), info: Int32 = 0
    sormtr_("L", "U", "N", &nn, &mm, a, &lda, tau, z, &ldz, work, &lw, &info)
    return info
  }
  static func syevd(_ n: Int, _ a: UnsafeMutablePointer<Float>, _ lda: Int, _ w: UnsafeMutablePointer<Float>,
                    _ work: UnsafeMutablePointer<Float>, _ lwork: Int, _ iwork: UnsafeMutablePointer<Int32>,
                    _ liwork: Int) -> Int32 {
    var nn = Int32(n), lda = Int32(lda), lw = Int32(lwork), liw = Int32(liwork), info: Int32 = 0
    ssyevd_("N", "U", &nn, a, &lda, w, work, &lw, iwork, &liw, &info)
    return info
  }
}

var fp32 = false
var bandwidths = [8, 16, 32]
var fraction = 1.0
var reps = 5
var blockedBulges = true
var sbrConfigs: [[Int]] = []
var windowed = true
var fastTail = false
var procs = 0
var jobs = 20
var workerVariant: Int? = nil
var workerStart = 0.0
var files: [String] = []
var argi = 1
let args = CommandLine.arguments
while argi < args.count {
  switch args[argi] {
  case "--fp32": fp32 = true; argi += 1
  case "--b": bandwidths = args[argi + 1].split(separator: ",").map { Int($0)! }; argi += 2
  case "--vectors": fraction = Double(args[argi + 1])!; argi += 2
  case "--reps": reps = Int(args[argi + 1])!; argi += 2
  case "--unblocked-bulges": blockedBulges = false; argi += 1
  case "--sbr":
    sbrConfigs = args[argi + 1].split(separator: ",").map { $0.split(separator: "-").map { Int($0)! } }; argi += 2
  case "--whole": windowed = false; argi += 1
  case "--fast-tail": fastTail = true; argi += 1
  case "--procs": procs = Int(args[argi + 1])!; argi += 2
  case "--jobs": jobs = Int(args[argi + 1])!; argi += 2
  case "--worker": workerVariant = Int(args[argi + 1])!; workerStart = Double(args[argi + 2])!; argi += 3
  default: files.append(args[argi]); argi += 1
  }
}

/// H' = X^T H X from a dump, column-major n x n; "random:N" makes a random symmetric N x N matrix instead (entries
/// uniform in [-1, 1], fixed seed), for edge cases of small and odd sizes.
func readMatrix(_ path: String) -> (Int, UnsafeMutablePointer<Double>) {
  if path.hasPrefix("random:") {
    let n = Int(path.dropFirst(7))!
    let H: UnsafeMutablePointer<Double> = buffer(n * n)
    srand48(n)
    for j in 0..<n { for i in 0...j { let x = 2 * drand48() - 1; H[i + j * n] = x; H[j + i * n] = x } }
    return (n, H)
  }
  guard let f = fopen(path, "rb") else { fatalError("cannot open \(path)") }
  defer { fclose(f) }
  var n32: Int32 = 0
  fread(&n32, 4, 1, f)
  let n = Int(n32)
  let H: UnsafeMutablePointer<Double> = buffer(n * n), S: UnsafeMutablePointer<Double> = buffer(n * n)
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

/// One eigensolver variant: run() fills w (eigenvalues) and the first m columns of Z, recording stage times (s).
struct Variant {
  var name: String
  var stages: [String]
  var run: (inout [Double]) -> Void
}

/// Max |w - wRef|, residual max|H X - X diag(w)| / max|wRef| and orthogonality max|X^T X - I|, in FP64, for the first
/// m columns of z (n x n).
func accuracy<Real: BinaryFloatingPoint>(
  _ n: Int, _ m: Int, _ H: UnsafePointer<Double>, _ wRef: UnsafePointer<Double>, _ w: UnsafePointer<Real>,
  _ z: UnsafePointer<Real>
) -> (Double, Double, Double) {
  var dw = 0.0, scale = 0.0
  for i in 0..<n { dw = max(dw, abs(Double(w[i]) - wRef[i])); scale = max(scale, abs(wRef[i])) }
  let X: UnsafeMutablePointer<Double> = buffer(n * m)
  for k in 0..<(n * m) { X[k] = Double(z[k]) }
  let R: UnsafeMutablePointer<Double> = buffer(n * m), G: UnsafeMutablePointer<Double> = buffer(m * m)
  var nn = Int32(n), mm = Int32(m), one = 1.0, zero = 0.0
  dgemm_("N", "N", &nn, &mm, &nn, &one, H, &nn, X, &nn, &zero, R, &nn)
  var res = 0.0
  for j in 0..<m { for i in 0..<n { res = max(res, abs(R[i + j * n] - Double(w[j]) * X[i + j * n])) } }
  dgemm_("T", "N", &mm, &mm, &nn, &one, X, &nn, X, &nn, &zero, G, &mm)
  var orth = 0.0
  for j in 0..<m { for i in 0..<m { orth = max(orth, abs(G[i + j * m] - (i == j ? 1 : 0))) } }
  X.deallocate(); R.deallocate(); G.deallocate()
  return (dw, res / scale, orth)
}

/// Runs every variant on one matrix in precision Real: latency per stage with accuracy checks, or (procs > 0 or in
/// a worker) throughput.
func benchmark<Real: EigenScalar>(_ path: String, _: Real.Type) {
  let (n, H) = readMatrix(path)
  let m = max(1, Int((fraction * Double(n)).rounded()))
  let Hr: UnsafeMutablePointer<Real> = buffer(n * n)
  for k in 0..<(n * n) { Hr[k] = Real(H[k]) }

  // Shared buffers.
  let A: UnsafeMutablePointer<Real> = buffer(n * n), Z: UnsafeMutablePointer<Real> = buffer(n * n)
  let w: UnsafeMutablePointer<Real> = buffer(n), e: UnsafeMutablePointer<Real> = buffer(n)
  let tau: UnsafeMutablePointer<Real> = buffer(n)
  let bmax = bandwidths.max() ?? 1
  let lwork = max(1 + 4 * n + n * n, 64 * n, bandReductionWorkspace(n, bmax), bulgeBackTransformWorkspace(bmax, n),
                  64 * n + 100_000, 1 + 6 * n + 2 * n * n)
  let work: UnsafeMutablePointer<Real> = buffer(lwork)
  let liwork = 3 + 5 * n
  let iwork = UnsafeMutablePointer<Int32>.allocate(capacity: liwork)
  let v1: UnsafeMutablePointer<Real> = buffer(n * n), t1: UnsafeMutablePointer<Real> = buffer(bmax * n)
  let v2: UnsafeMutablePointer<Real> = buffer(n * n)

  func stedc() {
    let info = Real.stedc(n, w, e, Z, n, work, 1 + 4 * n + n * n, iwork, liwork)
    if info != 0 { fatalError("stedc info \(info)") }
  }
  func ormtr() {
    let info = Real.ormtr(n, m, A, n, tau, Z, n, work, lwork)
    if info != 0 { fatalError("ormtr info \(info)") }
  }
  func timed(_ t: inout [Double], _ body: () -> Void) {
    let t0 = now(); body(); t.append(now() - t0)
  }

  var variants: [Variant] = []
  for symmetric in [true, false] {
    variants.append(Variant(
      name: symmetric ? "1-stage sym" : "1-stage unsym", stages: ["tridiag", "stedc", "ormtr"],
      run: { t in
        A.update(from: Hr, count: n * n)
        timed(&t) { tridiagonalize(n, A, n, w, e, tau, work, symmetric: symmetric) }
        timed(&t) { stedc() }
        timed(&t) { ormtr() }
      }))
  }
  for b in bandwidths where b < n - 1 {
    variants.append(Variant(
      name: "2-stage b=\(b)", stages: ["band", "bulge", "stedc", "bt-bulge", "bt-band"],
      run: { t in
        A.update(from: Hr, count: n * n)
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
  var leftovers = Array(repeating: "", count: variants.count + sbrConfigs.count)
  for config in sbrConfigs {
    // Bandwidths that don't fit this n are dropped (as successiveBandReduction does).
    var chain = [n - 1]
    for b in config + [1] where b < chain.last! && (b == 1 || b < n - 1) { chain.append(b) }
    guard chain.count >= 2, n >= 2 else { continue }
    let names = (1..<chain.count).map { "\(chain[$0 - 1] == n - 1 ? "n" : String(chain[$0 - 1]))>\(chain[$0])" }
    let vi = variants.count
    variants.append(Variant(
      name: "SBR " + chain.dropFirst().dropLast().map(String.init).joined(separator: "-"),
      stages: names + ["stedc"] + names.reversed().map { "bt" + $0 },
      run: { t in
        A.update(from: Hr, count: n * n)
        var stages: [BandReductionStage<Real>] = []
        let tail = fastTail && chain.count >= 3 ? chain[chain.count - 2] : 0
        for s in 1..<chain.count {
          timed(&t) {
            if tail > 0 && s == chain.count - 1 {
              chaseBulges(n, tail, A, n, v2, w, e, work)
            } else {
              stages.append(reduceBandwidth(n, A, n, bandIn: chain[s - 1], bandOut: chain[s], panel: chain[s],
                                            windowed: windowed))
            }
          }
        }
        for i in 0..<n { w[i] = A[i + i * n] }
        for i in 0..<(n - 1) { e[i] = A[(i + 1) + i * n] }
        timed(&t) { stedc() }
        if tail > 0 { timed(&t) { backTransformBulgesBlocked(n, tail, v2, Z, n, m, work) } }
        for stage in stages.reversed() { timed(&t) { backTransform(stage, Z, n, m) } }
        leftovers[vi] = "; outside band per stage " + stages.map { String(format: "%.0e", Double($0.leftover)) }
          .joined(separator: " ")
      }))
  }

  if let k = workerVariant {
    // Throughput worker: warm up, wait for the common start, run the variant `jobs` times, report.
    var t: [Double] = []
    variants[k].run(&t)
    while Date().timeIntervalSince1970 < workerStart { usleep(500) }
    let begin = Date().timeIntervalSince1970
    var per: [Double] = []
    for _ in 0..<jobs {
      let t0 = now(); t = []; variants[k].run(&t); per.append(now() - t0)
    }
    let end = Date().timeIntervalSince1970
    print("WORKER \(begin) \(end) " + per.map { String($0) }.joined(separator: " "))
    return
  }

  let precision = Real.self == Float.self ? "FP32" : "FP64"
  if procs > 0 {
    print("\(path)  n = \(n), \(precision), m = \(m), throughput: \(procs) processes x \(jobs) jobs each")
    for (k, v) in variants.enumerated() {
      let start = Date().timeIntervalSince1970 + 3 + 0.3 * Double(procs)
      var workerArgs = Array(args.dropFirst())
      if let i = workerArgs.firstIndex(of: "--procs") { workerArgs.removeSubrange(i...(i + 1)) }
      workerArgs = workerArgs.filter { !files.contains($0) || $0 == path } + ["--worker", String(k), String(start)]
      let pipes = (0..<procs).map { _ in Pipe() }
      let processes = pipes.map { pipe -> Process in
        let p = Process()
        p.executableURL = URL(fileURLWithPath: args[0])
        p.arguments = workerArgs
        p.standardOutput = pipe
        try! p.run()
        return p
      }
      var rate = 0.0, all: [Double] = [], spread: [Double] = []
      for (p, pipe) in zip(processes, pipes) {
        let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)!
        p.waitUntilExit()
        guard let line = out.split(separator: "\n").first(where: { $0.hasPrefix("WORKER") }) else {
          fatalError("worker failed: \(out)")
        }
        let f = line.split(separator: " ").dropFirst().map { Double($0)! }
        let elapsed = f[1] - f[0]
        rate += Double(jobs) / elapsed
        spread.append(elapsed)
        all += f.dropFirst(2)
      }
      all.sort()
      print("  " + v.name.padding(toLength: 14, withPad: " ", startingAt: 0)
        + String(format: " %7.2f jobs/s  median job %7.2f ms  worker elapsed %.2f-%.2f s", rate,
                 1e3 * all[all.count / 2], spread.min()!, spread.max()!))
    }
    for p in [A, Z, w, e, tau, work, v1, t1, v2, Hr] { p.deallocate() }
    H.deallocate(); iwork.deallocate()
    return
  }

  print("\(path)  n = \(n), \(precision), back-transforming m = \(m) eigenvectors, median of \(reps)")
  // Reference eigenvalues in FP64, and the floor of this precision: Accelerate's xsyevd on the same matrix.
  let wRef: UnsafeMutablePointer<Double> = buffer(n)
  do {
    let Ad: UnsafeMutablePointer<Double> = buffer(n * n), wd: UnsafeMutablePointer<Double> = buffer(lwork)
    Ad.update(from: H, count: n * n)
    _ = Double.syevd(n, Ad, n, wRef, wd, 1 + 6 * n + 2 * n * n, iwork, liwork)
    Ad.deallocate(); wd.deallocate()
    if Real.self == Float.self {
      A.update(from: Hr, count: n * n)
      _ = Real.syevd(n, A, n, w, work, 1 + 6 * n + 2 * n * n, iwork, liwork)
      var dw = 0.0
      for i in 0..<n { dw = max(dw, abs(Double(w[i]) - wRef[i])) }
      print(String(format: "  Accelerate ssyevd: eigenvalues %.1e Eh (FP32 floor)", dw))
    }
  }

  var times = Array(repeating: [[Double]](), count: variants.count)
  var checks = Array(repeating: "", count: variants.count)
  for r in -1..<reps {
    for (k, v) in variants.enumerated() {
      var t: [Double] = []
      v.run(&t)
      if r >= 0 { times[k].append(t) }
      if r == -1 {
        let (dw, res, orth) = accuracy(n, m, H, wRef, w, Z)
        checks[k] = String(format: "eigenvalues %.1e Eh, residual %.1e, orthogonality %.1e", dw, res, orth)
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
    print("                 \(checks[k])\(k < leftovers.count ? leftovers[k] : "")")
  }
  for p in [A, Z, w, e, tau, work, v1, t1, v2, Hr] { p.deallocate() }
  H.deallocate(); wRef.deallocate(); iwork.deallocate()
}

for path in files {
  if fp32 { benchmark(path, Float.self) } else { benchmark(path, Double.self) }
}
