// Tridiagonal reduction of dumped GFN2 matrices: Accelerate's dsytrd_ / ssytrd_ against the C port
// (xtb_dsytrd_upper / xtb_ssytrd_upper from xtb/src/mctc/lapack/sytrd.c, looked up in LIBXTB) and the Swift port in
// this package (Sources/Diagonalization).
//
//   swift build -c release --disable-sandbox
//   VECLIB_MAXIMUM_THREADS=1 .build/release/sytrd-bench [--lib LIBXTB] [--reps N] HS.bin ...
//
// LIBXTB defaults to build/xtb-build-diag/libxtb.dylib (GCC, -O3, contraction on). For a bit-for-bit check, build the
// C file alone without contraction, which is how the Swift port computes:
//   gcc-16 -O3 -ffp-contract=off -fPIC -shared xtb/src/mctc/lapack/sytrd.c -framework Accelerate -o sytrd-off.dylib
// HS.bin: dump from the xtb branch diag/dump (int32 n, H(n,n), S(n,n), ...). Reduces H' = X^T H X (S = U^T U,
// X = U^-1), as the SCC eigensolve does. Variants run round-robin, one call each per round, after one untimed round;
// reports the median. Checks: the Swift output (A, d, e, tau) against the C port, bit for bit; the eigenvalues of each
// tridiagonal matrix (dsterf / ssterf) against Accelerate's; and that each variant gives the same bits every round.
import CLAPACK
import Darwin
import Diagonalization
import Foundation

typealias SytrdD = @convention(c) (
  UnsafePointer<Int32>, UnsafeMutablePointer<Double>, UnsafePointer<Int32>, UnsafeMutablePointer<Double>,
  UnsafeMutablePointer<Double>, UnsafeMutablePointer<Double>, UnsafeMutablePointer<Double>) -> Void
typealias SytrdS = @convention(c) (
  UnsafePointer<Int32>, UnsafeMutablePointer<Float>, UnsafePointer<Int32>, UnsafeMutablePointer<Float>,
  UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>) -> Void

func now() -> Double {
  var t = timespec()
  clock_gettime(CLOCK_MONOTONIC_RAW, &t)
  return Double(t.tv_sec) + 1e-9 * Double(t.tv_nsec)
}

func median(_ x: [Double]) -> Double { x.sorted()[x.count / 2] }

var libPath = "build/xtb-build-diag/libxtb.dylib"
var reps = 21
var files: [String] = []
var argi = 1
let args = CommandLine.arguments
while argi < args.count {
  switch args[argi] {
  case "--lib": libPath = args[argi + 1]; argi += 2
  case "--reps": reps = Int(args[argi + 1])!; argi += 2
  default: files.append(args[argi]); argi += 1
  }
}
guard let handle = dlopen(libPath, RTLD_NOW | RTLD_LOCAL) else {
  fatalError("dlopen \(libPath): \(String(cString: dlerror()))")
}
guard let symD = dlsym(handle, "xtb_dsytrd_upper"), let symS = dlsym(handle, "xtb_ssytrd_upper") else {
  fatalError("\(libPath) has no xtb_dsytrd_upper / xtb_ssytrd_upper")
}
var info = Dl_info()
dladdr(symD, &info)
print("C port from \(String(cString: info.dli_fname))")
let cSytrdD = unsafeBitCast(symD, to: SytrdD.self)
let cSytrdS = unsafeBitCast(symS, to: SytrdS.self)

/// H' = X^T H X from a dump, column-major n x n.
func readMatrix(_ path: String) -> (Int, [Double]) {
  guard let f = fopen(path, "rb") else { fatalError("cannot open \(path)") }
  defer { fclose(f) }
  var n32: Int32 = 0
  fread(&n32, 4, 1, f)
  let n = Int(n32)
  var H = [Double](repeating: 0, count: n * n), S = H
  fread(&H, 8, n * n, f)
  fread(&S, 8, n * n, f)
  var nn = n32, info: Int32 = 0, one = 1.0
  dpotrf_("U", &nn, &S, &nn, &info)
  for j in 0..<n { for i in (j + 1)..<n { S[i + j * n] = 0 } }
  dtrtri_("U", "N", &nn, &S, &nn, &info)
  dtrmm_("R", "U", "N", "N", &nn, &nn, &one, S, &nn, &H, &nn)
  dtrmm_("L", "U", "T", "N", &nn, &nn, &one, S, &nn, &H, &nn)
  return (n, H)
}

struct Output<T: BinaryFloatingPoint> {
  var a: [T], d: [T], e: [T], tau: [T]
  init(_ n: Int) { a = []; d = Array(repeating: 0, count: n); e = d; tau = d }
}

/// Times the three variants round-robin; returns medians (ms), the outputs of the last round and whether each
/// variant's output was the same in every round (each round works on a fresh copy of the input).
func run<T: BinaryFloatingPoint>(
  _ n: Int, _ input: [T], _ variants: [(UnsafeMutablePointer<T>, UnsafeMutablePointer<T>, UnsafeMutablePointer<T>,
    UnsafeMutablePointer<T>, UnsafeMutablePointer<T>) -> Void]
) -> ([Double], [Output<T>], [Bool]) {
  var times = Array(repeating: [Double](), count: variants.count)
  var outs = Array(repeating: Output<T>(n), count: variants.count)
  var first = outs
  var repeatable = Array(repeating: true, count: variants.count)
  var work = [T](repeating: 0, count: 64 * n + 100_000)
  for r in -1..<reps {
    for (k, f) in variants.enumerated() {
      var a = input
      var o = outs[k]
      let t0 = now()
      f(&a, &o.d, &o.e, &o.tau, &work)
      let t1 = now()
      if r >= 0 { times[k].append(1e3 * (t1 - t0)) }
      o.a = a
      outs[k] = o
      if r == -1 { first[k] = o } else if !identical(n, o, first[k]) { repeatable[k] = false }
    }
  }
  return (times.map(median), outs, repeatable)
}

/// Largest |x - y| over the upper triangle of A, d, e, tau.
func maxDifference<T: BinaryFloatingPoint>(_ n: Int, _ x: Output<T>, _ y: Output<T>) -> Double {
  var m = 0.0
  for j in 0..<n { for i in 0...j { m = max(m, Double(abs(x.a[i + j * n] - y.a[i + j * n]))) } }
  for i in 0..<n {
    m = max(m, Double(abs(x.d[i] - y.d[i])), Double(abs(x.tau[i] - y.tau[i])))
    if i < n - 1 { m = max(m, Double(abs(x.e[i] - y.e[i]))) }
  }
  return m
}

func identical<T: BinaryFloatingPoint>(_ n: Int, _ x: Output<T>, _ y: Output<T>) -> Bool {
  var same = x.d == y.d && x.tau == y.tau && Array(x.e[0..<(n - 1)]) == Array(y.e[0..<(n - 1)])
  for j in 0..<n where same { for i in 0...j where x.a[i + j * n] != y.a[i + j * n] { same = false } }
  return same
}

func eigenvalues(_ n: Int, _ o: Output<Double>) -> [Double] {
  var d = o.d, e = o.e, nn = Int32(n), info: Int32 = 0
  dsterf_(&nn, &d, &e, &info)
  return d
}

func eigenvalues(_ n: Int, _ o: Output<Float>) -> [Double] {
  var d = o.d, e = o.e, nn = Int32(n), info: Int32 = 0
  ssterf_(&nn, &d, &e, &info)
  return d.map(Double.init)
}

func report<T: BinaryFloatingPoint>(
  _ label: String, _ n: Int, _ t: [Double], _ o: [Output<T>], _ w: [[Double]], _ repeatable: [Bool]
) {
  let names = ["Accelerate", "C (libxtb)", "Swift"]
  var line = "\(label)"
  for k in 0..<3 {
    line += "  \(names[k]) \(String(format: "%7.3f", t[k])) ms"
  }
  line += String(format: "  | C %.2fx, Swift %.2fx vs Accelerate; Swift/C %.3f", t[0] / t[1], t[0] / t[2], t[2] / t[1])
  print(line)
  let same = identical(n, o[2], o[1])
  let dw = (1..<3).map { k in zip(w[k], w[0]).map { abs($0 - $1) }.max()! }
  print("        Swift vs C: \(same ? "bit-identical" : String(format: "max |diff| %.1e", maxDifference(n, o[2], o[1])))"
    + String(format: "; eigenvalues vs Accelerate: C %.1e, Swift %.1e Eh", dw[0], dw[1])
    + "; same bits every round: " + zip(names, repeatable).map { "\($0) \($1 ? "yes" : "no")" }.joined(separator: ", "))
}

print("median of \(reps), round-robin, 1 thread (set VECLIB_MAXIMUM_THREADS=1)")
for path in files {
  let (n, H) = readMatrix(path)
  var nn = Int32(n), lda = Int32(n), lwork = Int32(64 * n + 100_000), info: Int32 = 0
  print("\(path)  n = \(n)")

  let (tD, oD, rD) = run(n, H, [
    { a, d, e, tau, work in dsytrd_("U", &nn, a, &lda, d, e, tau, work, &lwork, &info) },
    { a, d, e, tau, work in cSytrdD(&nn, a, &lda, d, e, tau, work) },
    { a, d, e, tau, work in xtb_dsytrd_upper(&nn, a, &lda, d, e, tau, work) },
  ])
  report("  FP64", n, tD, oD, oD.map { eigenvalues(n, $0) }, rD)

  let Hs = H.map(Float.init)
  let (tS, oS, rS) = run(n, Hs, [
    { a, d, e, tau, work in ssytrd_("U", &nn, a, &lda, d, e, tau, work, &lwork, &info) },
    { a, d, e, tau, work in cSytrdS(&nn, a, &lda, d, e, tau, work) },
    { a, d, e, tau, work in xtb_ssytrd_upper(&nn, a, &lda, d, e, tau, work) },
  ])
  report("  FP32", n, tS, oS, oS.map { eigenvalues(n, $0) }, rS)
}
