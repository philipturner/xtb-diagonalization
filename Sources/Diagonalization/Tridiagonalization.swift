// Reduction of a symmetric matrix (upper triangle) to tridiagonal form: LAPACK's blocked dsytrd (dlatrd panels of 32
// columns + dsyr2k, dsytd2 for the last block), a line-by-line port of xtb/src/mctc/lapack/sytrd.c. The output
// (Householder vectors in A, tau) is what dormtr expects. The changes from LAPACK: the symmetric matrix-vector
// product, the BLAS2 core of the reduction, is a vectorized kernel that reads the triangle once for eight columns at
// a time, and dlatrd's panel products are fused pairs of gemv calls (panelUpdate, panelDots, as sytrd.c). Everything
// else calls Accelerate's legacy BLAS/LAPACK.
//
// symmetric = false is the same reduction without exploiting symmetry: the full matrix (both triangles) is read and
// updated, with gemv, gemm and ger in place of symv, syr2k and syr2, at twice the work. The gemv is 'T' (A^T x = A x):
// Accelerate's dgemv 'N' reads 2-3x slower than 'T' (scripts/kernels/gemv_bw.c). It is the baseline for the
// successive band reduction work, which doesn't exploit symmetry either. Same Householder vectors up to rounding.
//
// Matrices are column-major. Indices in the LAPACK routines are 1-based, as in the Fortran and the C port.
// Arithmetic follows the C source as written: the vector multiply-adds are fused (vfmaq), every scalar a * b + c is
// rounded twice. Swift never contracts, so this port gives the same bits as sytrd.c compiled with
// -ffp-contract=off (GCC and clang, checked by sytrd-bench). With contraction on, compilers fuse some scalar
// multiply-adds and not others (vectorized in-order reductions stay unfused), so those builds differ in rounding.
import CLAPACK
import CSIMD
import Dispatch

public protocol TridiagonalScalar: BinaryFloatingPoint, SIMDScalar {
  /// One NEON register: SIMD2<Double> or SIMD4<Float>.
  associatedtype Vector: SIMD where Vector.Scalar == Self

  /// The horizontal sum of a NEON register, in the order of vaddvq_f64 / vaddvq_f32.
  static func horizontalSum(_ v: Vector) -> Self
  /// acc + a * b, rounded once (vfmaq_f64 / vfmaq_f32).
  static func fma(_ acc: Vector, _ a: Vector, _ b: Vector) -> Vector

  static func gemv(
    _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ alpha: Self, _ a: UnsafePointer<Self>, _ lda: Int,
    _ x: UnsafePointer<Self>, _ incx: Int, _ beta: Self, _ y: UnsafeMutablePointer<Self>)
  static func syr2k(
    _ n: Int, _ k: Int, _ alpha: Self, _ a: UnsafePointer<Self>, _ lda: Int, _ b: UnsafePointer<Self>, _ ldb: Int,
    _ c: UnsafeMutablePointer<Self>, _ ldc: Int)
  static func syr2(
    _ n: Int, _ alpha: Self, _ x: UnsafePointer<Self>, _ y: UnsafePointer<Self>, _ a: UnsafeMutablePointer<Self>,
    _ lda: Int)
  /// C = alpha op(A) op(B) + beta C
  static func gemm(
    _ transa: UnsafePointer<CChar>, _ transb: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int, _ alpha: Self,
    _ a: UnsafePointer<Self>, _ lda: Int, _ b: UnsafePointer<Self>, _ ldb: Int, _ beta: Self,
    _ c: UnsafeMutablePointer<Self>, _ ldc: Int)
  /// A = alpha x y^T + A, m x n
  static func ger(
    _ m: Int, _ n: Int, _ alpha: Self, _ x: UnsafePointer<Self>, _ y: UnsafePointer<Self>,
    _ a: UnsafeMutablePointer<Self>, _ lda: Int)
  static func larfg(
    _ n: Int, _ alpha: UnsafeMutablePointer<Self>, _ x: UnsafeMutablePointer<Self>,
    _ tau: UnsafeMutablePointer<Self>)
  static func dot(_ n: Int, _ x: UnsafePointer<Self>, _ y: UnsafePointer<Self>) -> Self
  static func axpy(_ n: Int, _ alpha: Self, _ x: UnsafePointer<Self>, _ y: UnsafeMutablePointer<Self>)
  static func scal(_ n: Int, _ alpha: Self, _ x: UnsafeMutablePointer<Self>)
}

extension Double: TridiagonalScalar {
  public typealias Vector = SIMD2<Double>

  @inline(__always)
  public static func horizontalSum(_ v: Vector) -> Double { v[0] + v[1] }
  @inline(__always)
  public static func fma(_ acc: Vector, _ a: Vector, _ b: Vector) -> Vector { csimd_fma_f64(acc, a, b) }

  @inline(__always)
  public static func gemv(
    _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ alpha: Double, _ a: UnsafePointer<Double>, _ lda: Int,
    _ x: UnsafePointer<Double>, _ incx: Int, _ beta: Double, _ y: UnsafeMutablePointer<Double>
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), incx = lint(incx), incy: lint = 1, alpha = alpha, beta = beta
    dgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy)
  }
  @inline(__always)
  public static func syr2k(
    _ n: Int, _ k: Int, _ alpha: Double, _ a: UnsafePointer<Double>, _ lda: Int, _ b: UnsafePointer<Double>,
    _ ldb: Int, _ c: UnsafeMutablePointer<Double>, _ ldc: Int
  ) {
    var n = lint(n), k = lint(k), lda = lint(lda), ldb = lint(ldb), ldc = lint(ldc), alpha = alpha, beta = 1.0
    dsyr2k_("U", "N", &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc)
  }
  @inline(__always)
  public static func syr2(
    _ n: Int, _ alpha: Double, _ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>,
    _ a: UnsafeMutablePointer<Double>, _ lda: Int
  ) {
    var n = lint(n), lda = lint(lda), incx: lint = 1, incy: lint = 1, alpha = alpha
    dsyr2_("U", &n, &alpha, x, &incx, y, &incy, a, &lda)
  }
  @inline(__always)
  public static func gemm(
    _ transa: UnsafePointer<CChar>, _ transb: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int, _ alpha: Double,
    _ a: UnsafePointer<Double>, _ lda: Int, _ b: UnsafePointer<Double>, _ ldb: Int, _ beta: Double,
    _ c: UnsafeMutablePointer<Double>, _ ldc: Int
  ) {
    var m = lint(m), n = lint(n), k = lint(k), lda = lint(lda), ldb = lint(ldb), ldc = lint(ldc)
    var alpha = alpha, beta = beta
    dgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc)
  }
  @inline(__always)
  public static func ger(
    _ m: Int, _ n: Int, _ alpha: Double, _ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>,
    _ a: UnsafeMutablePointer<Double>, _ lda: Int
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), incx: lint = 1, incy: lint = 1, alpha = alpha
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda)
  }
  @inline(__always)
  public static func larfg(
    _ n: Int, _ alpha: UnsafeMutablePointer<Double>, _ x: UnsafeMutablePointer<Double>,
    _ tau: UnsafeMutablePointer<Double>
  ) {
    var n = lint(n), incx: lint = 1
    dlarfg_(&n, alpha, x, &incx, tau)
  }
  @inline(__always)
  public static func dot(_ n: Int, _ x: UnsafePointer<Double>, _ y: UnsafePointer<Double>) -> Double {
    var n = lint(n), incx: lint = 1, incy: lint = 1
    return ddot_(&n, x, &incx, y, &incy)
  }
  @inline(__always)
  public static func axpy(_ n: Int, _ alpha: Double, _ x: UnsafePointer<Double>, _ y: UnsafeMutablePointer<Double>) {
    var n = lint(n), incx: lint = 1, incy: lint = 1, alpha = alpha
    daxpy_(&n, &alpha, x, &incx, y, &incy)
  }
  @inline(__always)
  public static func scal(_ n: Int, _ alpha: Double, _ x: UnsafeMutablePointer<Double>) {
    var n = lint(n), incx: lint = 1, alpha = alpha
    dscal_(&n, &alpha, x, &incx)
  }
}

extension Float: TridiagonalScalar {
  public typealias Vector = SIMD4<Float>

  @inline(__always)
  public static func horizontalSum(_ v: Vector) -> Float { (v[0] + v[1]) + (v[2] + v[3]) }
  @inline(__always)
  public static func fma(_ acc: Vector, _ a: Vector, _ b: Vector) -> Vector { csimd_fma_f32(acc, a, b) }

  @inline(__always)
  public static func gemv(
    _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ alpha: Float, _ a: UnsafePointer<Float>, _ lda: Int,
    _ x: UnsafePointer<Float>, _ incx: Int, _ beta: Float, _ y: UnsafeMutablePointer<Float>
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), incx = lint(incx), incy: lint = 1, alpha = alpha, beta = beta
    sgemv_(trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy)
  }
  @inline(__always)
  public static func syr2k(
    _ n: Int, _ k: Int, _ alpha: Float, _ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>,
    _ ldb: Int, _ c: UnsafeMutablePointer<Float>, _ ldc: Int
  ) {
    var n = lint(n), k = lint(k), lda = lint(lda), ldb = lint(ldb), ldc = lint(ldc), alpha = alpha
    var beta: Float = 1
    ssyr2k_("U", "N", &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc)
  }
  @inline(__always)
  public static func syr2(
    _ n: Int, _ alpha: Float, _ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>,
    _ a: UnsafeMutablePointer<Float>, _ lda: Int
  ) {
    var n = lint(n), lda = lint(lda), incx: lint = 1, incy: lint = 1, alpha = alpha
    ssyr2_("U", &n, &alpha, x, &incx, y, &incy, a, &lda)
  }
  @inline(__always)
  public static func gemm(
    _ transa: UnsafePointer<CChar>, _ transb: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int, _ alpha: Float,
    _ a: UnsafePointer<Float>, _ lda: Int, _ b: UnsafePointer<Float>, _ ldb: Int, _ beta: Float,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int
  ) {
    var m = lint(m), n = lint(n), k = lint(k), lda = lint(lda), ldb = lint(ldb), ldc = lint(ldc)
    var alpha = alpha, beta = beta
    sgemm_(transa, transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc)
  }
  @inline(__always)
  public static func ger(
    _ m: Int, _ n: Int, _ alpha: Float, _ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>,
    _ a: UnsafeMutablePointer<Float>, _ lda: Int
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), incx: lint = 1, incy: lint = 1, alpha = alpha
    sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda)
  }
  @inline(__always)
  public static func larfg(
    _ n: Int, _ alpha: UnsafeMutablePointer<Float>, _ x: UnsafeMutablePointer<Float>,
    _ tau: UnsafeMutablePointer<Float>
  ) {
    var n = lint(n), incx: lint = 1
    slarfg_(&n, alpha, x, &incx, tau)
  }
  // Computed here, as in sytrd.c: the legacy ABI of Accelerate's sdot_ returns a double.
  @inline(__always)
  public static func dot(_ n: Int, _ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>) -> Float {
    var s: Float = 0
    for i in 0..<n { s += x[i] * y[i] }
    return s
  }
  @inline(__always)
  public static func axpy(_ n: Int, _ alpha: Float, _ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>) {
    var n = lint(n), incx: lint = 1, incy: lint = 1, alpha = alpha
    saxpy_(&n, &alpha, x, &incx, y, &incy)
  }
  @inline(__always)
  public static func scal(_ n: Int, _ alpha: Float, _ x: UnsafeMutablePointer<Float>) {
    var n = lint(n), incx: lint = 1, alpha = alpha
    sscal_(&n, &alpha, x, &incx)
  }
}

@inline(__always)
private func load<T: TridiagonalScalar>(_ p: UnsafePointer<T>) -> T.Vector {
  UnsafeRawPointer(p).loadUnaligned(as: T.Vector.self)
}

@inline(__always)
private func store<T: TridiagonalScalar>(_ v: T.Vector, _ p: UnsafeMutablePointer<T>) {
  UnsafeMutableRawPointer(p).storeBytes(of: v, as: T.Vector.self)
}

/// symmetricMatrixVector's 4 x 4 diagonal block at column j0 (upper triangle).
@inline(__always)
private func symmetricDiagonalBlock<T: TridiagonalScalar>(
  _ j0: Int, _ A: UnsafePointer<T>, _ lda: Int, _ x: UnsafePointer<T>, _ y: UnsafeMutablePointer<T>
) {
  for c in 0..<4 {
    let col = A + (j0 + c) * lda
    for r in 0..<c {
      y[j0 + r] += col[j0 + r] * x[j0 + c]
      y[j0 + c] += col[j0 + r] * x[j0 + r]
    }
    y[j0 + c] += col[j0 + c] * x[j0 + c]
  }
}

/// y(0 ..< n) += the contributions of columns jbeg ..< n of the upper triangle of the symmetric A (jbeg a multiple of
/// 8; rows 0 ..< n, column-major with lda). For rows above each group of eight (then four) columns, every loaded
/// A(i, j0..j0+7) feeds both y(i) += sum_c A(i,c) x(c) and the column dot products s_c += A(i,c) x(i), so the
/// triangle is read once.
@inline(__always)
func symmetricColumns<T: TridiagonalScalar>(
  _ jbeg: Int, _ n: Int, _ A: UnsafePointer<T>, _ lda: Int, _ x: UnsafePointer<T>, _ y: UnsafeMutablePointer<T>
) {
  let w = T.Vector.scalarCount
  var j0 = jbeg
  // Eight columns per pass while they fit: each column's dot product is a chain of one fused multiply-add per vector
  // of rows (4-cycle latency), so eight chains keep the four FMA pipes busy where four can't (1.1-1.25x). Same bits as
  // two passes of four: rows above j0 see the eight columns in the same order, and the last four columns' rows
  // j0 ..< j0 + 4 follow the first four's diagonal block, as they would.
  while j0 + 8 <= n {
    let c0 = A + j0 * lda, c1 = c0 + lda, c2 = c1 + lda, c3 = c2 + lda
    let c4 = c3 + lda, c5 = c4 + lda, c6 = c5 + lda, c7 = c6 + lda
    let x0 = T.Vector(repeating: x[j0]), x1 = T.Vector(repeating: x[j0 + 1])
    let x2 = T.Vector(repeating: x[j0 + 2]), x3 = T.Vector(repeating: x[j0 + 3])
    let x4 = T.Vector(repeating: x[j0 + 4]), x5 = T.Vector(repeating: x[j0 + 5])
    let x6 = T.Vector(repeating: x[j0 + 6]), x7 = T.Vector(repeating: x[j0 + 7])
    var s0 = T.Vector(), s1 = T.Vector(), s2 = T.Vector(), s3 = T.Vector()
    var s4 = T.Vector(), s5 = T.Vector(), s6 = T.Vector(), s7 = T.Vector()
    var i = 0
    while i + w <= j0 {
      let a0: T.Vector = load(c0 + i), a1: T.Vector = load(c1 + i)
      let a2: T.Vector = load(c2 + i), a3: T.Vector = load(c3 + i)
      let a4: T.Vector = load(c4 + i), a5: T.Vector = load(c5 + i)
      let a6: T.Vector = load(c6 + i), a7: T.Vector = load(c7 + i)
      let xi: T.Vector = load(x + i)
      var yi: T.Vector = load(y + i)
      yi = T.fma(yi, a0, x0)
      yi = T.fma(yi, a1, x1)
      yi = T.fma(yi, a2, x2)
      yi = T.fma(yi, a3, x3)
      yi = T.fma(yi, a4, x4)
      yi = T.fma(yi, a5, x5)
      yi = T.fma(yi, a6, x6)
      yi = T.fma(yi, a7, x7)
      store(yi, y + i)
      s0 = T.fma(s0, a0, xi)
      s1 = T.fma(s1, a1, xi)
      s2 = T.fma(s2, a2, xi)
      s3 = T.fma(s3, a3, xi)
      s4 = T.fma(s4, a4, xi)
      s5 = T.fma(s5, a5, xi)
      s6 = T.fma(s6, a6, xi)
      s7 = T.fma(s7, a7, xi)
      i += w
    }
    y[j0] += T.horizontalSum(s0)
    y[j0 + 1] += T.horizontalSum(s1)
    y[j0 + 2] += T.horizontalSum(s2)
    y[j0 + 3] += T.horizontalSum(s3)
    symmetricDiagonalBlock(j0, A, lda, x, y)
    while i + w <= j0 + 4 {
      let a4: T.Vector = load(c4 + i), a5: T.Vector = load(c5 + i)
      let a6: T.Vector = load(c6 + i), a7: T.Vector = load(c7 + i)
      let xi: T.Vector = load(x + i)
      var yi: T.Vector = load(y + i)
      yi = T.fma(yi, a4, x4)
      yi = T.fma(yi, a5, x5)
      yi = T.fma(yi, a6, x6)
      yi = T.fma(yi, a7, x7)
      store(yi, y + i)
      s4 = T.fma(s4, a4, xi)
      s5 = T.fma(s5, a5, xi)
      s6 = T.fma(s6, a6, xi)
      s7 = T.fma(s7, a7, xi)
      i += w
    }
    y[j0 + 4] += T.horizontalSum(s4)
    y[j0 + 5] += T.horizontalSum(s5)
    y[j0 + 6] += T.horizontalSum(s6)
    y[j0 + 7] += T.horizontalSum(s7)
    symmetricDiagonalBlock(j0 + 4, A, lda, x, y)
    j0 += 8
  }
  while j0 + 4 <= n {
    let c0 = A + j0 * lda, c1 = c0 + lda, c2 = c1 + lda, c3 = c2 + lda
    let x0 = T.Vector(repeating: x[j0]), x1 = T.Vector(repeating: x[j0 + 1])
    let x2 = T.Vector(repeating: x[j0 + 2]), x3 = T.Vector(repeating: x[j0 + 3])
    var s0 = T.Vector(), s1 = T.Vector(), s2 = T.Vector(), s3 = T.Vector()
    var i = 0
    while i + w <= j0 {
      let a0: T.Vector = load(c0 + i), a1: T.Vector = load(c1 + i)
      let a2: T.Vector = load(c2 + i), a3: T.Vector = load(c3 + i)
      let xi: T.Vector = load(x + i)
      var yi: T.Vector = load(y + i)
      yi = T.fma(yi, a0, x0)
      yi = T.fma(yi, a1, x1)
      yi = T.fma(yi, a2, x2)
      yi = T.fma(yi, a3, x3)
      store(yi, y + i)
      s0 = T.fma(s0, a0, xi)
      s1 = T.fma(s1, a1, xi)
      s2 = T.fma(s2, a2, xi)
      s3 = T.fma(s3, a3, xi)
      i += w
    }
    var t0 = T.horizontalSum(s0), t1 = T.horizontalSum(s1)
    var t2 = T.horizontalSum(s2), t3 = T.horizontalSum(s3)
    while i < j0 {
      y[i] += c0[i] * x[j0] + c1[i] * x[j0 + 1] + c2[i] * x[j0 + 2] + c3[i] * x[j0 + 3]
      t0 += c0[i] * x[i]
      t1 += c1[i] * x[i]
      t2 += c2[i] * x[i]
      t3 += c3[i] * x[i]
      i += 1
    }
    y[j0] += t0
    y[j0 + 1] += t1
    y[j0 + 2] += t2
    y[j0 + 3] += t3
    symmetricDiagonalBlock(j0, A, lda, x, y)
    j0 += 4
  }
  while j0 < n {
    let col = A + j0 * lda
    var s: T = 0
    for i in 0..<j0 {
      y[i] += col[i] * x[j0]
      s += col[i] * x[i]
    }
    y[j0] += s + col[j0] * x[j0]
    j0 += 1
  }
}

/// y = alpha A x, A symmetric n x n, upper triangle, column-major with lda (symmetricColumns over all columns).
@inline(__always)
public func symmetricMatrixVector<T: TridiagonalScalar>(
  _ n: Int, _ alpha: T, _ A: UnsafePointer<T>, _ lda: Int, _ x: UnsafePointer<T>, _ y: UnsafeMutablePointer<T>
) {
  for i in 0..<n { y[i] = 0 }
  symmetricColumns(0, n, A, lda, x, y)
  if alpha != 1 {
    for i in 0..<n { y[i] *= alpha }
  }
}

/// dlatrd's panel products, fused (each pair of LAPACK's gemv calls reads the panels of A and W that share their
/// rows): y -= P1 c1 + P2 c2 in one pass over y, eight columns per load of y. P1 and P2 are rows x m, column-major.
/// Same operations as sytrd.c's panel_update.
@inline(__always)
func panelUpdate<T: TridiagonalScalar>(
  _ rows: Int, _ m: Int, _ p1: UnsafePointer<T>, _ ld1: Int, _ c1: UnsafePointer<T>, _ p2: UnsafePointer<T>,
  _ ld2: Int, _ c2: UnsafePointer<T>, _ y: UnsafeMutablePointer<T>
) {
  let w = T.Vector.scalarCount
  var c = 0
  while c + 4 <= m {
    let a0 = p1 + c * ld1, a1 = a0 + ld1, a2 = a1 + ld1, a3 = a2 + ld1
    let b0 = p2 + c * ld2, b1 = b0 + ld2, b2 = b1 + ld2, b3 = b2 + ld2
    let ka0 = T.Vector(repeating: -c1[c]), ka1 = T.Vector(repeating: -c1[c + 1])
    let ka2 = T.Vector(repeating: -c1[c + 2]), ka3 = T.Vector(repeating: -c1[c + 3])
    let kb0 = T.Vector(repeating: -c2[c]), kb1 = T.Vector(repeating: -c2[c + 1])
    let kb2 = T.Vector(repeating: -c2[c + 2]), kb3 = T.Vector(repeating: -c2[c + 3])
    var i = 0
    while i + w <= rows {
      var yi: T.Vector = load(y + i)
      yi = T.fma(yi, load(a0 + i), ka0)
      yi = T.fma(yi, load(b0 + i), kb0)
      yi = T.fma(yi, load(a1 + i), ka1)
      yi = T.fma(yi, load(b1 + i), kb1)
      yi = T.fma(yi, load(a2 + i), ka2)
      yi = T.fma(yi, load(b2 + i), kb2)
      yi = T.fma(yi, load(a3 + i), ka3)
      yi = T.fma(yi, load(b3 + i), kb3)
      store(yi, y + i)
      i += w
    }
    while i < rows {
      var t = a0[i] * c1[c] + b0[i] * c2[c]
      t += a1[i] * c1[c + 1]
      t += b1[i] * c2[c + 1]
      t += a2[i] * c1[c + 2]
      t += b2[i] * c2[c + 2]
      t += a3[i] * c1[c + 3]
      t += b3[i] * c2[c + 3]
      y[i] -= t
      i += 1
    }
    c += 4
  }
  while c < m {
    let a0 = p1 + c * ld1, b0 = p2 + c * ld2
    let ka = T.Vector(repeating: -c1[c]), kb = T.Vector(repeating: -c2[c])
    var i = 0
    while i + w <= rows {
      var yi: T.Vector = load(y + i)
      yi = T.fma(yi, load(a0 + i), ka)
      yi = T.fma(yi, load(b0 + i), kb)
      store(yi, y + i)
      i += w
    }
    while i < rows {
      y[i] -= a0[i] * c1[c] + b0[i] * c2[c]
      i += 1
    }
    c += 1
  }
}

/// u1 = P1^T x and u2 = P2^T x in one pass over x, eight dot products per load of x (sytrd.c's panel_dots).
@inline(__always)
func panelDots<T: TridiagonalScalar>(
  _ rows: Int, _ m: Int, _ p1: UnsafePointer<T>, _ ld1: Int, _ p2: UnsafePointer<T>, _ ld2: Int,
  _ x: UnsafePointer<T>, _ u1: UnsafeMutablePointer<T>, _ u2: UnsafeMutablePointer<T>
) {
  let w = T.Vector.scalarCount
  var c = 0
  while c + 4 <= m {
    let a0 = p1 + c * ld1, a1 = a0 + ld1, a2 = a1 + ld1, a3 = a2 + ld1
    let b0 = p2 + c * ld2, b1 = b0 + ld2, b2 = b1 + ld2, b3 = b2 + ld2
    var sa0 = T.Vector(), sa1 = T.Vector(), sa2 = T.Vector(), sa3 = T.Vector()
    var sb0 = T.Vector(), sb1 = T.Vector(), sb2 = T.Vector(), sb3 = T.Vector()
    var i = 0
    while i + w <= rows {
      let xi: T.Vector = load(x + i)
      sa0 = T.fma(sa0, load(a0 + i), xi)
      sb0 = T.fma(sb0, load(b0 + i), xi)
      sa1 = T.fma(sa1, load(a1 + i), xi)
      sb1 = T.fma(sb1, load(b1 + i), xi)
      sa2 = T.fma(sa2, load(a2 + i), xi)
      sb2 = T.fma(sb2, load(b2 + i), xi)
      sa3 = T.fma(sa3, load(a3 + i), xi)
      sb3 = T.fma(sb3, load(b3 + i), xi)
      i += w
    }
    var ta0 = T.horizontalSum(sa0), ta1 = T.horizontalSum(sa1), ta2 = T.horizontalSum(sa2)
    var ta3 = T.horizontalSum(sa3), tb0 = T.horizontalSum(sb0), tb1 = T.horizontalSum(sb1)
    var tb2 = T.horizontalSum(sb2), tb3 = T.horizontalSum(sb3)
    while i < rows {
      ta0 += a0[i] * x[i]
      ta1 += a1[i] * x[i]
      ta2 += a2[i] * x[i]
      ta3 += a3[i] * x[i]
      tb0 += b0[i] * x[i]
      tb1 += b1[i] * x[i]
      tb2 += b2[i] * x[i]
      tb3 += b3[i] * x[i]
      i += 1
    }
    u1[c] = ta0; u1[c + 1] = ta1; u1[c + 2] = ta2; u1[c + 3] = ta3
    u2[c] = tb0; u2[c + 1] = tb1; u2[c + 2] = tb2; u2[c + 3] = tb3
    c += 4
  }
  while c < m {
    let a0 = p1 + c * ld1, b0 = p2 + c * ld2
    var sa = T.Vector(), sb = T.Vector()
    var i = 0
    while i + w <= rows {
      let xi: T.Vector = load(x + i)
      sa = T.fma(sa, load(a0 + i), xi)
      sb = T.fma(sb, load(b0 + i), xi)
      i += w
    }
    var ta = T.horizontalSum(sa), tb = T.horizontalSum(sb)
    while i < rows {
      ta += a0[i] * x[i]
      tb += b0[i] * x[i]
      i += 1
    }
    u1[c] = ta
    u2[c] = tb
    c += 1
  }
}

/// Offset of element (i, j), 1-based, in a column-major matrix with leading dimension ld.
@inline(__always)
private func at(_ i: Int, _ j: Int, _ ld: Int) -> Int { (i - 1) + (j - 1) * ld }

/// LAPACK dlatrd, UPLO = 'U'.
@inline(__always)
func latrdUpper<T: TridiagonalScalar>(
  _ n: Int, _ nb: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ w: UnsafeMutablePointer<T>, _ ldw: Int, _ symmetric: Bool
) {
  for i in stride(from: n, through: n - nb + 1, by: -1) {
    let iw = i - n + nb
    if i < n {
      let m = n - i
      // The row i of W's and A's trailing panel columns, contiguous (m < nb = 32).
      withUnsafeTemporaryAllocation(of: T.self, capacity: 2 * m) { rowsBuffer in
        let wr = rowsBuffer.baseAddress!, ar = wr + m
        for c in 0..<m {
          wr[c] = w[at(i, iw + 1 + c, ldw)]
          ar[c] = a[at(i, i + 1 + c, lda)]
        }
        panelUpdate(i, m, a + at(1, i + 1, lda), lda, wr, w + at(1, iw + 1, ldw), ldw, ar, a + at(1, i, lda))
      }
    }
    if i > 1 {
      let im1 = i - 1
      T.larfg(im1, a + at(i - 1, i, lda), a + at(1, i, lda), tau + (i - 2))
      e[i - 2] = a[at(i - 1, i, lda)]
      a[at(i - 1, i, lda)] = 1
      if symmetric {
        symmetricMatrixVector(im1, 1, a, lda, a + at(1, i, lda), w + at(1, iw, ldw))
      } else {
        T.gemv("T", im1, im1, 1, a, lda, a + at(1, i, lda), 1, 0, w + at(1, iw, ldw))
      }
      if i < n {
        let m = n - i
        // u = W^T v and u2 = A^T v in one pass, then w -= A u + W u2 in one; W(i + 1 ... n, iw) ends as u2, as
        // with LAPACK's call order.
        withUnsafeTemporaryAllocation(of: T.self, capacity: m) { uBuffer in
          let u = uBuffer.baseAddress!
          panelDots(im1, m, w + at(1, iw + 1, ldw), ldw, a + at(1, i + 1, lda), lda, a + at(1, i, lda), u,
                    w + at(i + 1, iw, ldw))
          panelUpdate(im1, m, a + at(1, i + 1, lda), lda, u, w + at(1, iw + 1, ldw), ldw, w + at(i + 1, iw, ldw),
                      w + at(1, iw, ldw))
        }
      }
      T.scal(im1, tau[i - 2], w + at(1, iw, ldw))
      let alpha = -0.5 * tau[i - 2] * T.dot(im1, w + at(1, iw, ldw), a + at(1, i, lda))
      T.axpy(im1, alpha, a + at(1, i, lda), w + at(1, iw, ldw))
    }
  }
}

/// LAPACK dsytd2, UPLO = 'U'.
@inline(__always)
func sytd2Upper<T: TridiagonalScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ d: UnsafeMutablePointer<T>, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ symmetric: Bool
) {
  for i in stride(from: n - 1, through: 1, by: -1) {
    var taui: T = 0
    T.larfg(i, a + at(i, i + 1, lda), a + at(1, i + 1, lda), &taui)
    e[i - 1] = a[at(i, i + 1, lda)]
    if taui != 0 {
      a[at(i, i + 1, lda)] = 1
      if symmetric {
        symmetricMatrixVector(i, taui, a, lda, a + at(1, i + 1, lda), tau)
      } else {
        T.gemv("T", i, i, taui, a, lda, a + at(1, i + 1, lda), 1, 0, tau)
      }
      let alpha = -0.5 * taui * T.dot(i, tau, a + at(1, i + 1, lda))
      T.axpy(i, alpha, a + at(1, i + 1, lda), tau)
      if symmetric {
        T.syr2(i, -1, a + at(1, i + 1, lda), tau, a, lda)
      } else {
        T.ger(i, i, -1, a + at(1, i + 1, lda), tau, a, lda)
        T.ger(i, i, -1, tau, a + at(1, i + 1, lda), a, lda)
      }
      a[at(i, i + 1, lda)] = e[i - 1]
    }
    d[i] = a[at(i + 1, i + 1, lda)]
    tau[i - 1] = taui
  }
  d[0] = a[at(1, 1, lda)]
}

/// LAPACK dsytrd, UPLO = 'U', with ilaenv's defaults: block size 32, crossover 32. work holds at least n * 32.
/// symmetric = false: A holds the full symmetric matrix and both triangles are updated (see the top of the file).
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
public func tridiagonalize<T: TridiagonalScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ d: UnsafeMutablePointer<T>, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ work: UnsafeMutablePointer<T>, symmetric: Bool = true
) {
  let nb = 32, nx = 32
  var kk = n
  if n > nx {
    kk = n - ((n - nx + nb - 1) / nb) * nb
    for i in stride(from: n - nb + 1, through: kk + 1, by: -nb) {
      latrdUpper(i + nb - 1, nb, a, lda, e, tau, work, n, symmetric)
      if symmetric {
        T.syr2k(i - 1, nb, -1, a + at(1, i, lda), lda, work, n, a, lda)
      } else {
        T.gemm("N", "T", i - 1, i - 1, nb, -1, a + at(1, i, lda), lda, work, n, 1, a, lda)
        T.gemm("N", "T", i - 1, i - 1, nb, -1, work, n, a + at(1, i, lda), lda, 1, a, lda)
      }
      for j in i...(i + nb - 1) {
        a[at(j - 1, j, lda)] = e[j - 2]
        d[j - 1] = a[at(j, j, lda)]
      }
    }
  }
  sytd2Upper(kk, a, lda, d, e, tau, symmetric)
}

/// Start of part t of 0 ..< length in `parts` parts, at multiples of 32 elements (whole cache lines).
@inline(__always)
private func teamRows(_ length: Int, _ t: Int, _ parts: Int) -> Int {
  min(length, (length * t / parts + 31) / 32 * 32)
}

/// Start of the columns of part t of the upper triangle of order `length`: equal areas, multiples of 8.
@inline(__always)
private func teamColumns(_ length: Int, _ t: Int, _ parts: Int) -> Int {
  if t == parts { return length }
  return min(length, Int((Double(length) * (Double(t) / Double(parts)).squareRoot() / 8 + 0.5)) * 8)
}

/// dlatrd, UPLO = 'U', on worker t of a team of `threads` (sytrd.c's latrd_team): per column, the panel update of
/// the column split by rows -> barrier -> larfg on worker 0 -> barrier -> symv split by columns (equal triangle
/// areas) into one vector per worker, and panelDots split by rows -> barrier -> sum of the vectors, panelUpdate,
/// scaling and the partial dot, split by rows -> barrier -> axpy, same rows (then the next column's panel update,
/// same rows again: no barrier). W(i, iw + 1), which the next column's panel update reads as a coefficient while
/// its owner updates it, is recomputed by every worker from a copy published before the last barrier.
@inline(__always)
func latrdUpperTeam<T: TridiagonalScalar>(
  _ t: Int, _ threads: Int, _ n: Int, _ nb: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int,
  _ e: UnsafeMutablePointer<T>, _ tau: UnsafeMutablePointer<T>, _ w: UnsafeMutablePointer<T>, _ ldw: Int,
  _ scratch: TeamScratch<T>, _ barrier: SpinBarrier
) {
  let ybuf = scratch.vectors, ldy = scratch.ldy, ubuf = scratch.dots, dbuf = scratch.partials
  let shared = scratch.shared
  withUnsafeTemporaryAllocation(of: T.self, capacity: 128) { local in
    let wr = local.baseAddress!, ar = wr + 32, us = wr + 64
    withUnsafeTemporaryAllocation(of: Int.self, capacity: threads + 1) { cend in
      var wlast: T = 0  // the last column's W(i, iw + 1)
      for i in stride(from: n, through: n - nb + 1, by: -1) {
        let iw = i - n + nb
        if i < n {
          let m = n - i, r0 = teamRows(i, t, threads), r1 = teamRows(i, t + 1, threads)
          wr[0] = wlast
          ar[0] = a[at(i, i + 1, lda)]
          for c in 1..<max(m, 1) {
            wr[c] = w[at(i, iw + 1 + c, ldw)]
            ar[c] = a[at(i, i + 1 + c, lda)]
          }
          if r1 > r0 {
            panelUpdate(r1 - r0, m, a + at(1 + r0, i + 1, lda), lda, wr, w + at(1 + r0, iw + 1, ldw), ldw, ar,
                        a + at(1 + r0, i, lda))
          }
        }
        if i == 1 { break }
        let im1 = i - 1, m = n - i
        let v = a + at(1, i, lda), wc = w + at(1, iw, ldw)
        barrier.wait()
        if t == 0 {
          T.larfg(im1, v + (i - 2), v, tau + (i - 2))
          e[i - 2] = v[i - 2]
          v[i - 2] = 1
        }
        barrier.wait()
        for q in 0...threads { cend[q] = teamColumns(im1, q, threads) }
        let c0 = cend[t], c1 = cend[t + 1], y = ybuf + t * ldy
        for r in 0..<c1 { y[r] = 0 }
        if c1 > c0 { symmetricColumns(c0, c1, a, lda, v, y) }
        let r0 = teamRows(im1, t, threads), r1 = teamRows(im1, t + 1, threads)
        let u = ubuf + t * 64
        for c in 0..<m {
          u[c] = 0
          u[32 + c] = 0
        }
        if m > 0 && r1 > r0 {
          panelDots(r1 - r0, m, w + at(1 + r0, iw + 1, ldw), ldw, a + at(1 + r0, i + 1, lda), lda, v + r0, u,
                    u + 32)
        }
        barrier.wait()
        for c in 0..<m {
          var s1: T = 0, s2: T = 0
          for q in 0..<threads {
            s1 += ubuf[q * 64 + c]
            s2 += ubuf[q * 64 + 32 + c]
          }
          us[c] = s1
          us[32 + c] = s2
        }
        if t == 0 {
          for c in 0..<m { w[at(i + 1 + c, iw, ldw)] = us[32 + c] }
        }
        for r in r0..<r1 {
          var s: T = 0
          for q in 0..<threads where r < cend[q + 1] { s += ybuf[q * ldy + r] }
          wc[r] = s
        }
        if m > 0 && r1 > r0 {
          panelUpdate(r1 - r0, m, a + at(1 + r0, i + 1, lda), lda, us, w + at(1 + r0, iw + 1, ldw), ldw, us + 32,
                      wc + r0)
        }
        let taui = tau[i - 2]
        var dot: T = 0
        for r in r0..<r1 {
          wc[r] *= taui
          dot += wc[r] * v[r]
        }
        dbuf[t * 32] = dot
        if r0 < im1 && im1 <= r1 { shared[0] = wc[im1 - 1] }
        barrier.wait()
        var sum: T = 0
        for q in 0..<threads { sum += dbuf[q * 32] }
        let alpha = -0.5 * taui * sum
        for r in r0..<r1 { wc[r] = wc[r].addingProduct(alpha, v[r]) }
        wlast = shared[0].addingProduct(alpha, v[im1 - 1])
      }
    }
  }
}

/// Buffers shared by the team of tridiagonalizeParallel: one symv vector per worker (leading dimension ldy), the
/// panel dot products per worker (64 apart), the partial dots (32 apart: one cache line each) and one shared value.
struct TeamScratch<T: TridiagonalScalar> {
  let vectors: UnsafeMutablePointer<T>, ldy: Int, dots: UnsafeMutablePointer<T>
  let partials: UnsafeMutablePointer<T>, shared: UnsafeMutablePointer<T>
}

/// tridiagonalize (symmetric) on a team of `threads` workers (sytrd.c's sytrd_team): one concurrentPerform for the
/// whole call, the workers synchronized by a SpinBarrier, four barriers per column (latrdUpperTeam); the syr2k after
/// each panel split by columns of equal areas (gemm above the diagonal blocks); the leading columns below order 128
/// serially (tridiagonalize). Same result up to rounding, the same bits for the same thread count; threads <= 1 or
/// n < 256 is tridiagonalize itself. Swift never contracts, so this differs from sytrd.c (built with contraction) in
/// rounding, as tridiagonalize does.
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
public func tridiagonalizeParallel<T: TridiagonalScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ d: UnsafeMutablePointer<T>, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ work: UnsafeMutablePointer<T>, threads: Int
) {
  let nb = 32, nx = 32, tail = 128
  guard threads > 1, n >= 256 else {
    tridiagonalize(n, a, lda, d, e, tau, work)
    return
  }
  let kk = n - ((n - nx + nb - 1) / nb) * nb, ldy = (n + 31) / 32 * 32
  var last = n - nb + 1  // first column of the last panel the team reduces
  while last - nb >= kk + 1 && last - 1 >= tail { last -= nb }
  let buffer = UnsafeMutablePointer<T>.allocate(capacity: threads * (ldy + 64 + 32) + 32)
  defer { buffer.deallocate() }
  let scratch = TeamScratch(
    vectors: buffer, ldy: ldy, dots: buffer + threads * ldy, partials: buffer + threads * (ldy + 64),
    shared: buffer + threads * (ldy + 96))
  let barrier = SpinBarrier(threads: threads)
  DispatchQueue.concurrentPerform(iterations: threads) { t in
    for i in stride(from: n - nb + 1, through: last, by: -nb) {
      let im1 = i - 1
      latrdUpperTeam(t, threads, i + nb - 1, nb, a, lda, e, tau, work, n, scratch, barrier)
      barrier.wait()
      // columns c0 ..< c1 of the update, equal areas
      let c0 = teamColumns(im1, t, threads), c1 = teamColumns(im1, t + 1, threads), nc = c1 - c0
      let v = a + at(1, i, lda)
      if c0 > 0 && nc > 0 {
        T.gemm("N", "T", c0, nc, nb, -1, v, lda, work + c0, n, 1, a + c0 * lda, lda)
        T.gemm("N", "T", c0, nc, nb, -1, work, n, v + c0, lda, 1, a + c0 * lda, lda)
      }
      if nc > 0 { T.syr2k(nc, nb, -1, v + c0, lda, work + c0, n, a + c0 + c0 * lda, lda) }
      barrier.wait()
      if t == 0 {
        for j in i...(i + nb - 1) {
          a[at(j - 1, j, lda)] = e[j - 2]
          d[j - 1] = a[at(j, j, lda)]
        }
      }
      barrier.wait()
    }
  }
  // the leading last - 1 columns, serially
  tridiagonalize(last - 1, a, lda, d, e, tau, work)
}
