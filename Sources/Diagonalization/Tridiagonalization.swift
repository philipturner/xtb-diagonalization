// Reduction of a symmetric matrix (upper triangle) to tridiagonal form: LAPACK's blocked dsytrd (dlatrd panels of 32
// columns + dsyr2k, dsytd2 for the last block), a line-by-line port of xtb/src/mctc/lapack/sytrd.c. The output
// (Householder vectors in A, tau) is what dormtr expects. The only change from LAPACK: the symmetric matrix-vector
// product, the BLAS2 core of the reduction, is a vectorized kernel that reads the triangle once for four columns at
// a time. Everything else calls Accelerate's legacy BLAS/LAPACK.
//
// Matrices are column-major. Indices in the LAPACK routines are 1-based, as in the Fortran and the C port.
// Arithmetic follows the C source as written: the vector multiply-adds are fused (vfmaq), every scalar a * b + c is
// rounded twice. Swift never contracts, so this port gives the same bits as sytrd.c compiled with
// -ffp-contract=off (GCC and clang, checked by sytrd-bench). With contraction on, compilers fuse some scalar
// multiply-adds and not others (vectorized in-order reductions stay unfused), so those builds differ in rounding.
import CLAPACK
import CSIMD

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

/// y = alpha A x, A symmetric n x n, upper triangle, column-major with lda. For rows above each group of four
/// columns, every loaded A(i, j0..j0+3) feeds both y(i) += sum_c A(i,c) x(c) and the column dot products
/// s_c += A(i,c) x(i), so the triangle is read once.
@inline(__always)
public func symmetricMatrixVector<T: TridiagonalScalar>(
  _ n: Int, _ alpha: T, _ A: UnsafePointer<T>, _ lda: Int, _ x: UnsafePointer<T>, _ y: UnsafeMutablePointer<T>
) {
  let w = T.Vector.scalarCount
  for i in 0..<n { y[i] = 0 }
  var j0 = 0
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
    // The 4x4 diagonal block, upper triangle.
    for c in 0..<4 {
      let col = A + (j0 + c) * lda
      for r in 0..<c {
        y[j0 + r] += col[j0 + r] * x[j0 + c]
        y[j0 + c] += col[j0 + r] * x[j0 + r]
      }
      y[j0 + c] += col[j0 + c] * x[j0 + c]
    }
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
  if alpha != 1 {
    for i in 0..<n { y[i] *= alpha }
  }
}

/// Offset of element (i, j), 1-based, in a column-major matrix with leading dimension ld.
@inline(__always)
private func at(_ i: Int, _ j: Int, _ ld: Int) -> Int { (i - 1) + (j - 1) * ld }

/// LAPACK dlatrd, UPLO = 'U'.
@inline(__always)
func latrdUpper<T: TridiagonalScalar>(
  _ n: Int, _ nb: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ w: UnsafeMutablePointer<T>, _ ldw: Int
) {
  for i in stride(from: n, through: n - nb + 1, by: -1) {
    let iw = i - n + nb
    if i < n {
      let m = n - i
      T.gemv("N", i, m, -1, a + at(1, i + 1, lda), lda, w + at(i, iw + 1, ldw), ldw, 1, a + at(1, i, lda))
      T.gemv("N", i, m, -1, w + at(1, iw + 1, ldw), ldw, a + at(i, i + 1, lda), lda, 1, a + at(1, i, lda))
    }
    if i > 1 {
      let im1 = i - 1
      T.larfg(im1, a + at(i - 1, i, lda), a + at(1, i, lda), tau + (i - 2))
      e[i - 2] = a[at(i - 1, i, lda)]
      a[at(i - 1, i, lda)] = 1
      symmetricMatrixVector(im1, 1, a, lda, a + at(1, i, lda), w + at(1, iw, ldw))
      if i < n {
        let m = n - i
        T.gemv("T", im1, m, 1, w + at(1, iw + 1, ldw), ldw, a + at(1, i, lda), 1, 0, w + at(i + 1, iw, ldw))
        T.gemv("N", im1, m, -1, a + at(1, i + 1, lda), lda, w + at(i + 1, iw, ldw), 1, 1, w + at(1, iw, ldw))
        T.gemv("T", im1, m, 1, a + at(1, i + 1, lda), lda, a + at(1, i, lda), 1, 0, w + at(i + 1, iw, ldw))
        T.gemv("N", im1, m, -1, w + at(1, iw + 1, ldw), ldw, w + at(i + 1, iw, ldw), 1, 1, w + at(1, iw, ldw))
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
  _ tau: UnsafeMutablePointer<T>
) {
  for i in stride(from: n - 1, through: 1, by: -1) {
    var taui: T = 0
    T.larfg(i, a + at(i, i + 1, lda), a + at(1, i + 1, lda), &taui)
    e[i - 1] = a[at(i, i + 1, lda)]
    if taui != 0 {
      a[at(i, i + 1, lda)] = 1
      symmetricMatrixVector(i, taui, a, lda, a + at(1, i + 1, lda), tau)
      let alpha = -0.5 * taui * T.dot(i, tau, a + at(1, i + 1, lda))
      T.axpy(i, alpha, a + at(1, i + 1, lda), tau)
      T.syr2(i, -1, a + at(1, i + 1, lda), tau, a, lda)
      a[at(i, i + 1, lda)] = e[i - 1]
    }
    d[i] = a[at(i + 1, i + 1, lda)]
    tau[i - 1] = taui
  }
  d[0] = a[at(1, 1, lda)]
}

/// LAPACK dsytrd, UPLO = 'U', with ilaenv's defaults: block size 32, crossover 32. work holds at least n * 32.
@inline(__always)
public func tridiagonalize<T: TridiagonalScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<T>, _ lda: Int, _ d: UnsafeMutablePointer<T>, _ e: UnsafeMutablePointer<T>,
  _ tau: UnsafeMutablePointer<T>, _ work: UnsafeMutablePointer<T>
) {
  let nb = 32, nx = 32
  var kk = n
  if n > nx {
    kk = n - ((n - nx + nb - 1) / nb) * nb
    for i in stride(from: n - nb + 1, through: kk + 1, by: -nb) {
      latrdUpper(i + nb - 1, nb, a, lda, e, tau, work, n)
      T.syr2k(i - 1, nb, -1, a + at(1, i, lda), lda, work, n, a, lda)
      for j in i...(i + nb - 1) {
        a[at(j - 1, j, lda)] = e[j - 2]
        d[j - 1] = a[at(j, j, lda)]
      }
    }
  }
  sytd2Upper(kk, a, lda, d, e, tau)
}
