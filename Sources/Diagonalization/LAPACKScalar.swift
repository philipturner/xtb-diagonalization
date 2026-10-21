// The LAPACK routines the successive band reduction needs beyond TridiagonalScalar's BLAS: Householder QR of a panel,
// the triangular factor T of its compact WY form, and applying I - V T V^T to a block. Accelerate's legacy symbols.
import CLAPACK

public protocol BandReductionScalar: TridiagonalScalar {
  /// QR of the m x n block a: R on and above the diagonal, the Householder vectors below it (unit diagonal implied).
  /// work holds at least n * 64 elements.
  static func geqrf(
    _ m: Int, _ n: Int, _ a: UnsafeMutablePointer<Self>, _ lda: Int, _ tau: UnsafeMutablePointer<Self>,
    _ work: UnsafeMutablePointer<Self>, _ lwork: Int)
  /// T of Q = H(1) ... H(k) = I - V T V^T, forward, column-wise V (n x k).
  static func larft(
    _ n: Int, _ k: Int, _ v: UnsafePointer<Self>, _ ldv: Int, _ tau: UnsafePointer<Self>,
    _ t: UnsafeMutablePointer<Self>, _ ldt: Int)
  /// C = op(Q) C ('L') or C op(Q) ('R'), Q = I - V T V^T, forward, column-wise V. work holds ldwork * k elements, with
  /// ldwork >= n ('L') or m ('R').
  static func larfb(
    _ side: UnsafePointer<CChar>, _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int,
    _ v: UnsafePointer<Self>, _ ldv: Int, _ t: UnsafePointer<Self>, _ ldt: Int, _ c: UnsafeMutablePointer<Self>,
    _ ldc: Int, _ work: UnsafeMutablePointer<Self>, _ ldwork: Int)
}

extension Double: BandReductionScalar {
  @inline(__always)
  public static func geqrf(
    _ m: Int, _ n: Int, _ a: UnsafeMutablePointer<Double>, _ lda: Int, _ tau: UnsafeMutablePointer<Double>,
    _ work: UnsafeMutablePointer<Double>, _ lwork: Int
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), lwork = lint(lwork), info: lint = 0
    dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info)
  }
  @inline(__always)
  public static func larft(
    _ n: Int, _ k: Int, _ v: UnsafePointer<Double>, _ ldv: Int, _ tau: UnsafePointer<Double>,
    _ t: UnsafeMutablePointer<Double>, _ ldt: Int
  ) {
    var n = lint(n), k = lint(k), ldv = lint(ldv), ldt = lint(ldt)
    dlarft_("F", "C", &n, &k, v, &ldv, tau, t, &ldt)
  }
  @inline(__always)
  public static func larfb(
    _ side: UnsafePointer<CChar>, _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int,
    _ v: UnsafePointer<Double>, _ ldv: Int, _ t: UnsafePointer<Double>, _ ldt: Int,
    _ c: UnsafeMutablePointer<Double>, _ ldc: Int, _ work: UnsafeMutablePointer<Double>, _ ldwork: Int
  ) {
    var m = lint(m), n = lint(n), k = lint(k), ldv = lint(ldv), ldt = lint(ldt), ldc = lint(ldc)
    var ldwork = lint(ldwork)
    dlarfb_(side, trans, "F", "C", &m, &n, &k, v, &ldv, t, &ldt, c, &ldc, work, &ldwork)
  }
}

extension Float: BandReductionScalar {
  @inline(__always)
  public static func geqrf(
    _ m: Int, _ n: Int, _ a: UnsafeMutablePointer<Float>, _ lda: Int, _ tau: UnsafeMutablePointer<Float>,
    _ work: UnsafeMutablePointer<Float>, _ lwork: Int
  ) {
    var m = lint(m), n = lint(n), lda = lint(lda), lwork = lint(lwork), info: lint = 0
    sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info)
  }
  @inline(__always)
  public static func larft(
    _ n: Int, _ k: Int, _ v: UnsafePointer<Float>, _ ldv: Int, _ tau: UnsafePointer<Float>,
    _ t: UnsafeMutablePointer<Float>, _ ldt: Int
  ) {
    var n = lint(n), k = lint(k), ldv = lint(ldv), ldt = lint(ldt)
    slarft_("F", "C", &n, &k, v, &ldv, tau, t, &ldt)
  }
  @inline(__always)
  public static func larfb(
    _ side: UnsafePointer<CChar>, _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ k: Int,
    _ v: UnsafePointer<Float>, _ ldv: Int, _ t: UnsafePointer<Float>, _ ldt: Int,
    _ c: UnsafeMutablePointer<Float>, _ ldc: Int, _ work: UnsafeMutablePointer<Float>, _ ldwork: Int
  ) {
    var m = lint(m), n = lint(n), k = lint(k), ldv = lint(ldv), ldt = lint(ldt), ldc = lint(ldc)
    var ldwork = lint(ldwork)
    slarfb_(side, trans, "F", "C", &m, &n, &k, v, &ldv, t, &ldt, c, &ldc, work, &ldwork)
  }
}
