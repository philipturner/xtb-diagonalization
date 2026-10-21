// The compact WY representation of a block of Householder reflectors, Q = H_1 H_2 ... H_k = I - V T V^T (V: height x k,
// unit lower trapezoidal, first entry of each column implied; T: k x k upper triangular), and the ways it's applied.
// Every step of the decomposition that applies more than one reflector at a time goes through this file:
//   BandForm.swift                      T per panel, built once in the reduction (wyFactor); Q^T A Q by two wyApply or
//                                       by the sy2sb update; V and T kept
//   BandToBand.swift                    T per bulge step (wyFactor); Q^T A Q by two wyApply on a window; V and T kept
//   BulgeChasing.swift                  none: one reflector at a time, applied as soon as it's made
//   BulgeBackTransformation.swift       V and T per group of b sweeps, rebuilt on the fly after divide and conquer
//                                       (wyFactor; swift-xtb's draft did the same) or built in advance; applied by wyApply
//                                       or wyApplyParallelogram
//   BandToBandBackTransformation.swift  the kept V and T per step (wyApply), no rebuild
//   BandFormBackTransformation.swift    the kept V and T per panel (wyApply), no rebuild
// wyFactor is Accelerate's larft. wyApply is Accelerate's larfb (a triangular product with V's top block, a gemm with
// the rest, a triangular product with T, then the same in reverse, through a transposed workspace) or, with
// SBROptions.gemmWY, three gemm calls (V^T C, T times that, C minus V times that), as swift-xtb's WYTransform applied
// it. The gemm version needs V stored explicitly (unit diagonal, zeros above it): every step stores it that way.
// Measured on one bulge block (scripts/kernels/wy.c, 2b - 1 rows, b reflectors, m = 588 columns, FP64): larfb
// 45 / 87 / 260 us at b = 16 / 32 / 64, three gemm calls 17 / 32 / 88 us.

/// T of Q = H_1 ... H_count = I - V T V^T (forward, column-wise V of height rows; tau from geqrf or larfg).
@inline(__always)
func wyFactor<Real: BandReductionScalar>(
  _ height: Int, _ count: Int, _ v: UnsafePointer<Real>, _ ldv: Int, _ tau: UnsafePointer<Real>,
  _ t: UnsafeMutablePointer<Real>, _ ldt: Int
) {
  Real.larft(height, count, v, ldv, tau, t, ldt)
}

/// Workspace (elements) for wyApply on C with `other` columns ("L") or rows ("R"), `count` reflectors: enough for
/// either implementation.
func wyApplyWorkspace(_ other: Int, _ count: Int) -> Int {
  2 * other * count + count * count
}

/// C = op(Q) C (side "L") or C op(Q) (side "R"), op = "N" or "T", with Q = I - V T V^T; C is m x n. work holds
/// wyApplyWorkspace(n ("L") or m ("R"), count); ldwork >= n ("L") or m ("R"). gemm: three gemm calls instead of larfb,
/// with V stored explicitly (unit diagonal, zeros above).
@inline(__always)
func wyApply<Real: BandReductionScalar>(
  _ side: UnsafePointer<CChar>, _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ count: Int,
  _ v: UnsafePointer<Real>, _ ldv: Int, _ t: UnsafePointer<Real>, _ ldt: Int, _ c: UnsafeMutablePointer<Real>,
  _ ldc: Int, _ work: UnsafeMutablePointer<Real>, _ ldwork: Int, gemm: Bool = false
) {
  guard gemm else {
    Real.larfb(side, trans, m, n, count, v, ldv, t, ldt, c, ldc, work, ldwork)
    return
  }
  guard m > 0, n > 0, count > 0 else { return }
  // T as a dense count x count matrix (larft writes only its upper triangle), then W = V^T C or C V, W2 = op(T) W or
  // W op(T), C -= V W2 or W2 V^T.
  let k = count
  let td = work, w = td + k * k
  for j in 0..<k {
    for i in 0..<k { td[i + j * k] = i <= j ? t[i + j * ldt] : 0 }
  }
  if side[0] == CChar(UInt8(ascii: "L")) {
    let w2 = w + k * n
    Real.gemm("T", "N", k, n, m, 1, v, ldv, c, ldc, 0, w, k)
    Real.gemm(trans, "N", k, n, k, 1, td, k, w, k, 0, w2, k)
    Real.gemm("N", "N", m, n, k, -1, v, ldv, w2, k, 1, c, ldc)
  } else {
    let w2 = w + m * k
    Real.gemm("N", "N", m, k, n, 1, c, ldc, v, ldv, 0, w, m)
    Real.gemm("N", trans, m, k, k, 1, w, m, td, k, 0, w2, m)
    Real.gemm("N", "T", m, n, k, -1, w2, m, v, ldv, 1, c, ldc)
  }
}

/// Stores the first `count` columns of a geqrf V (height rows, ldv) explicitly: unit diagonal, zeros above it (R's
/// entries there are overwritten).
@inline(__always)
func wyMakeExplicit<Real: BandReductionScalar>(
  _ height: Int, _ count: Int, _ v: UnsafeMutablePointer<Real>, _ ldv: Int
) {
  for c in 0..<min(count, height) {
    for i in 0..<c { v[i + c * ldv] = 0 }
    v[c + c * ldv] = 1
  }
}

/// Z = Q Z for a full parallelogram block of the bulge back-transformation: b reflectors on 2b - 1 rows, column c of V
/// nonzero in rows c ..< c + b. V = [V1; V2] with V1 (rows 0 ..< b) unit lower triangular and V2 (rows b ..< 2b, the
/// last row zero) strictly upper triangular, so W = T (V1^T Z1 + V2^T Z2), Z1 -= V1 W, Z2 -= V2 W takes five b x b x m
/// triangular products where larfb spends seven (it treats V2 as dense). w1 and w2 hold b x m each. Measured slower
/// than wyApply with Accelerate's trmm (LOG.md); kept behind SBROptions.parallelogram.
func wyApplyParallelogram<Real: BandReductionScalar>(
  _ b: Int, _ m: Int, _ v: UnsafePointer<Real>, _ ldv: Int, _ t: UnsafePointer<Real>, _ ldt: Int,
  _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ w1: UnsafeMutablePointer<Real>, _ w2: UnsafeMutablePointer<Real>
) {
  let z2 = z + b
  // w1 = V1^T Z1, w2 = V2^T Z2 (Z2 padded with a zero row), w1 += w2
  for col in 0..<m {
    for i in 0..<b { w1[offset(i, col, b)] = z[offset(i, col, ldz)] }
    for i in 0..<(b - 1) { w2[offset(i, col, b)] = z2[offset(i, col, ldz)] }
    w2[offset(b - 1, col, b)] = 0
  }
  Real.trmm("L", "L", "T", "U", b, m, 1, v, ldv, w1, b)
  Real.trmm("L", "U", "T", "N", b, m, 1, v + b, ldv, w2, b)
  for i in 0..<(b * m) { w1[i] += w2[i] }
  // w1 = T w1
  Real.trmm("L", "U", "N", "N", b, m, 1, t, ldt, w1, b)
  // Z1 -= V1 w1, Z2 -= V2 w1
  for i in 0..<(b * m) { w2[i] = w1[i] }
  Real.trmm("L", "L", "N", "U", b, m, 1, v, ldv, w1, b)
  Real.trmm("L", "U", "N", "N", b, m, 1, v + b, ldv, w2, b)
  for col in 0..<m {
    for i in 0..<b { z[offset(i, col, ldz)] -= w1[offset(i, col, b)] }
    for i in 0..<(b - 1) { z2[offset(i, col, ldz)] -= w2[offset(i, col, b)] }
  }
}
