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
// wyFactor and wyApply are Accelerate's larft and larfb (larfb: a triangular product with V's top block, a gemm with
// the rest, a triangular product with T, then the same in reverse). swift-xtb's WYTransform built T in blocks of a
// "small block size" with sger/sgemm and applied V, T, V as three sgemm calls; that version hasn't been compared with
// larft/larfb here yet.

/// T of Q = H_1 ... H_count = I - V T V^T (forward, column-wise V of height rows; tau from geqrf or larfg).
@inline(__always)
func wyFactor<Real: BandReductionScalar>(
  _ height: Int, _ count: Int, _ v: UnsafePointer<Real>, _ ldv: Int, _ tau: UnsafePointer<Real>,
  _ t: UnsafeMutablePointer<Real>, _ ldt: Int
) {
  Real.larft(height, count, v, ldv, tau, t, ldt)
}

/// C = op(Q) C (side "L") or C op(Q) (side "R"), op = "N" or "T", with Q = I - V T V^T. work holds ldwork * count,
/// ldwork >= columns of C ("L") or rows of C ("R").
@inline(__always)
func wyApply<Real: BandReductionScalar>(
  _ side: UnsafePointer<CChar>, _ trans: UnsafePointer<CChar>, _ m: Int, _ n: Int, _ count: Int,
  _ v: UnsafePointer<Real>, _ ldv: Int, _ t: UnsafePointer<Real>, _ ldt: Int, _ c: UnsafeMutablePointer<Real>,
  _ ldc: Int, _ work: UnsafeMutablePointer<Real>, _ ldwork: Int
) {
  Real.larfb(side, trans, m, n, count, v, ldv, t, ldt, c, ldc, work, ldwork)
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
