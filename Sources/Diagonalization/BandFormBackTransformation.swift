// Back-transformation through the first stage (BandForm.swift), the last one applied: z = Q_1 Q_2 ... z with each
// panel's stored V and T (compact WY, WYTransform.swift), panels in reverse order. T isn't rebuilt: it was kept from the
// reduction.

/// Applies stage 1's panels to the first m columns of z (n x m, ldz): z = Q_1 Q_2 ... z.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBand<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v1: UnsafePointer<Real>, _ t1: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>,
  _ ldz: Int, _ m: Int, _ work: UnsafeMutablePointer<Real>
) {
  for j in bandPanels(n, b).reversed() {
    let r = j + b
    let rows = n - r
    let k = min(b, rows)
    wyApply("L", "N", rows, m, k, v1 + offset(r, j, n), n, t1 + offset(0, j, b), b, z + r, ldz, work, max(m, 1))
  }
}
