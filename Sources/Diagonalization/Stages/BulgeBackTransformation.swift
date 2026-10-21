// Back-transformation through the bulge chase (BulgeChasing.swift), the first one applied to the tridiagonal
// eigenvectors after divide and conquer: z = H_1 H_2 ... H_N z. The reflectors of b consecutive sweeps at the same bulge
// index are combined into one compact WY block (WYTransform.swift), either rebuilt on the fly just before it's applied
// (backTransformBulgesBlocked, as swift-xtb's draft does after divide and conquer) or built once in advance
// (prepareBulgeBlocks, for the parallelogram kernel and column blocks). backTransformBulges is the one-reflector-at-a-time
// reference.

/// Applies stage 2's reflectors to the first m columns of z (n x m, ldz): z = H_1 H_2 ... H_N z.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBulges<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v2: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int
) {
  guard b > 1 else { return }
  for s in stride(from: n - 3, through: 0, by: -1) {
    var kmax = 0
    while bulgeReflectorRows(n, b, s, kmax + 1).0 < n { kmax += 1 }
    for k in stride(from: kmax, through: 0, by: -1) {
      let (rs, re) = bulgeReflectorRows(n, b, s, k)
      let len = re - rs
      let w = v2 + offset(rs, s, n)
      for col in 0..<m {
        let zc = z + offset(rs, col, ldz)
        var dot: Real = 0
        for i in 0..<len { dot += w[i] * zc[i] }
        for i in 0..<len { zc[i] -= w[i] * dot }
      }
    }
  }
}

/// Workspace (elements) for backTransformBulgesBlocked.
public func bulgeBackTransformWorkspace(_ b: Int, _ m: Int) -> Int {
  2 * b * b + b * b + b + wyApplyWorkspace(m, b)
}

/// backTransformBulges with the reflectors of b consecutive sweeps at the same bulge index k combined into one block
/// reflector I - V T V^T (V: 2b - 1 rows, b columns, each column shifted down one row from the previous), applied with
/// larfb (swift-xtb's draft). Order: k ascending, sweep blocks descending, which equals the reverse of the reduction
/// order because every pair it swaps acts on disjoint rows: H(s, k) and H(s', k') with s > s' and k > k' are at least
/// b + 1 rows apart, and the same sweep's reflectors are disjoint. gemm: each block by three gemm calls instead of larfb
/// (SBROptions.gemmWY; V is built explicitly either way). group: sweeps per block (SBROptions.bulgeGroup, 1 ... b; 0
/// means b). The sequence of reflectors is the same for every group size: g sweeps make V b + g - 1 rows by g columns,
/// so the arithmetic is (b + g - 1) / b times the reflectors' own (2x at g = b) for a thinner gemm.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformBulgesBlocked<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v2: UnsafePointer<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int,
  _ work: UnsafeMutablePointer<Real>, gemm: Bool = false, group: Int = 0
) {
  guard b > 1, n > 2 else { return }
  let g = group > 0 ? min(group, b) : b
  let ldv = 2 * b
  let v = work, t = work + ldv * b, tau = t + b * b, scratch = tau + b
  let sweeps = n - 2
  var k = 0
  while bulgeReflectorRows(n, b, 0, k).0 < n {
    for s0 in stride(from: (sweeps - 1) / g * g, through: 0, by: -g) {
      let r0 = bulgeReflectorRows(n, b, s0, k).0
      guard r0 < n else { continue }
      let height = min(b + g - 1, n - r0)
      // Reflectors of sweeps s0 ..< s0 + count, column c starting at row c of the block.
      let count = min(g, sweeps - s0, height)
      for c in 0..<count {
        let s = s0 + c
        let (rs, re) = bulgeReflectorRows(n, b, s, k)
        let col = v + c * ldv
        for i in 0..<height { col[i] = 0 }
        let w = v2 + offset(rs, s, n)
        // w = sqrt(tau) [1, v(1 ...)]: tau = w0^2, v = w / w0.
        let w0 = w[0]
        if rs < n, w0 != 0 {
          tau[c] = w0 * w0
          col[c] = 1
          for i in 1..<(re - rs) { col[c + i] = w[i] / w0 }
        } else {
          tau[c] = 0
        }
      }
      wyFactor(height, count, v, ldv, tau, t, b)
      wyApply("L", "N", height, m, count, v, ldv, t, b, z + r0, ldz, scratch, max(m, 1), gemm: gemm)
    }
    k += 1
  }
}

/// Stage 2's reflectors as WY blocks, built once (backTransformBulgesBlocked rebuilds them on every call): the groups
/// of b consecutive sweeps at the same bulge index k, in application order. Group g acts on rows rowStart[g] ..<
/// rowStart[g] + height[g]; its V (leading dimension 2b, b columns; column c holds its vector in rows c ..< c + b with
/// an explicit unit first entry, zeros elsewhere, row 2b - 1 always zero) and T (b x b) sit at g * 2b * b and g * b * b.
public struct BulgeBlocks<Real: BandReductionScalar> {
  public var b: Int
  public var rowStart: [Int] = []
  public var height: [Int] = []
  public var count: [Int] = []
  public var v: [Real] = []
  public var t: [Real] = []

  public init(b: Int) { self.b = b }
}

@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func prepareBulgeBlocks<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ v2: UnsafePointer<Real>
) -> BulgeBlocks<Real> {
  var blocks = BulgeBlocks<Real>(b: b)
  guard b > 1, n > 2 else { return blocks }
  let ldv = 2 * b, sweeps = n - 2
  var tau = [Real](repeating: 0, count: b)
  var k = 0
  while bulgeReflectorRows(n, b, 0, k).0 < n {
    for s0 in stride(from: (sweeps - 1) / b * b, through: 0, by: -b) {
      let r0 = bulgeReflectorRows(n, b, s0, k).0
      guard r0 < n else { continue }
      let height = min(2 * b - 1, n - r0)
      let count = min(b, sweeps - s0, height)
      let vOffset = blocks.v.count
      blocks.v.append(contentsOf: repeatElement(0, count: ldv * b))
      blocks.t.append(contentsOf: repeatElement(0, count: b * b))
      blocks.v.withUnsafeMutableBufferPointer { vp in
        let v = vp.baseAddress! + vOffset
        for c in 0..<count {
          let s = s0 + c
          let (rs, re) = bulgeReflectorRows(n, b, s, k)
          let w = v2 + offset(rs, s, n)
          let w0 = rs < n ? w[0] : 0
          if w0 != 0 {
            tau[c] = w0 * w0
            v[offset(c, c, ldv)] = 1
            for i in 1..<(re - rs) { v[offset(c + i, c, ldv)] = w[i] / w0 }
          } else {
            tau[c] = 0
          }
        }
        blocks.t.withUnsafeMutableBufferPointer { tp in
          wyFactor(height, count, v, ldv, tau, tp.baseAddress! + blocks.rowStart.count * b * b, b)
        }
      }
      blocks.rowStart.append(r0)
      blocks.height.append(height)
      blocks.count.append(count)
    }
    k += 1
  }
  return blocks
}

/// Workspace (elements) for applyBulgeBlocks.
public func bulgeBlocksWorkspace(_ b: Int, _ m: Int) -> Int { 2 * b * m + wyApplyWorkspace(m, b) }

/// z = Q z for the first m columns of z with stage 2's precomputed blocks (as backTransformBulgesBlocked). With
/// parallelogram, each full block (2b - 1 rows, b vectors) goes through wyApplyParallelogram (triangular products at
/// both ends of V); the others through wyApply.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func applyBulgeBlocks<Real: BandReductionScalar>(
  _ blocks: BulgeBlocks<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int, parallelogram: Bool,
  _ work: UnsafeMutablePointer<Real>
) {
  let b = blocks.b, ldv = 2 * b
  let w1 = work, w2 = work + b * m, scratch = w2 + b * m
  blocks.v.withUnsafeBufferPointer { vp in
    blocks.t.withUnsafeBufferPointer { tp in
      for g in 0..<blocks.rowStart.count {
        let v = vp.baseAddress! + g * ldv * b, t = tp.baseAddress! + g * b * b
        let r0 = blocks.rowStart[g], height = blocks.height[g], count = blocks.count[g]
        let z1 = z + r0
        guard parallelogram, height == 2 * b - 1, count == b else {
          wyApply("L", "N", height, m, count, v, ldv, t, b, z1, ldz, scratch, max(m, 1))
          continue
        }
        wyApplyParallelogram(b, m, v, ldv, t, b, z1, ldz, w1, w2)
      }
    }
  }
}
