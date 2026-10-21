// Step 2 of successive band reduction (3 or more stages only): bandwidth b_in -> b_out, any number of times
// (Bischof, Lang and Sun's band reduction), panel width p (1 <= p <= b_out < b_in), lower band, full storage:
//   sweep j (j = 0, p, 2p, ...): columns j ..< j + p have nonzeros below their b_out band, rows j + b_out ..< j + p + b_in.
//   step 0  QR of A[j + b_out ..< j + b_out + h, j ..< j + p], h = p + b_in - b_out: R keeps bandwidth b_out for these
//           columns. Applying Q from the right mixes columns j + b_out ..< j + b_out + h, which fills rows below them
//           beyond the b_in band: a bulge.
//   step k  QR of A[rs ..< rs + h, cs ..< cs + p] with cs = the previous step's first row, rs = cs + b_in: the bulge's
//           first p columns go back to bandwidth b_in. The rest of the bulge belongs to the next sweep's columns.
// Every step is one compact WY block (geqrf, then T from WYTransform.swift) applied to both sides on a window around
// it; V and T are kept per step for BandToBandBackTransformation.swift. The routine is general: with b_in = n - 1 it
// reduces a dense matrix (as BandForm.swift, slower) and with b_out = p = 1 it is the classical bulge chase (as
// BulgeChasing.swift, much slower); the driver uses it only for the middle stages. Correctness first: full storage,
// no symmetry, a generous window.

/// One step's block reflector Q = I - V T V^T, acting on rows rowStart ..< rowStart + height.
public struct BandReductionStep {
  public var rowStart: Int
  public var height: Int
  /// Number of Householder vectors (columns of V)
  public var count: Int
  /// Offsets of V (height x count, leading dimension height, geqrf layout) and T (count x count) in the stage's storage
  public var vOffset: Int
  public var tOffset: Int
}

/// The reflectors of one stage, in the order they were applied.
public struct BandReductionStage<Real: BandReductionScalar> {
  public var bandIn: Int
  public var bandOut: Int
  public var panel: Int
  public var steps: [BandReductionStep] = []
  public var v: [Real] = []
  public var t: [Real] = []
  /// Largest |A(i, j)| outside the b_out band before it was cleared (should be rounding-level)
  public var leftover: Real = 0
}

/// Reduces the symmetric matrix a (n x n, full storage) from bandwidth bandIn to bandOut, in place, with panel width
/// panel (1 ... bandOut). Entries outside the new band are set to zero (their largest magnitude is reported as
/// leftover). windowed = false applies every step to the whole matrix (slow; a cross-check of the window).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func reduceBandwidth<Real: BandReductionScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, bandIn: Int, bandOut: Int, panel: Int,
  windowed: Bool = true
) -> BandReductionStage<Real> {
  precondition(bandOut >= 1 && bandOut < bandIn && panel >= 1 && panel <= bandOut, "invalid band reduction")
  var stage = BandReductionStage<Real>(bandIn: bandIn, bandOut: bandOut, panel: panel)
  let h = panel + bandIn - bandOut
  var block = [Real](repeating: 0, count: h * panel)
  var tau = [Real](repeating: 0, count: panel)
  var tBlock = [Real](repeating: 0, count: panel * panel)
  var work = [Real](repeating: 0, count: max(64 * panel, n * panel))

  for j in stride(from: 0, to: max(0, n - bandOut - 1), by: panel) {
    var cs = j, rs = j + bandOut
    while rs < n - 1 {
      let height = min(h, n - rs)
      let width = min(panel, n - cs)
      let count = min(height, width)
      // QR of the block A[rs ..< rs + height, cs ..< cs + width].
      for c in 0..<width {
        for i in 0..<height { block[i + c * height] = a[(rs + i) + (cs + c) * lda] }
      }
      block.withUnsafeMutableBufferPointer { v in
        tau.withUnsafeMutableBufferPointer { tp in
          work.withUnsafeMutableBufferPointer { wp in
            tBlock.withUnsafeMutableBufferPointer { t in
              Real.geqrf(height, width, v.baseAddress!, height, tp.baseAddress!, wp.baseAddress!, wp.count)
              wyFactor(height, count, v.baseAddress!, height, tp.baseAddress!, t.baseAddress!, panel)
            }
          }
        }
      }
      // The window holding every nonzero of rows and columns rs ..< rs + height (generous: bulges reach about
      // 2 b_in + h beyond the block).
      let margin = 2 * (bandIn + h)
      let lo = windowed ? max(0, rs - margin) : 0
      let hi = windowed ? min(n, rs + height + margin) : n
      block.withUnsafeBufferPointer { v in
        tBlock.withUnsafeBufferPointer { t in
          work.withUnsafeMutableBufferPointer { wp in
            // A = Q^T A Q on the window.
            wyApply("L", "T", height, hi - lo, count, v.baseAddress!, height, t.baseAddress!, panel,
                       a + rs + lo * lda, lda, wp.baseAddress!, hi - lo)
            wyApply("R", "N", hi - lo, height, count, v.baseAddress!, height, t.baseAddress!, panel,
                       a + lo + rs * lda, lda, wp.baseAddress!, hi - lo)
          }
        }
      }
      // Exact zeros below R in the annihilated columns, both triangles.
      for c in 0..<width {
        for i in (c + 1)..<max(height, c + 1) {
          a[(rs + i) + (cs + c) * lda] = 0
          a[(cs + c) + (rs + i) * lda] = 0
        }
      }
      // Keep V and T for the back-transformation.
      let vOffset = stage.v.count, tOffset = stage.t.count
      stage.v.append(contentsOf: block[0..<(height * count)])
      for c in 0..<count {
        for r in 0..<count { stage.t.append(tBlock[r + c * panel]) }
      }
      stage.steps.append(BandReductionStep(
        rowStart: rs, height: height, count: count, vOffset: vOffset, tOffset: tOffset))
      // Next step: the bulge's first columns.
      cs = rs
      rs += bandIn
    }
  }
  for j in 0..<n {
    for i in 0..<n where abs(i - j) > bandOut {
      stage.leftover = max(stage.leftover, abs(a[i + j * lda]))
      a[i + j * lda] = 0
    }
  }
  return stage
}
