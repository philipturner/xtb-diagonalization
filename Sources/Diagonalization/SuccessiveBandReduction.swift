// Successive band reduction (SBR), the driver: a symmetric matrix -> tridiagonal form through a chain of bandwidths
// n - 1 -> b1 -> ... -> 1, each stage by its own routine (one file per step of the decomposition):
//   BandForm.swift                       dense -> b1
//   BandToBand.swift                     b_i -> b_{i+1}, the middle stages of 3 or more
//   BulgeChasing.swift                   b_last -> 1
//   DivideAndConquer.swift               the tridiagonal eigenproblem
//   BulgeBackTransformation.swift        eigenvectors back through the bulge chase
//   BandToBandBackTransformation.swift   ... through the middle stages
//   BandFormBackTransformation.swift     ... through the first stage
// with the compact WY transforms in WYTransform.swift and the runtime switches in SBROptions.swift. 2-stage is the chain
// [b], at no cost over a dedicated 2-stage driver; 3-stage [b1, b2]. 1-stage reduction is Tridiagonalization.swift.
// sbrReduceGeneral is a test reference: every stage by BandToBand.swift's general routine, any chain including [].

/// Test reference: successive band reduction of the symmetric matrix a (n x n, full storage) to tridiagonal form, every
/// stage by BandToBand.swift's general routine, through the given bandwidths (decreasing, each < n - 1 and > 1; [] reduces directly). Panel widths default to each stage's b_out.
/// Writes the diagonal to d (n) and subdiagonal to e (n - 1); returns the stages in order, for backTransform.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func sbrReduceGeneral<Real: BandReductionScalar>(
  _ n: Int, _ a: UnsafeMutablePointer<Real>, _ lda: Int, bandwidths: [Int], panels: [Int]? = nil,
  _ d: UnsafeMutablePointer<Real>, _ e: UnsafeMutablePointer<Real>, windowed: Bool = true
) -> [BandReductionStage<Real>] {
  let chain = [n - 1] + bandwidths.filter { $0 > 1 && $0 < n - 1 } + [1]
  var stages: [BandReductionStage<Real>] = []
  for s in 1..<chain.count where chain[s] < chain[s - 1] {
    let p = panels.map { $0[s - 1] } ?? chain[s]
    stages.append(reduceBandwidth(n, a, lda, bandIn: chain[s - 1], bandOut: chain[s], panel: p, windowed: windowed))
  }
  for i in 0..<n { d[i] = a[i + i * lda] }
  for i in 0..<max(0, n - 1) { e[i] = a[(i + 1) + i * lda] }
  return stages
}

/// Buffers for one SBR configuration on n x n matrices, allocated once (page-aligned) and reused across solves.
public final class SBRWorkspace<Real: BandReductionScalar> {
  public let n: Int
  /// n - 1, the intermediate bandwidths that fit n (decreasing), 1
  public let chain: [Int]
  public var options: SBROptions
  // First stage (dense -> chain[1], when chain[1] > 1): Householder vectors and T factors (reduceToBand layout).
  let v1: UnsafeMutablePointer<Real>, t1: UnsafeMutablePointer<Real>
  // Last stage (chain[count - 2] -> 1, when > 1): bulge reflectors (chaseBulges layout).
  let v2: UnsafeMutablePointer<Real>
  let work: UnsafeMutablePointer<Real>
  let workCount: Int
  // Middle stages, filled by reduce.
  var middle: [BandReductionStage<Real>] = []

  public init(n: Int, bandwidths: [Int], options: SBROptions = SBROptions(), vectors: Int? = nil) {
    var chain = [n - 1]
    for b in bandwidths + [1] where b < chain.last! && (b == 1 || b < n - 1) { chain.append(b) }
    self.n = n
    self.chain = chain
    self.options = options
    let b1 = chain.count > 2 ? chain[1] : 1, bt = chain.count > 2 ? chain[chain.count - 2] : 1
    let m = vectors ?? n
    workCount = max(bandReductionWorkspace(n, b1), bulgeChasingWorkspace(bt), bulgeBackTransformWorkspace(bt, m), 1)
    func alloc(_ count: Int) -> UnsafeMutablePointer<Real> {
      let p = UnsafeMutableRawPointer.allocate(byteCount: max(count, 1) * MemoryLayout<Real>.stride, alignment: 16384)
        .bindMemory(to: Real.self, capacity: max(count, 1))
      p.initialize(repeating: 0, count: max(count, 1))
      return p
    }
    v1 = alloc(chain.count > 2 ? n * n : 1)
    t1 = alloc(chain.count > 2 ? b1 * n : 1)
    v2 = alloc(chain.count > 2 ? n * n : 1)
    work = alloc(workCount)
  }

  deinit {
    v1.deallocate(); t1.deallocate(); v2.deallocate(); work.deallocate()
  }

  /// Names of the reduction stages, e.g. ["n>64", "64>16", "16>1"].
  public var stageNames: [String] {
    (1..<chain.count).map { "\(chain[$0 - 1] == n - 1 ? "n" : String(chain[$0 - 1]))>\(chain[$0])" }
  }
}

/// Successive band reduction of the symmetric matrix a (n x n, full storage) to tridiagonal form along
/// workspace.chain, each stage by its fastest routine: the first (dense -> b1) by reduceToBand, the last (b -> 1) by
/// chaseBulges, any in between by reduceBandwidth; the SBROptions apply to the first and last. 2-stage is the
/// chain [b], at no extra cost. Writes the diagonal to d and the subdiagonal to e. stageTime (optional) receives each
/// stage's seconds.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func sbrReduce<Real: BandReductionScalar>(
  _ ws: SBRWorkspace<Real>, _ a: UnsafeMutablePointer<Real>, _ lda: Int, _ d: UnsafeMutablePointer<Real>,
  _ e: UnsafeMutablePointer<Real>, stageTime: ((Int) -> Void)? = nil
) {
  let n = ws.n, chain = ws.chain
  precondition(chain.count > 2, "SBR needs an intermediate bandwidth (1-stage is Tridiagonalization.swift)")
  ws.middle = []
  // The first stage may skip the upper triangle only when the next stage is the symmetric bulge chase.
  var first = ws.options
  if chain.count > 3 { first.symmetricChase = false; first.lowerTriangle = false }
  reduceToBand(n, chain[1], a, lda, ws.v1, ws.t1, ws.work, first)
  stageTime?(0)
  for s in 2..<(chain.count - 1) {
    ws.middle.append(reduceBandwidth(n, a, lda, bandIn: chain[s - 1], bandOut: chain[s], panel: chain[s]))
    stageTime?(s - 1)
  }
  chaseBulges(n, chain[chain.count - 2], a, lda, ws.v2, d, e, ws.work, ws.options)
  stageTime?(chain.count - 2)
}

/// z = Q z for the first m columns of z (n x m, ldz), Q the product of every stage of the last sbrReduce, applied in
/// reverse stage order. stageTime (optional) receives each back-transformation's index in stage order.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func sbrBackTransform<Real: BandReductionScalar>(
  _ ws: SBRWorkspace<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int, stageTime: ((Int) -> Void)? = nil
) {
  let n = ws.n, chain = ws.chain
  precondition(chain.count > 2, "SBR needs an intermediate bandwidth")
  backTransformBulgesBlocked(n, chain[chain.count - 2], ws.v2, z, ldz, m, ws.work)
  stageTime?(chain.count - 2)
  for s in stride(from: chain.count - 2, through: 2, by: -1) {
    backTransform(ws.middle[s - 2], z, ldz, m)
    stageTime?(s - 1)
  }
  backTransformBand(n, chain[1], ws.v1, ws.t1, z, ldz, m, ws.work)
  stageTime?(0)
}

/// Both back-transformations of the 2-stage path with the precomputed bulge blocks: z = Q1 Q2 z for the first m
/// columns, in blocks of options.columnBlock columns (both stages per block while it is in cache). work holds
/// max(bulgeBlocksWorkspace(b, block), bandReductionWorkspace(n, b)).
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransformTwoStage<Real: BandReductionScalar>(
  _ n: Int, _ b: Int, _ blocks: BulgeBlocks<Real>, _ v1: UnsafePointer<Real>, _ t1: UnsafePointer<Real>,
  _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int, _ options: SBROptions, _ work: UnsafeMutablePointer<Real>
) {
  let block = options.columnBlock > 0 ? options.columnBlock : m
  for c0 in stride(from: 0, to: m, by: block) {
    let width = min(block, m - c0)
    applyBulgeBlocks(blocks, z + c0 * ldz, ldz, width, parallelogram: options.parallelogram, work)
    backTransformBand(n, b, v1, t1, z + c0 * ldz, ldz, width, work)
  }
}
