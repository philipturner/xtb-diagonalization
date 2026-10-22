// Runtime switches for the successive band reduction (SuccessiveBandReduction.swift), passed down to the stages.

/// Runtime switches for the structural optimizations of successive band reduction (first stage, last stage and the
/// bulge back-transformation). All off reproduces the development version. None of them adds threads.
public struct SBROptions {
  /// Stage 1 updates only the trailing block (rows and columns j ..< n of panel j) instead of the whole matrix.
  public var trailingOnly = false
  /// Stage 1 as one two-sided update (LAPACK's sy2sb): X = A V T, W = X - 1/2 V (T^T V^T X), A -= V W^T + W V^T.
  /// Implies trailingOnly.
  public var combinedUpdate = false
  /// With combinedUpdate: only the lower triangle (symm, syr2k), half the arithmetic; the band is mirrored at the end.
  public var lowerTriangle = false
  /// Stage 2 on the lower triangle: per bulge a symmetric rank-2 update of the reflector's block and rank-1 updates of
  /// the rows and columns around it, column-contiguous loops; about half the arithmetic and memory traffic.
  public var symmetricChase = false
  /// With symmetricChase: the chase runs on a copy of the lower band in band storage (leading dimension 2b) instead of
  /// in the n x n matrix: same arithmetic, (2b + 1) n elements in place of columns 8n bytes apart.
  public var compactBand = false
  /// With symmetricChase: every loop of the chase by explicit vector fused multiply-adds (VectorKernels.swift).
  public var vectorChase = false
  /// With lowerTriangle: the rank-2b update of the lower triangle by gemm per block column of 128 instead of
  /// Accelerate's syr2k, after which the bulge chase runs 15-30% slower (a hardware state, LOG.md).
  public var lowerGEMM = false
  /// With vectorChase: the chase's left and below blocks four columns per pass (the shared vector loaded once for
  /// all four).
  public var quadChase = false
  /// With vectorChase: eight columns per pass (about 17 of the 32 NEON registers), then four for a remainder of 4-7
  /// columns, then one. Implies quadChase.
  public var octChase = false
  /// Bulge back-transformation with triangular products at both ends of each parallelogram block (larfb treats the
  /// lower (b - 1) x b part as dense, though half of it is zero). Uses the precomputed blocks (prepareBulgeBlocks).
  public var parallelogram = false
  /// Back-transform the eigenvectors in blocks of this many columns (0: all at once), both stages per block.
  public var columnBlock = 0
  /// Every compact WY block applied as three gemm calls instead of Accelerate's larfb (WYTransform.swift), in the
  /// back-transformations and in the reductions that use larfb (BandForm without combinedUpdate, BandToBand).
  public var gemmWY = false
  /// Sweeps per block of the bulge back-transformation (1 ... b; 0 means b): fewer than b wastes less arithmetic on the
  /// block's zeros ((b + g - 1) / b times the reflectors' own) for a thinner gemm.
  public var bulgeGroup = 0
  /// Worker threads for one eigensolve (Parallel.swift); 1 runs everything on the calling thread. So far: stage 1's
  /// combined update (its gemm calls by row or column blocks; the panel QR and symm stay serial), the symmetric bulge
  /// chase (pipelined sweeps, same bits), divide and conquer (divideAndConquerParallel) and the 2-stage
  /// back-transformations (column blocks).
  public var threads = 1
  /// Workers for stage 1's update alone (0: `threads`). Its gemm calls share one AMX unit per cluster, so more
  /// workers can be slower (LOG.md).
  public var bandThreads = 0
  /// Workers for the bulge chase alone (0: `threads`). The pipeline gains at 2 and loses beyond (LOG.md).
  public var chaseThreads = 0
  /// Stage 1 and the bulge chase at once (reduceToTridiagonalOverlapped): the chase follows stage 1's frontier down
  /// the matrix on chaseThreads workers (0: threads - 1, at least 1). Same bits. Needs c and s, not b.
  public var overlapChase = false

  public init() {}

  /// Every structural optimization on.
  public static var all: SBROptions {
    var o = SBROptions()
    o.trailingOnly = true; o.combinedUpdate = true; o.lowerTriangle = true; o.symmetricChase = true
    o.compactBand = true; o.vectorChase = true; o.lowerGEMM = true; o.quadChase = true; o.octChase = true
    o.parallelogram = true; o.gemmWY = true
    return o
  }

  /// The measured best 2-stage configuration (LOG.md, 2026-10-21; M1 Max, GFN2 matrices n = 412-1788): options and
  /// the intermediate bandwidth b, for one process per core (manyProcesses, as 8 x 1 throughput) or one process alone.
  /// FP32 takes the gemm WY apply everywhere; FP64 only alone or below n = 700, since with 8 processes larfb wins at
  /// n >= 776 (AMX contention). From n = 700 the first stage updates only the lower triangle, by gemm blocks (not
  /// syr2k, after which the bulge chase runs 15-30% slower): 1-5% faster alone, 13-21% more throughput at 8 x 1 for
  /// n >= 1204 (FP32 n = 1204 unchanged). The chase's left and below blocks eight columns per pass: chase 10-17%
  /// faster with four, another 7-9% with eight, at n = 412-1788.
  public static func recommended(n: Int, fp32: Bool, manyProcesses: Bool) -> (options: SBROptions, bandwidth: Int) {
    var o = SBROptions()
    o.trailingOnly = true; o.combinedUpdate = true; o.symmetricChase = true; o.vectorChase = true
    o.quadChase = true; o.octChase = true
    o.lowerTriangle = n >= 700
    o.lowerGEMM = n >= 700
    o.gemmWY = fp32 || !manyProcesses || n < 700
    let b = n < 700 ? 32 : n < 1500 ? 48 : 64
    return (o, b)
  }
}
