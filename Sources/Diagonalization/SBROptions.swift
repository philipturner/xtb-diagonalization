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

  public init() {}

  /// Every structural optimization on.
  public static var all: SBROptions {
    var o = SBROptions()
    o.trailingOnly = true; o.combinedUpdate = true; o.lowerTriangle = true; o.symmetricChase = true
    o.parallelogram = true; o.gemmWY = true
    return o
  }
}
