// swift-tools-version: 5.9
// Swift prototypes of the eigensolver kernels, linkable into libxtb (see Sources/Diagonalization/Exports.swift and
// scripts/build_xtb_swift.sh). Build with `swift build -c release`.
import PackageDescription

let package = Package(
  name: "xtb-diagonalization",
  platforms: [.macOS(.v13)],
  products: [
    // Linked into libxtb.dylib in place of xtb/src/mctc/lapack/sytrd.c.
    .library(name: "Diagonalization", type: .static, targets: ["Diagonalization"]),
    .executable(name: "sytrd-bench", targets: ["SytrdBench"]),
    .executable(name: "eigen-bench", targets: ["EigenBench"]),
  ],
  dependencies: [
    // ManagedAtomic for the work counters of the parallel stages (Parallel.swift).
    .package(url: "https://github.com/apple/swift-atomics.git", from: "1.2.0"),
  ],
  targets: [
    // The legacy Fortran-ABI BLAS/LAPACK symbols in Accelerate, the ones libxtb links (BLA_VENDOR=Apple).
    .target(
      name: "CLAPACK",
      linkerSettings: [.linkedFramework("Accelerate")]),
    // Vector fused multiply-add for the Swift kernels.
    .target(name: "CSIMD"),
    .target(
      name: "Diagonalization",
      dependencies: ["CLAPACK", "CSIMD", .product(name: "Atomics", package: "swift-atomics")],
      swiftSettings: [.unsafeFlags(["-enforce-exclusivity=unchecked"])]),
    .executableTarget(
      name: "SytrdBench",
      dependencies: ["CLAPACK", "Diagonalization"]),
    .executableTarget(
      name: "EigenBench",
      dependencies: ["CLAPACK", "Diagonalization"]),
  ]
)
