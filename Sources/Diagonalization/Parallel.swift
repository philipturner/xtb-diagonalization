// Bounded parallelism for one eigensolve (SBROptions.threads): a fixed number of workers, not one task per block.
// parallelBlocks runs DispatchQueue.concurrentPerform with exactly `threads` iterations; each worker claims the next
// block from a shared atomic counter until it passes the block count, so the number of blocks (which scales with the
// problem) and the number of threads (2 / 4 / 8) are independent. Workers that must wait on another worker's progress
// (Progress.wait) spin on an atomic with a deadline and trap if it passes, instead of hanging.
import Atomics
import Darwin
import Dispatch

/// A deadline for spin-waits: a stalled pipeline traps after this many seconds instead of spinning forever.
let parallelStallSeconds = 10.0

/// Runs body(block) for every block in 0 ..< blocks on at most `threads` workers (blocks claimed in increasing order;
/// threads <= 1 runs them in order on the calling thread). Returns when every block has finished.
@inline(__always)
func parallelBlocks(_ blocks: Int, threads: Int, _ body: (Int) -> Void) {
  guard threads > 1, blocks > 1 else {
    for b in 0..<blocks { body(b) }
    return
  }
  let next = ManagedAtomic<Int>(0)
  DispatchQueue.concurrentPerform(iterations: min(threads, blocks)) { _ in
    while true {
      let b = next.loadThenWrappingIncrement(ordering: .relaxed)
      if b >= blocks { break }
      body(b)
    }
  }
}

/// Progress counters shared by pipelined workers (e.g. one per sweep of the bulge chase): set publishes a value with
/// release ordering, wait spins (acquire) until a counter reaches a value, trapping after parallelStallSeconds.
final class Progress {
  let counters: [ManagedAtomic<Int>]

  init(count: Int, initial: Int = 0) {
    counters = (0..<count).map { _ in ManagedAtomic<Int>(initial) }
  }

  @inline(__always)
  func set(_ i: Int, _ value: Int) {
    counters[i].store(value, ordering: .releasing)
  }

  @inline(__always)
  func wait(_ i: Int, atLeast value: Int) {
    if counters[i].load(ordering: .acquiring) >= value { return }
    let start = DispatchTime.now().uptimeNanoseconds
    var spins = 0
    while counters[i].load(ordering: .acquiring) < value {
      spins += 1
      if spins & 1023 == 0 {
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
        precondition(elapsed < parallelStallSeconds, "parallel pipeline stalled waiting on counter \(i) >= \(value)")
      }
    }
  }
}

/// A spinning barrier for a team of exactly `threads` workers that all run at once (tridiagonalizeParallel: about
/// four per column of the reduction, too many for anything that sleeps). Traps after parallelStallSeconds.
final class SpinBarrier {
  let count = ManagedAtomic<Int>(0)
  let generation = ManagedAtomic<Int>(0)
  let threads: Int

  init(threads: Int) { self.threads = threads }

  @inline(__always)
  func wait() {
    let g = generation.load(ordering: .acquiring)
    if count.loadThenWrappingIncrement(ordering: .acquiringAndReleasing) == threads - 1 {
      count.store(0, ordering: .relaxed)
      generation.store(g + 1, ordering: .releasing)
      return
    }
    let start = DispatchTime.now().uptimeNanoseconds
    var spins = 0
    while generation.load(ordering: .acquiring) == g {
      spins += 1
      if spins & 1023 == 0 {
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
        precondition(elapsed < parallelStallSeconds, "barrier stalled with \(threads) workers")
      }
    }
  }
}

/// Performance cores per cluster: hw.perflevel0.physicalcpu over its clusters, ceil(physicalcpu / cpusperl2), so a
/// cluster with disabled cores counts the enabled ones whether cpusperl2 reports enabled or designed cores (6
/// performance cores, cpusperl2 4 or 3: 3); 4 where the sysctls are missing. tridiagonalizeParallel's team stays
/// within one cluster: on the M1 Max (4 per cluster), 6 and 8 workers were slower than 4.
public let performanceClusterSize: Int = {
  func query(_ name: String) -> Int {
    var value: Int32 = 0
    var size = MemoryLayout<Int32>.size
    return sysctlbyname(name, &value, &size, nil, 0) == 0 ? Int(value) : 0
  }
  let cores = query("hw.perflevel0.physicalcpu"), shared = query("hw.perflevel0.cpusperl2")
  guard cores > 0 && shared > 0 else { return 4 }
  return cores / ((cores + shared - 1) / shared)
}()
