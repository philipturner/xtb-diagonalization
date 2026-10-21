// Back-transformation through the middle stages (BandToBand.swift): each step's stored compact WY block, applied in
// reverse order of the reduction, one block per step (no grouping across sweeps yet).

/// z = Q z for the first m columns of z (n x m, ldz), Q = Q_1 Q_2 ... the stage's steps in application order.
@_specialize(exported: true, where Real == Double)
@_specialize(exported: true, where Real == Float)
public func backTransform<Real: BandReductionScalar>(
  _ stage: BandReductionStage<Real>, _ z: UnsafeMutablePointer<Real>, _ ldz: Int, _ m: Int
) {
  var work = [Real](repeating: 0, count: max(1, m * stage.panel))
  stage.v.withUnsafeBufferPointer { v in
    stage.t.withUnsafeBufferPointer { t in
      work.withUnsafeMutableBufferPointer { wp in
        for step in stage.steps.reversed() {
          wyApply("L", "N", step.height, m, step.count, v.baseAddress! + step.vOffset, step.height,
                     t.baseAddress! + step.tOffset, step.count, z + step.rowStart, ldz, wp.baseAddress!, max(m, 1))
        }
      }
    }
  }
}
