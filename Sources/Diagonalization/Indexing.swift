// Element offsets in column-major matrices, shared by the band-reduction files. (Tridiagonalization.swift has its own
// 1-based helper, as in the LAPACK code it ports.)



/// Offset of element (i, j), 0-based, in a column-major matrix with leading dimension ld.
@inline(__always)
func offset(_ i: Int, _ j: Int, _ ld: Int) -> Int { i + j * ld }
