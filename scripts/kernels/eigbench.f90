!> Benchmark of the per-SCC-cycle generalized eigensolve on a real GFN2 matrix pair (BRIEF.md, Phase 2).
!>
!>   eigbench HS.bin NLOW [REPS]
!>
!> HS.bin is a dump from the diag/dump branch (int32 n, H(n,n), S(n,n), ...). NLOW is the number of lowest eigenpairs
!> for the partial dsyevr (occupied orbitals plus a margin). Every variant runs REPS times (default 30) after one
!> warm-up; the median is printed in ms. The Cholesky factor U of S (S = U^T U) and X = U^-1 are formed once, as once
!> per geometry in xTB, and timed separately.
!>   A  dsygst + dsyevd + dtrsm                         (xTB main, mctc_dsygvd_factorized)
!>   B  dtrmm (H X) + dtrmm (X^T .) + dsyevd + dtrmm (X C')
!>   C  dgemm (H X) + dgemm (X^T .) + dsyevd + dgemm (X C')
!> and on the transformed H' = X^T H X: dsyevd, dsyevr (all), dsyevr (lowest NLOW).
!> Accuracy of each variant against A: max |eigenvalue difference| over the lowest NLOW, and the residual
!> max |H C - S C diag(e)| and orthonormality max |C^T S C - I| over those eigenvectors.
program eigbench
   use, intrinsic :: iso_fortran_env, only : dp => real64, int32, int64
   implicit none
   integer :: n, nlow, reps, iu, i, info, lwork, liwork, m
   integer(int32) :: n4
   real(dp), allocatable :: H(:, :), S(:, :), U(:, :), X(:, :), A(:, :), T(:, :), C(:, :), W(:, :)
   real(dp), allocatable :: e(:), eref(:), Cref(:, :), work(:)
   integer, allocatable :: iwork(:), isuppz(:)
   real(dp), allocatable :: times(:)
   character(len=1024) :: path, arg
   real(dp) :: t_potrf, t_trtri, dummy
   real(dp), external :: dlamch

   call get_command_argument(1, path)
   call get_command_argument(2, arg); read(arg, *) nlow
   reps = 30
   if (command_argument_count() >= 3) then
      call get_command_argument(3, arg); read(arg, *) reps
   end if
   open(newunit=iu, file=path, access='stream', form='unformatted', status='old', action='read')
   read(iu) n4; n = n4
   allocate(H(n, n), S(n, n))
   read(iu) H, S
   close(iu)
   allocate(U(n, n), X(n, n), A(n, n), T(n, n), C(n, n), W(n, n), e(n), eref(n), Cref(n, n), times(reps))
   lwork = 1 + 6*n + 2*n*n; liwork = 3 + 5*n
   allocate(work(max(lwork, 26*n)), iwork(max(liwork, 10*n)), isuppz(2*n))

   ! once per geometry
   t_potrf = timeit_potrf()
   X = U
   t_trtri = timeit_trtri()

   write(*, '(a,i0,a,i0,a,i0)') 'n = ', n, ', nlow = ', nlow, ', reps = ', reps
   write(*, '(a,f8.3,a,f8.3,a)') 'once per geometry: dpotrf ', t_potrf, ' ms, dtrtri ', t_trtri, ' ms'

   ! A: reference path
   call run('A  sygst+syevd+trsm      ', 1)
   eref = e; Cref = A
   call report('A', e, A)
   call run('B  trmm x2+syevd+trmm    ', 2)
   call report('B', e, A)
   call run('C  gemm x2+syevd+gemm    ', 3)
   call report('C', e, A)
   ! stages of B, and drivers on H'
   call run('B: transform only (2 trmm)', 4)
   call run('A: transform only (sygst) ', 5)
   call run('B: back-transform (trmm)  ', 6)
   call run('A: back-transform (trsm)  ', 7)
   call run('C: transform only (2 gemm)', 8)
   call run('C: back-transform (gemm)  ', 9)
   call run("H': dsyevd                ", 10)
   call run("H': dsyevr all            ", 11)
   call run("H': dsyevr lowest NLOW    ", 12)
   ! accuracy of the dsyevr variants after back-transformation (B's transforms)
   call run('B with dsyevr all         ', 13)
   call report('B/syevr-all', e, A)
   call run('B with dsyevr lowest NLOW ', 14)
   call report('B/syevr-low', e, A)

contains

   real(dp) function now()
      integer(int64) :: c, r
      call system_clock(c, r)
      now = real(c, dp) / real(r, dp)
   end function now

   real(dp) function timeit_potrf()
      real(dp) :: t0
      U = S
      t0 = now()
      call dpotrf('U', n, U, n, info)
      timeit_potrf = 1e3_dp * (now() - t0)
      do i = 1, n
         U(i+1:, i) = 0.0_dp
      end do
   end function timeit_potrf

   real(dp) function timeit_trtri()
      real(dp) :: t0
      t0 = now()
      call dtrtri('U', 'N', n, X, n, info)
      timeit_trtri = 1e3_dp * (now() - t0)
   end function timeit_trtri

   subroutine variant(k)
      integer, intent(in) :: k
      select case (k)
      case (1)
         A = H
         call dsygst(1, 'U', n, A, n, U, n, info)
         call dsyevd('V', 'U', n, A, n, e, work, lwork, iwork, liwork, info)
         call dtrsm('L', 'U', 'N', 'N', n, n, 1.0_dp, U, n, A, n)
      case (2)
         A = H
         call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, X, n, A, n)
         call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, X, n, A, n)
         call dsyevd('V', 'U', n, A, n, e, work, lwork, iwork, liwork, info)
         call dtrmm('L', 'U', 'N', 'N', n, n, 1.0_dp, X, n, A, n)
      case (3)
         call dgemm('N', 'N', n, n, n, 1.0_dp, H, n, X, n, 0.0_dp, T, n)
         call dgemm('T', 'N', n, n, n, 1.0_dp, X, n, T, n, 0.0_dp, A, n)
         call dsyevd('V', 'U', n, A, n, e, work, lwork, iwork, liwork, info)
         call dgemm('N', 'N', n, n, n, 1.0_dp, X, n, A, n, 0.0_dp, T, n)
         A = T
      case (4)
         A = H
         call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, X, n, A, n)
         call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, X, n, A, n)
      case (5)
         A = H
         call dsygst(1, 'U', n, A, n, U, n, info)
      case (6)
         A = Cref
         call dtrmm('L', 'U', 'N', 'N', n, n, 1.0_dp, X, n, A, n)
      case (7)
         A = Cref
         call dtrsm('L', 'U', 'N', 'N', n, n, 1.0_dp, U, n, A, n)
      case (8)
         call dgemm('N', 'N', n, n, n, 1.0_dp, H, n, X, n, 0.0_dp, T, n)
         call dgemm('T', 'N', n, n, n, 1.0_dp, X, n, T, n, 0.0_dp, A, n)
      case (9)
         call dgemm('N', 'N', n, n, n, 1.0_dp, X, n, Cref, n, 0.0_dp, T, n)
      case (10)
         A = W
         call dsyevd('V', 'U', n, A, n, e, work, lwork, iwork, liwork, info)
      case (11)
         A = W
         call dsyevr('V', 'A', 'U', n, A, n, 0.0_dp, 0.0_dp, 1, n, 0.0_dp, m, e, C, n, isuppz, &
            & work, 26*n, iwork, 10*n, info)
      case (12)
         A = W
         call dsyevr('V', 'I', 'U', n, A, n, 0.0_dp, 0.0_dp, 1, nlow, 0.0_dp, m, e, C, n, isuppz, &
            & work, 26*n, iwork, 10*n, info)
      case (13, 14)
         A = H
         call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, X, n, A, n)
         call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, X, n, A, n)
         if (k == 13) then
            call dsyevr('V', 'A', 'U', n, A, n, 0.0_dp, 0.0_dp, 1, n, 0.0_dp, m, e, C, n, isuppz, &
               & work, 26*n, iwork, 10*n, info)
         else
            call dsyevr('V', 'I', 'U', n, A, n, 0.0_dp, 0.0_dp, 1, nlow, 0.0_dp, m, e, C, n, isuppz, &
               & work, 26*n, iwork, 10*n, info)
         end if
         call dtrmm('L', 'U', 'N', 'N', n, m, 1.0_dp, X, n, C, n)
         A(:, :m) = C(:, :m)
      end select
      if (info /= 0) then
         write(*, *) 'info', info, 'in variant', k
         stop 1
      end if
   end subroutine variant

   subroutine run(label, k)
      character(len=*), intent(in) :: label
      integer, intent(in) :: k
      integer :: r
      real(dp) :: t0
      if (k >= 10 .and. k <= 12) then  ! drivers act on the transformed matrix H'
         W = H
         call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, X, n, W, n)
         call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, X, n, W, n)
      end if
      call variant(k)  ! warm-up
      do r = 1, reps
         t0 = now(); call variant(k); times(r) = 1e3_dp * (now() - t0)
      end do
      call sort(times)
      write(*, '(a,a,f9.3,a,f9.3,a,f9.3,a)') label, '  median ', times((reps+1)/2), ' ms  (min ', times(1), &
         & ', max ', times(reps), ')'
   end subroutine run

   subroutine report(label, ev, Cv)
      character(len=*), intent(in) :: label
      real(dp), intent(in) :: ev(:), Cv(:, :)
      real(dp), allocatable :: R1(:, :), R2(:, :), O(:, :)
      integer :: j
      allocate(R1(n, nlow), R2(n, nlow), O(nlow, nlow))
      call dgemm('N', 'N', n, nlow, n, 1.0_dp, H, n, Cv, n, 0.0_dp, R1, n)
      call dgemm('N', 'N', n, nlow, n, 1.0_dp, S, n, Cv, n, 0.0_dp, R2, n)
      call dgemm('T', 'N', nlow, nlow, n, 1.0_dp, Cv, n, R2, n, 0.0_dp, O, nlow)
      do j = 1, nlow
         R1(:, j) = R1(:, j) - ev(j) * R2(:, j)
         O(j, j) = O(j, j) - 1.0_dp
      end do
      write(*, '(3x,a,a,es9.2,a,es9.2,a,es9.2)') label, ': max|de| vs A ', maxval(abs(ev(:nlow) - eref(:nlow))), &
         & '   residual ', maxval(abs(R1)), '   orthonormality ', maxval(abs(O))
   end subroutine report

   subroutine sort(v)
      real(dp), intent(inout) :: v(:)
      integer :: i, j
      real(dp) :: x
      do i = 2, size(v)
         x = v(i); j = i - 1
         do while (j >= 1)
            if (v(j) <= x) exit
            v(j+1) = v(j); j = j - 1
         end do
         v(j+1) = x
      end do
   end subroutine sort

end program eigbench
