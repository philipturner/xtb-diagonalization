!> Tridiagonal eigensolvers on a real GFN2 matrix (BRIEF.md, Phase 2 driver comparison): after dsytrd, how fast are
!> all eigenvectors (dstedc, as dsyevd and grimme-lab/xtb#1466 do) against only the lowest M (MRRR, bisection +
!> inverse iteration), in FP64 and FP32?
!>
!>   tridiag HS.bin M [REPS]
!>
!> HS.bin is a dump from the diag/dump branch. M = number of lowest eigenpairs needed (occupied plus orbitals with
!> non-negligible Fermi occupation). Median of REPS (default 20) in ms, and the largest eigenvalue difference from
!> dstedc over the lowest M.
program tridiag
   use, intrinsic :: iso_fortran_env, only : dp => real64, sp => real32, int32, int64
   implicit none
   integer :: n, m, reps, iu, info, j, nfound
   integer(int32) :: n4
   real(dp), allocatable :: H(:, :), S(:, :), A(:, :), d(:), e(:), tau(:), work(:), Z(:, :), w(:), dref(:), d0(:), e0(:)
   real(dp), allocatable :: times(:), wl(:)
   integer, allocatable :: iwork(:), isuppz(:), iblock(:), isplit(:), ifail(:)
   real(sp), allocatable :: ds(:), es(:), Zs(:, :), works(:), ws(:)
   character(len=1024) :: path, arg
   logical :: tryrac

   call get_command_argument(1, path)
   call get_command_argument(2, arg); read(arg, *) m
   reps = 20
   if (command_argument_count() >= 3) then
      call get_command_argument(3, arg); read(arg, *) reps
   end if
   open(newunit=iu, file=path, access='stream', form='unformatted', status='old', action='read')
   read(iu) n4; n = n4
   allocate(H(n, n), S(n, n))
   read(iu) H, S
   close(iu)
   allocate(A(n, n), d(n), e(n), tau(n), Z(n, n), w(n), dref(n), d0(n), e0(n), times(reps), wl(n))
   allocate(work(1 + 4*n + n*n + 64*n), iwork(3 + 5*n + 10*n), isuppz(2*n), iblock(n), isplit(n), ifail(n))
   allocate(ds(n), es(n), Zs(n, n), works(1 + 4*n + n*n + 64*n), ws(n))

   ! H' = X^T H X, then the tridiagonal form T = (d0, e0)
   A = S
   call dpotrf('U', n, A, n, info)
   do j = 1, n
      A(j+1:, j) = 0.0_dp
   end do
   call dtrtri('U', 'N', n, A, n, info)
   Z = H
   call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, A, n, Z, n)
   call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, A, n, Z, n)
   A = Z
   call time_it('dsytrd (reduction, for scale)', 0)
   d0 = d; e0 = e

   write(*, '(a,i0,a,i0)') 'n = ', n, ', M = ', m
   call time_it('dstedc  all vectors      ', 1)
   dref = d
   call time_it('dstemr  lowest M         ', 2)
   call time_it('dstebz+dstein lowest M   ', 3)
   call time_it('dsterf  values only      ', 4)
   call time_it('sstedc  all vectors FP32 ', 5)
   call time_it('sstemr  lowest M FP32    ', 6)

contains

   real(dp) function now()
      integer(int64) :: c, r
      call system_clock(c, r)
      now = real(c, dp) / real(r, dp)
   end function now

   subroutine run(k)
      integer, intent(in) :: k
      real(sp) :: vls, vus
      select case (k)
      case (0)
         Z = A
         call dsytrd('U', n, Z, n, d, e, tau, work, size(work), info)
      case (1)
         d = d0; e = e0
         call dstedc('I', n, d, e, Z, n, work, size(work), iwork, size(iwork), info)
      case (2)
         d = d0; e = e0; tryrac = .true.
         call dstemr('V', 'I', n, d, e, 0.0_dp, 0.0_dp, 1, m, nfound, w, Z, n, m, isuppz, tryrac, &
            & work, size(work), iwork, size(iwork), info)
         d(1:m) = w(1:m)
      case (3)
         call dstebz('I', 'E', n, 0.0_dp, 0.0_dp, 1, m, 0.0_dp, d0, e0, nfound, j, w, iblock, isplit, &
            & work, iwork, info)
         call dstein(n, d0, e0, nfound, w, iblock, isplit, Z, n, work, iwork, ifail, info)
         d(1:m) = w(1:m)
      case (4)
         d = d0; wl = e0
         call dsterf(n, d, wl, info)
      case (5)
         ds = real(d0, sp); es = real(e0, sp)
         call sstedc('I', n, ds, es, Zs, n, works, size(works), iwork, size(iwork), info)
         d = real(ds, dp)
      case (6)
         ds = real(d0, sp); es = real(e0, sp); tryrac = .true.
         call sstemr('V', 'I', n, ds, es, 0.0_sp, 0.0_sp, 1, m, nfound, ws, Zs, n, m, isuppz, tryrac, &
            & works, size(works), iwork, size(iwork), info)
         d(1:m) = real(ws(1:m), dp)
      end select
      if (info /= 0) then
         write(*, *) 'info', info, 'in case', k
      end if
   end subroutine run

   subroutine time_it(label, k)
      character(len=*), intent(in) :: label
      integer, intent(in) :: k
      integer :: r, i, jj
      real(dp) :: t0, x
      call run(k)
      do r = 1, reps
         t0 = now(); call run(k); times(r) = 1e3_dp * (now() - t0)
      end do
      do i = 2, reps
         x = times(i); jj = i - 1
         do while (jj >= 1)
            if (times(jj) <= x) exit
            times(jj+1) = times(jj); jj = jj - 1
         end do
         times(jj+1) = x
      end do
      if (k >= 2) then
         write(*, '(a,f9.3,a,es9.2)') label, times((reps+1)/2), ' ms   max|de| lowest M vs dstedc ', &
            & maxval(abs(d(1:m) - dref(1:m)))
      else
         write(*, '(a,f9.3,a)') label, times((reps+1)/2), ' ms'
      end if
   end subroutine time_it

end program tridiag
