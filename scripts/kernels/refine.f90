!> Phase 3 offline study: how well do last cycle's eigenvectors, cheaply corrected, solve this cycle's eigenproblem?
!>
!>   refine PREFIX NEL UHF ETEMP_K [REPS]
!>
!> PREFIX_<k>.bin are per-cycle dumps from the diag/dump branch (int32 n, H(n,n), S(n,n), e(n), C(n,n)), where H and S
!> are the cycle's input and e, C its exact solution. For every cycle k >= 2 the eigenproblem H_k C = S C e is solved
!> from C_{k-1}, in the orthonormal basis of the Cholesky factor (S = U^T U, X = U^-1, H' = X^T H X, C' = U C):
!>   exact  dsyevd on H' (the current cost)
!>   A      pseudodiag (xTB's Stewart-Csaszar-Pulay, one occupied-virtual Jacobi sweep) of A = C'^T H' C'
!>   B1     one Ogita-Aishima refinement step of C'_{k-1}, near-degenerate pairs treated as clustered (E_ij = R_ij/2)
!>   B1c    as B1, but every cluster of near-degenerate eigenvalues is diagonalized exactly (small dsyev)
!>   B2c    two steps of B1c
!>   F32    Option C: ssyevd on H' rounded to FP32 (no reuse), vectors promoted back to FP64
!> Reported per cycle: max off-diagonal |A_ij| (how far C_{k-1} is from diagonalizing H_k; hypothesis H2), and per
!> method the time (ms, median of REPS) and the error of the Fermi-smeared density matrix in the orthonormal basis,
!> max |P - P_exact|, plus de_F (below). Occupations: alpha and beta
!> separately Fermi-smeared at ETEMP_K, as xTB does with nalpha = (NEL+UHF)/2, nbeta = (NEL-UHF)/2. de_F is the largest
!> eigenvalue error over the 10 orbitals nearest the Fermi level.
program refine
   use, intrinsic :: iso_fortran_env, only : dp => real64, sp => real32, int32, int64
   implicit none
   character(len=1024) :: prefix, arg, path
   integer :: nel, uhf, reps, k, n, info, i, lwork, liwork
   real(dp) :: etemp, kt
   real(dp), allocatable :: H(:, :), S(:, :), U(:, :), X(:, :), Hp(:, :), Cp(:, :), Cprev(:, :), ex(:), Cx(:, :)
   real(dp), allocatable :: A(:, :), T(:, :), Cn(:, :), en(:), work(:), Pex(:, :), Pap(:, :), times(:)
   real(dp), allocatable :: e_k(:), C_k(:, :), C2(:, :)
   integer, allocatable :: iwork(:)
   logical :: there
   real(dp), parameter :: autoev = 27.211386_dp, kb = 3.166808578545117e-06_dp

   call get_command_argument(1, prefix)
   call get_command_argument(2, arg); read(arg, *) nel
   call get_command_argument(3, arg); read(arg, *) uhf
   call get_command_argument(4, arg); read(arg, *) etemp
   reps = 5
   if (command_argument_count() >= 5) then
      call get_command_argument(5, arg); read(arg, *) reps
   end if
   kt = kb * etemp
   allocate(times(reps))

   k = 1
   call load(k, there)
   if (.not. there) stop 'no cycle 1'
   Cprev = C_k
   write(*, '(a)') ' cycle   max|A_ij|   max|A_ov|  | exact ms  | A: ms   dP      de_F   | B1: ms  dP      de_F   | B1c: ms dP      de_F   | B2c: ms dP      de_F   | F32: ms dP      de_F'
   do
      k = k + 1
      call load(k, there)
      if (.not. there) exit
      call cycle_study()
      Cprev = C_k
   end do

contains

   real(dp) function now()
      integer(int64) :: c, r
      call system_clock(c, r)
      now = real(c, dp) / real(r, dp)
   end function now

   subroutine load(kk, ok)
      integer, intent(in) :: kk
      logical, intent(out) :: ok
      integer :: iu
      integer(int32) :: n4
      write(path, '(a,"_",i0,".bin")') trim(prefix), kk
      inquire(file=path, exist=ok)
      if (.not. ok) return
      open(newunit=iu, file=path, access='stream', form='unformatted', status='old', action='read')
      read(iu) n4
      if (.not. allocated(H)) then
         n = n4
         allocate(H(n, n), S(n, n), U(n, n), X(n, n), Hp(n, n), Cp(n, n), Cx(n, n), A(n, n), T(n, n), Cn(n, n), &
            & en(n), ex(n), Pex(n, n), Pap(n, n), e_k(n), C_k(n, n), Cprev(n, n), C2(n, n))
         lwork = 1 + 6*n + 2*n*n; liwork = 3 + 5*n
         allocate(work(lwork), iwork(liwork))
      end if
      read(iu) H, S, e_k, C_k
      close(iu)
   end subroutine load

   !> occupations: alpha and beta Fermi-smeared separately
   subroutine occupations(e, f)
      real(dp), intent(in) :: e(:)
      real(dp), intent(out) :: f(:)
      real(dp) :: fa(size(e)), fb(size(e))
      call fermi(e, 0.5_dp * real(nel + uhf, dp), fa)
      call fermi(e, 0.5_dp * real(nel - uhf, dp), fb)
      f = fa + fb
   end subroutine occupations

   subroutine fermi(e, ne, f)
      real(dp), intent(in) :: e(:), ne
      real(dp), intent(out) :: f(:)
      real(dp) :: lo, hi, mu
      integer :: it
      lo = minval(e) - 1.0_dp; hi = maxval(e) + 1.0_dp
      do it = 1, 200
         mu = 0.5_dp * (lo + hi)
         f = 1.0_dp / (1.0_dp + exp(min(700.0_dp, (e - mu) / kt)))
         if (sum(f) > ne) then
            hi = mu
         else
            lo = mu
         end if
      end do
   end subroutine fermi

   real(dp) function fermi_level(e)
      real(dp), intent(in) :: e(:)
      real(dp) :: f(size(e))
      integer :: m
      call occupations(e, f)
      m = count(f > 1.0_dp)
      fermi_level = 0.5_dp * (e(max(1, m)) + e(min(n, m + 1)))
   end function fermi_level

   !> density in the orthonormal basis, P = C diag(f) C^T
   subroutine density(C, e, P)
      real(dp), intent(in) :: C(:, :), e(:)
      real(dp), intent(out) :: P(:, :)
      real(dp) :: f(size(e))
      integer :: j
      call occupations(e, f)
      do j = 1, n
         T(:, j) = C(:, j) * f(j)
      end do
      call dgemm('N', 'T', n, n, n, 1.0_dp, T, n, C, n, 0.0_dp, P, n)
   end subroutine density

   subroutine cycle_study()
      real(dp) :: t_ex, t(5), dpm(5), de(5), ef, offmax, ovmax
      integer :: r, m, j, nocc, nf
      ! orthonormal basis
      U = S
      call dpotrf('U', n, U, n, info)
      do j = 1, n
         U(j+1:, j) = 0.0_dp
      end do
      X = U
      call dtrtri('U', 'N', n, X, n, info)
      Hp = H
      call dtrmm('R', 'U', 'N', 'N', n, n, 1.0_dp, X, n, Hp, n)
      call dtrmm('L', 'U', 'T', 'N', n, n, 1.0_dp, X, n, Hp, n)
      ! previous eigenvectors in that basis
      Cp = Cprev
      call dtrmm('L', 'U', 'N', 'N', n, n, 1.0_dp, U, n, Cp, n)

      ! exact
      do r = 1, reps
         t_ex = now()
         Cx = Hp
         call dsyevd('V', 'U', n, Cx, n, ex, work, lwork, iwork, liwork, info)
         times(r) = 1e3_dp * (now() - t_ex)
      end do
      t_ex = median(times)
      call density(Cx, ex, Pex)
      ef = fermi_level(ex)
      nf = minloc(abs(ex - ef), 1)

      ! A = Cp^T H' Cp
      call rotate_in(Cp)
      offmax = 0.0_dp; ovmax = 0.0_dp
      nocc = (nel + uhf + 1) / 2
      do j = 1, n
         do m = 1, n
            if (m /= j) offmax = max(offmax, abs(A(m, j)))
         end do
      end do
      ovmax = maxval(abs(A(1:nocc, nocc+1:n)))

      do m = 1, 5
         do r = 1, reps
            t(m) = now()
            select case (m)
            case (1)
               call rotate_in(Cp); call method_pseudodiag(nocc)
            case (2)
               call rotate_in(Cp); call method_oa(.false.)
            case (3)
               call rotate_in(Cp); call method_oa(.true.)
            case (4)
               call rotate_in(Cp); call method_oa(.true.)
               C2 = Cn; call rotate_in(C2); call method_oa_from(C2, .true.)
            case (5)
               call method_fp32()
            end select
            times(r) = 1e3_dp * (now() - t(m))
         end do
         t(m) = median(times)
         call density(Cn, en, Pap)
         dpm(m) = maxval(abs(Pap - Pex))
         ! the 10 orbitals nearest the Fermi level
         de(m) = maxval(abs(en(max(1, nf-4):min(n, nf+5)) - ex(max(1, nf-4):min(n, nf+5))))
      end do
      write(*, '(i5,2es12.3," | ",f8.2," | ",5(f6.2,2es9.2," | "))') k, offmax, ovmax, t_ex, &
         & (t(m), dpm(m), de(m), m = 1, 5)
   end subroutine cycle_study

   !> A = C^T H' C for orthonormal-basis vectors C
   subroutine rotate_in(C)
      real(dp), intent(in) :: C(:, :)
      call dgemm('N', 'N', n, n, n, 1.0_dp, Hp, n, C, n, 0.0_dp, T, n)
      call dgemm('T', 'N', n, n, n, 1.0_dp, C, n, T, n, 0.0_dp, A, n)
   end subroutine rotate_in

   !> xTB's pseudodiag on A, then Cn = Cp V; eigenvalues = diag(A)
   subroutine method_pseudodiag(nocc)
      integer, intent(in) :: nocc
      real(sp), allocatable :: V(:, :)
      real(sp) :: al, be, x1, x2
      real(dp) :: c, d, e
      integer :: io, iv, j
      allocate(V(n, n)); V = 0.0_sp
      do j = 1, n
         V(j, j) = 1.0_sp; en(j) = A(j, j)
      end do
      do io = 1, nocc
         do iv = nocc + 1, n
            c = A(io, iv); d = A(io, io) - A(iv, iv)
            e = sign(sqrt(4.0_dp*c*c + d*d), d)
            al = real(sqrt(0.5_dp*(1.0_dp + d/e)), sp)
            be = real(-sign(sqrt(1.0_dp - real(al, dp)**2), c), sp)
            do j = 1, n
               x1 = V(j, io); x2 = V(j, iv)
               V(j, io) = al*x1 + be*x2
               V(j, iv) = al*x2 - be*x1
            end do
         end do
      end do
      T = real(V, dp)
      call dgemm('N', 'N', n, n, n, 1.0_dp, Cp, n, T, n, 0.0_dp, Cn, n)
   end subroutine method_pseudodiag

   !> Option C: the whole eigensolve in FP32
   subroutine method_fp32()
      real(sp), allocatable, save :: As(:, :), ws(:), works(:)
      integer, allocatable, save :: iworks(:)
      integer :: info3
      if (.not. allocated(As)) allocate(As(n, n), ws(n), works(1 + 6*n + 2*n*n), iworks(3 + 5*n))
      As = real(Hp, sp)
      call ssyevd('V', 'U', n, As, n, ws, works, size(works), iworks, size(iworks), info3)
      Cn = real(As, dp); en = real(ws, dp)
   end subroutine method_fp32

   subroutine method_oa(clusters)
      logical, intent(in) :: clusters
      call method_oa_from(Cp, clusters)
   end subroutine method_oa

   !> One Ogita-Aishima step for the standard problem H' from vectors C0 (A = C0^T H' C0 already formed):
   !> R = I - C0^T C0, lambda_i = A_ii / (1 - R_ii), E_ij = (A_ij + lambda_j R_ij) / (lambda_j - lambda_i) for
   !> separated pairs and R_ij / 2 for near-degenerate ones; Cn = C0 (I + E). With clusters, each group of
   !> near-degenerate eigenvalues is instead diagonalized exactly within its own block.
   subroutine method_oa_from(C0, clusters)
      real(dp), intent(in) :: C0(:, :)
      logical, intent(in) :: clusters
      real(dp), allocatable :: R(:, :), E(:, :), B(:, :), w(:), wk(:)
      integer, allocatable :: start(:)
      real(dp) :: delta, lam(n)
      integer :: i, j, nc, c0i, c1, m, info2
      allocate(R(n, n), E(n, n))
      call dsyrk('U', 'T', n, n, -1.0_dp, C0, n, 0.0_dp, R, n)
      do j = 1, n
         R(j, j) = R(j, j) + 1.0_dp
         do i = j + 1, n
            R(i, j) = R(j, i)
         end do
      end do
      do i = 1, n
         lam(i) = A(i, i) / (1.0_dp - R(i, i))
      end do
      ! near-degenerate threshold: twice the largest off-diagonal element, at least 1e-10 Eh
      delta = 1e-10_dp
      do j = 1, n
         do i = 1, n
            if (i /= j) delta = max(delta, 2.0_dp * abs(A(i, j)))
         end do
      end do
      delta = min(delta, 1e-3_dp)
      do j = 1, n
         do i = 1, n
            if (i == j) then
               E(i, j) = 0.5_dp * R(i, j)
            else if (abs(lam(j) - lam(i)) > delta) then
               E(i, j) = (A(i, j) + lam(j) * R(i, j)) / (lam(j) - lam(i))
            else
               E(i, j) = 0.5_dp * R(i, j)
            end if
         end do
      end do
      if (clusters) then
         ! clusters of consecutive near-degenerate eigenvalues (lam is nearly sorted): solve each block exactly
         i = 1
         do while (i <= n)
            c0i = i
            do while (i < n)
               if (abs(lam(i+1) - lam(i)) > delta) exit
               i = i + 1
            end do
            c1 = i
            nc = c1 - c0i + 1
            if (nc > 1) then
               allocate(B(nc, nc), w(nc), wk(3*nc))
               B = A(c0i:c1, c0i:c1)
               call dsyev('V', 'U', nc, B, nc, w, wk, 3*nc, info2)
               E(c0i:c1, c0i:c1) = B
               do m = c0i, c1
                  E(m, m) = E(m, m) - 1.0_dp
               end do
               lam(c0i:c1) = w
               deallocate(B, w, wk)
            end if
            i = i + 1
         end do
      end if
      do i = 1, n
         E(i, i) = E(i, i) + 1.0_dp
      end do
      call dgemm('N', 'N', n, n, n, 1.0_dp, C0, n, E, n, 0.0_dp, Cn, n)
      en = lam
   end subroutine method_oa_from

   real(dp) function median(v)
      real(dp), intent(in) :: v(:)
      real(dp) :: w(size(v)), x
      integer :: i, j
      w = v
      do i = 2, size(w)
         x = w(i); j = i - 1
         do while (j >= 1)
            if (w(j) <= x) exit
            w(j+1) = w(j); j = j - 1
         end do
         w(j+1) = x
      end do
      median = w((size(w)+1)/2)
   end function median

end program refine
