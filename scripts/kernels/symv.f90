!> Is Accelerate's dsymv the bottleneck of dsytrd? Symmetric matrix-vector product y = A x (upper triangle), the
!> BLAS2 core of dsytrd/dlatrd, at the trailing sizes dsytrd sees for n = 448 and 776:
!>   Accelerate dsymv, Accelerate dgemv on the full matrix, and a fused single pass over the upper triangle
!>   (each column j does y(1:j-1) += A(1:j-1,j) x(j) and y(j) += A(1:j,j) . x(1:j)), compiled -O3 -mcpu=native.
!> Median of 200 repetitions, microseconds and effective GB/s (bytes of the upper triangle, or full matrix for dgemv).
!>
!>   gfortran-16 -O3 -mcpu=native symv.f90 -framework Accelerate -o symv
program symv_bench
   use, intrinsic :: iso_fortran_env, only : dp => real64, int64
   implicit none
   integer, parameter :: sizes(6) = [448, 300, 150, 776, 500, 250]
   integer :: is, n, r, reps
   real(dp), allocatable :: A(:, :), x(:), y(:), y2(:), t(:)
   real(dp) :: t0, err
   reps = 200
   allocate(t(reps))
   write(*, '(a)') '    n   dsymv us  GB/s | dgemv us  GB/s | fused us  GB/s | max|y_fused - y_dsymv|'
   do is = 1, size(sizes)
      n = sizes(is)
      allocate(A(n, n), x(n), y(n), y2(n))
      call random_number(A); A = A + transpose(A); call random_number(x)
      call time3()
      deallocate(A, x, y, y2)
   end do

contains

   real(dp) function now()
      integer(int64) :: c, rr
      call system_clock(c, rr)
      now = real(c, dp) / real(rr, dp)
   end function now

   subroutine time3()
      real(dp) :: m(3), bytes_u, bytes_f
      integer :: k
      do k = 1, 3
         do r = 0, reps
            t0 = now()
            select case (k)
            case (1)
               call dsymv('U', n, 1.0_dp, A, n, x, 1, 0.0_dp, y, 1)
            case (2)
               call dgemv('N', n, n, 1.0_dp, A, n, x, 1, 0.0_dp, y2, 1)
            case (3)
               call fused(n, A, x, y2)
            end select
            if (r > 0) t(r) = 1e6_dp * (now() - t0)
         end do
         call sort(t); m(k) = t(reps/2)
      end do
      err = maxval(abs(y2 - y))
      bytes_u = 8.0_dp * n * (n + 1) / 2; bytes_f = 8.0_dp * n * n
      write(*, '(i5,3(f10.2,f6.1," |"),es10.2)') n, m(1), bytes_u/m(1)/1e3_dp, m(2), bytes_f/m(2)/1e3_dp, &
         & m(3), bytes_u/m(3)/1e3_dp, err
   end subroutine time3

   subroutine fused(n, A, x, y)
      integer, intent(in) :: n
      real(dp), intent(in) :: A(n, n), x(n)
      real(dp), intent(out) :: y(n)
      integer :: i, j
      real(dp) :: xj, s
      y = 0.0_dp
      do j = 1, n
         xj = x(j); s = 0.0_dp
         do i = 1, j - 1
            y(i) = y(i) + A(i, j) * xj
            s = s + A(i, j) * x(i)
         end do
         y(j) = y(j) + s + A(j, j) * xj
      end do
   end subroutine fused

   subroutine sort(v)
      real(dp), intent(inout) :: v(:)
      integer :: i, j
      real(dp) :: xx
      do i = 2, size(v)
         xx = v(i); j = i - 1
         do while (j >= 1)
            if (v(j) <= xx) exit
            v(j+1) = v(j); j = j - 1
         end do
         v(j+1) = xx
      end do
   end subroutine sort

end program symv_bench
