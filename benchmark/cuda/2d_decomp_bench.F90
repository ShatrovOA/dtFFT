module decomp_bench
use iso_c_binding
use decomp_2d
use decomp_2d_fft
use decomp_2d_constants
use decomp_2d_mpi
use mpi
implicit none

contains

   subroutine run_2d_decomp_bench(nx, ny, nz, p_row, p_col, warmup_iters, measure_iters, time) bind(C)
      integer(c_int), value   :: nx, ny, nz
      integer(c_int), value   :: warmup_iters, measure_iters
      integer(c_int), value   :: p_row, p_col
      real(c_double)          :: time

      integer :: comm_rank
      integer :: nranks_tot

      integer :: ntest = 10  ! repeat test this times

      type(decomp_info), pointer :: ph => null()
      complex(mytype), allocatable, dimension(:, :, :) :: in, out

      real(mytype) :: dr, di, error
      integer :: ierror, i, j, k, m
      integer :: xst1, xst2, xst3
      integer :: xen1, xen2, xen3
      double precision :: timer_start, timer_end, min_s, max_s, avg_s

      ! To resize the domain we need to know global number of ranks
      ! This operation is also done as part of decomp_2d_init
      call MPI_COMM_SIZE(MPI_COMM_WORLD, nranks_tot, ierror)
      call mpi_comm_rank(MPI_COMM_WORLD, comm_rank, ierror)
      call decomp_2d_init(nx, ny, nz, p_row, p_col, complex_pool=.true.)

      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ! Test the c2c interface
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      call decomp_2d_fft_init(PHYSICAL_IN_X) ! force the default x pencil
      ! ph => decomp_2d_fft_get_ph()
      ! !  input is X-pencil data
      ! ! output is Z-pencil data
      ! call alloc_x(in, ph, .true.)
      ! call alloc_z(out, ph, .true.)

   call alloc_x(in, opt_global=.true.)
   ! call alloc_y(u2, opt_global=.true.)
   call alloc_z(out, opt_global=.true.)


      in(:,:,:) = (0._mytype, 0._mytype)

!$acc data copyin(in, out)
      ! First iterations out of the counting loop
      do m = 1, warmup_iters
         call decomp_2d_fft_3d(in, out, DECOMP_2D_FFT_FORWARD)
         call decomp_2d_fft_3d(out, in, DECOMP_2D_FFT_BACKWARD)
      enddo
      call MPI_Barrier(MPI_COMM_WORLD, ierror)

      ! Init the time

      timer_start = MPI_WTIME()
      do m = 1, measure_iters
         call decomp_2d_fft_3d(in, out, DECOMP_2D_FFT_FORWARD)
         call decomp_2d_fft_3d(out, in, DECOMP_2D_FFT_BACKWARD)
      end do
      timer_end = MPI_WTIME()
      time = (timer_end - timer_start)
!$acc end data

      call MPI_Allreduce(time, min_s, 1, MPI_DOUBLE_PRECISION, MPI_MIN, MPI_COMM_WORLD, ierror)
      call MPI_Allreduce(time, max_s, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierror)
      call MPI_Allreduce(time, avg_s, 1, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierror)
      ! time = avg_s / dble(nranks_tot)
      time = max_s * 1000._c_double

      ! if ( comm_rank == 0 ) then
      !    print*,'min time (s) = ', min_s
      !    print*,'max time (s) = ', max_s
      !    print*,'avg time (s) = ', time
      !    print*,'time per FFT (s) = ', time / real(measure_iters, kind=c_double)
      ! endif

      deallocate (in, out)
      nullify (ph)
      call decomp_2d_fft_finalize
      call decomp_2d_finalize
   end subroutine run_2d_decomp_bench

end module decomp_bench