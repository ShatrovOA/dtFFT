!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
#include "dtfft_config.h"
program test_r2c_2d
use iso_fortran_env, only: R8P => real64, I4P => int32, I8P => int64, I1P => int8, output_unit, error_unit
use dtfft
use test_utils
use iso_c_binding
#ifdef DTFFT_WITH_CUDA
use cudafor
use dtfft_utils
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  ! real(R8P),     allocatable :: in(:,:), check(:,:)
  real(R8P),     allocatable :: inout(:), check(:)
  real(R8P) :: local_error, rnd
#ifdef DTFFT_WITH_CUDA
  integer(I4P), parameter :: nx = 999, ny = 344
#else
  integer(I4P), parameter :: nx = 64, ny = 32
#endif
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  integer(I1P) :: executor_type
  type(dtfft_plan_r2c) :: plan
  integer(I4P) :: in_counts(2), out_counts(2)
  real(R8P) :: tf, tb
  integer(I8P) :: alloc_size, upper_bound, cmplx_upper_bound

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2c_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif
#ifdef DTFFT_TRANSPOSE_ONLY
  if ( comm_rank == 0 ) &
    write(output_unit, '(a)') "R2C Transpose plan not supported, skipping it"

  call MPI_Finalize(ierr)
  stop
#endif

#if defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL
#elif defined(DTFFT_WITH_VKFFT)
  executor_type = DTFFT_EXECUTOR_VKFFT
#endif

  call attach_gpu_to_process()

  call plan%create([nx, ny], effort_flag=DTFFT_PATIENT, executor_type=executor_type, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

#ifdef DTFFT_WITH_CUDA
  block
    integer(I1P) :: selected_backend

    selected_backend = plan%get_gpu_backend(error_code=ierr)
    DTFFT_CHECK(ierr)
    if(comm_rank == 0) then
      write(output_unit, '(a)') "Using backend: "//dtfft_get_gpu_backend_string(selected_backend)
    endif
  endblock
#endif

  upper_bound = product(in_counts)
  cmplx_upper_bound = 2 * product(out_counts)
  allocate(inout(alloc_size))

  allocate(check(upper_bound))

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd)
      inout((j - 1) * in_counts(1) + i) = rnd
      check((j - 1) * in_counts(1) + i) = rnd
    enddo
  enddo

!$acc enter data copyin(inout)

  tf = 0.0_R8P - MPI_Wtime()
!$acc host_data use_device(inout)
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_OUT, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
#endif
  tf = tf + MPI_Wtime()

!$acc kernels present(inout)
  inout(:cmplx_upper_bound) = inout(:cmplx_upper_bound) / real(nx * ny, R8P)
!$acc end kernels

  tb = 0.0_R8P - MPI_Wtime()
!$acc host_data use_device(inout)
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_IN, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#ifdef DTFFT_WITH_CUDA
  CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
#endif
  tb = tb + MPI_Wtime()

!$acc update self(inout)
!$acc exit data delete(inout)

  local_error = maxval(abs(inout(:upper_bound) - check(:upper_bound)))

  call report("r2c_2d", tf, tb, local_error)

  ! call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  ! tf = t_sum / real(comm_size, R8P)
  ! call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  ! tb = t_sum / real(comm_size, R8P)

  ! if(comm_rank == 0) then
  !   write(output_unit, '(a, f16.10)') "Forward execution time: ", tf
  !   write(output_unit, '(a, f16.10)') "Backward execution time: ", tb
  !   write(output_unit, '(a)') "----------------------------------------"
  ! endif

  
  ! call MPI_Allreduce(local_error, global_error, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD, ierr)
  ! if(comm_rank == 0) then
  !   if(global_error < 1.e-6) then
  !     write(output_unit, '(a)') "Test 'r2c_2d' PASSED!"
  !   else
  !     write(error_unit, '(a, f16.10)') "Test 'r2c_2d' FAILED... error = ", global_error
  !     error stop
  !   endif
  !   write(output_unit, '(a)') "----------------------------------------"
  ! endif

  deallocate(inout)
  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2c_2d