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
use iso_fortran_env
use dtfft
use test_utils
use iso_c_binding
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
use dtfft_utils
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none
#ifndef DTFFT_TRANSPOSE_ONLY
  real(real64),     allocatable :: inout(:), check(:)
  real(real64) :: local_error, rnd
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  integer(int32), parameter :: nx = 999, ny = 344
#else
  integer(int32), parameter :: nx = 64, ny = 32
#endif
  integer(int32) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2c_t) :: plan
  integer(int32) :: in_counts(2), out_counts(2)
  real(real64) :: tf, tb
  integer(int64) :: alloc_size, upper_bound, cmplx_upper_bound
  type(dtfft_config_t) :: conf
#if defined(DTFFT_WITH_CUDA) 
  type(dtfft_platform_t) :: platform
#endif

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
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined (DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#else
  executor = DTFFT_EXECUTOR_NONE
#endif

#ifdef DTFFT_WITH_CUDA
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
# if defined( DTFFT_WITH_CUFFT )
      executor = DTFFT_EXECUTOR_CUFFT
# elif defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT
# else
      executor = DTFFT_EXECUTOR_NONE
# endif
    endif
  endblock
#endif

  call attach_gpu_to_process()

  conf = dtfft_config_t()
#if defined(DTFFT_WITH_CUDA)
  !! Using openacc, disable nvshmem
  conf%enable_nvshmem_backends = .false.
  conf%platform = DTFFT_PLATFORM_CUDA
#endif
  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny], effort=DTFFT_PATIENT, executor=executor, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(ierr); DTFFT_CHECK(ierr)
  block
    type(dtfft_backend_t) :: selected_backend

    selected_backend = plan%get_backend(error_code=ierr); DTFFT_CHECK(ierr)
    if(comm_rank == 0) then
      write(output_unit, '(a)') "Using backend: "//dtfft_get_backend_string(selected_backend)
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

!$acc enter data copyin(inout) if( platform == DTFFT_PLATFORM_CUDA )

  tf = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(inout) if( platform == DTFFT_PLATFORM_CUDA )
  call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
!$acc wait
  endif
#endif
  tf = tf + MPI_Wtime()

!$acc kernels present(inout) if( platform == DTFFT_PLATFORM_CUDA )
  inout(:cmplx_upper_bound) = inout(:cmplx_upper_bound) / real(nx * ny, real64)
!$acc end kernels

  tb = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(inout) if( platform == DTFFT_PLATFORM_CUDA )
  call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
!$acc wait
  endif
#endif
  tb = tb + MPI_Wtime()

!$acc update self(inout) if( platform == DTFFT_PLATFORM_CUDA )
!$acc exit data delete(inout)  if( platform == DTFFT_PLATFORM_CUDA )

  local_error = maxval(abs(inout(:upper_bound) - check(:upper_bound)))

  call report(tf, tb, local_error, nx, ny)

  deallocate(inout)
  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
#endif
end program test_r2c_2d