!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
program test_r2r_2d_float
use iso_fortran_env
use iso_c_binding
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#endif
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  type(c_ptr) :: in, out, check
  integer(int32), parameter :: nx = 17, ny = 4
  integer(int32) :: comm_size, comm_rank, ierr
  integer(int64) :: alloc_size, in_size, out_size, element_size, scaler
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2r_t) :: plan
  integer(int32) :: in_counts(2), out_counts(2)
  real(real64) :: tf, tb
  TYPE_MPI_COMM :: comm_1d
  type(dtfft_config_t) :: config
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
#endif

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_2d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

  call attach_gpu_to_process()

  executor = DTFFT_EXECUTOR_NONE

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#endif

#if defined(DTFFT_WITH_CUDA)
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
# if defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT
# endif
    endif
  endblock
#endif

  call MPI_Cart_create(MPI_COMM_WORLD, 1, [comm_size], [.false.], .true., comm_1d, ierr)

  config = dtfft_config_t()
  config%backend = DTFFT_BACKEND_MPI_P2P_PIPELINED
  call dtfft_set_config(config, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny], [DTFFT_DST_2, DTFFT_DST_3], comm=comm_1d, precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  element_size = plan%get_element_size(error_code=ierr); DTFFT_CHECK(ierr)
  in = plan%mem_alloc_ptr(element_size * alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  out = plan%mem_alloc_ptr(element_size * alloc_size, error_code=ierr); DTFFT_CHECK(ierr)

  in_size = product(in_counts)
  out_size = product(out_counts)
  check = mem_alloc_host(in_size * element_size)
  call setTestValuesFloat(check, in_size)

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
  call floatH2D(check, in, in_size, platform%val)
#else
  call floatH2D(check, in, in_size)
#endif


  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute_ptr(in, out, DTFFT_EXECUTE_FORWARD, c_null_ptr, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tf = tf + MPI_Wtime()

  scaler = 4 * nx * ny
#if defined(DTFFT_WITH_CUDA)
  call scaleFloat(executor%val, out, out_size, scaler, platform%val, NULL_STREAM)
#else
  call scaleFloat(executor%val, out, out_size, scaler)
#endif


  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute_ptr(out, in, DTFFT_EXECUTE_BACKWARD, c_null_ptr, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call checkAndReportFloat(int(nx * ny, int64), tf, tb, in, in_size, check, platform%val)
#else
  call checkAndReportFloat(int(nx * ny, int64), tf, tb, in, in_size, check)
#endif

  call plan%mem_free_ptr(in)
  call plan%mem_free_ptr(out)
  call mem_free_host(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_2d_float