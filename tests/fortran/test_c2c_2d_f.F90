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
program test_c2c_2d
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
  type(c_ptr) :: check
  complex(real64),  pointer :: in(:,:), out(:,:)
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  integer(int32), parameter :: nx = 4431, ny = 31
#else
  integer(int32), parameter :: nx = 12, ny = 12
#endif
  integer(int32) :: comm_size, comm_rank, ierr
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(2), out_counts(2)
  real(real64) :: tf, tb
  type(dtfft_executor_t) :: executor = DTFFT_EXECUTOR_NONE
  integer(int64) :: alloc_size, element_size, in_size
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
  logical :: is_cuda_platform
#endif


  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a,i0)') "dtFFT Version = ",dtfft_get_version()
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       dtFFT test: c2c_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#ifdef DTFFT_WITH_FFTW
  executor = DTFFT_EXECUTOR_FFTW3;
#elif defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL;
#endif
#ifdef DTFFT_WITH_CUDA
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    is_cuda_platform = .false.
    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
      is_cuda_platform = .true.
# if defined( DTFFT_WITH_CUFFT )
      executor = DTFFT_EXECUTOR_CUFFT;
# elif defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT;
# else
      executor = DTFFT_EXECUTOR_NONE;
# endif
    endif
  endblock
#endif

  call attach_gpu_to_process()

#if defined(DTFFT_WITH_CUDA)
  block
    type(dtfft_config_t) :: conf
    conf = dtfft_config_t(platform=DTFFT_PLATFORM_CUDA)

    conf%backend = DTFFT_BACKEND_MPI_A2A
    if ( is_cuda_platform ) then
#ifdef DTFFT_WITH_NCCL
      conf%backend = DTFFT_BACKEND_NCCL
#endif
    end if
    call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  endblock
#endif


  call plan%create([nx, ny], effort=DTFFT_MEASURE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)

  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  element_size = plan%get_element_size(error_code=ierr); DTFFT_CHECK(ierr)
  if ( element_size /= 16_int64 ) error stop

  in_size = product(in_counts)

  call plan%mem_alloc(alloc_size, in, in_counts, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_size, out, out_counts, error_code=ierr); DTFFT_CHECK(ierr)

  check = mem_alloc_host(in_size * DOUBLE_COMPLEX_STORAGE_SIZE)
  call setTestValuesComplexDouble(check, in_size)

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
  call complexDoubleH2D(check, c_loc(in), in_size, platform%val)
#else
  call complexDoubleH2D(check, c_loc(in), in_size)
#endif


  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaDeviceSynchronize() )
  endif
#endif
  tf = tf + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call scaleComplexDouble(executor%val, c_loc(out), int(product(out_counts), int64), int(nx * ny, int64), platform%val, NULL_STREAM)
#else
  call scaleComplexDouble(executor%val, c_loc(out), int(product(out_counts), int64), int(nx * ny, int64))
#endif

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr);  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call checkAndReportComplexDouble(int(nx * ny, int64), tf, tb, c_loc(in), in_size, check, platform%val)
#else
  call checkAndReportComplexDouble(int(nx * ny, int64), tf, tb, c_loc(in), in_size, check)
#endif

  call plan%mem_free(in, ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(out, ierr); DTFFT_CHECK(ierr)
  call mem_free_host(check)

  call plan%destroy(error_code=ierr); DTFFT_CHECK(ierr)
  call MPI_Finalize(ierr)
end program test_c2c_2d