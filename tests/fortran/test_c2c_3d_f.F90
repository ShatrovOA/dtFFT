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
program test_c2c_3d
use iso_fortran_env
use iso_c_binding
use dtfft
use iso_c_binding
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#endif
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  complex(real64),  pointer :: inout(:,:,:), aux(:)
  type(c_ptr) :: check
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  integer(int32), parameter :: nx = 255, ny = 333, nz = 135
#else
  integer(int32), parameter :: nx = 129, ny = 123, nz = 33
#endif
  integer(int32) :: comm_size, comm_rank, ierr
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(3), out_counts(3), iter
  integer(int64)  :: alloc_size, element_size, in_size, out_size
  real(real64) :: ts, tf, tb
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_stream_t)  :: stream
  type(dtfft_platform_t) :: platform
#endif
  type(dtfft_backend_t) :: selected_backend
  type(dtfft_config_t) :: conf

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_3d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#else
  executor = DTFFT_EXECUTOR_NONE
#endif

#ifdef DTFFT_WITH_CUDA
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
# if defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT
# elif defined( DTFFT_WITH_CUFFT )
      executor = DTFFT_EXECUTOR_CUFFT
# else
      executor = DTFFT_EXECUTOR_NONE
# endif
    endif
  endblock
#endif

  call attach_gpu_to_process()

  conf = dtfft_config_t()

#if defined(DTFFT_WITH_CUDA)
  conf%platform = DTFFT_PLATFORM_CUDA
#endif

  conf%enable_mpi_backends = .true.
  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  ! Setting effort=DTFFT_PATIENT will ignore value of `conf%backend` and will run autotune to find best backend
  ! Fastest backend will be selected
  call plan%create([nx, ny, nz], executor=executor, effort=DTFFT_PATIENT, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, error_code=ierr); DTFFT_CHECK(ierr)
  alloc_size = plan%get_alloc_size(ierr); DTFFT_CHECK(ierr)
  element_size = plan%get_element_size(ierr); DTFFT_CHECK(ierr)
  in_size = product(in_counts)
  out_size = product(out_counts)

  call plan%report()

  if ( element_size /= 16_int64 ) error stop "element_size /= 16_int64"

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(ierr); DTFFT_CHECK(ierr)

  if ( platform == DTFFT_PLATFORM_CUDA ) then
    call plan%get_stream(stream, error_code=ierr); DTFFT_CHECK(ierr)
  endif
#endif

  selected_backend = plan%get_backend(error_code=ierr); DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Selected backend: '"//dtfft_get_backend_string(selected_backend)//"'"
  endif

  call plan%mem_alloc(alloc_size, inout, in_counts, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_size, aux, error_code=ierr); DTFFT_CHECK(ierr)

  check = mem_alloc_host(in_size * element_size)
  call setTestValuesComplexDouble(check, in_size)


#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
  call complexDoubleH2D(check, c_loc(inout), in_size, platform%val)
#else
  call complexDoubleH2D(check, c_loc(inout), in_size)
#endif


  tf = 0.0_real64
  tb = 0.0_real64
  do iter = 1, 3
    ts = - MPI_Wtime()

    call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, aux, error_code=ierr);  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaStreamSynchronize(stream) )
    endif
#endif
    tf = tf + ts + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call scaleComplexDouble(executor%val, c_loc(inout), int(product(out_counts), int64), int(nx * ny * nz, int64), platform%val, stream)
#else
  call scaleComplexDouble(executor%val, c_loc(inout), int(product(out_counts), int64), int(nx * ny * nz, int64))
#endif

    ts = 0.0_real64 - MPI_Wtime()
    call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD, aux, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( cudaStreamSynchronize(stream) )
    endif
#endif
    tb = tb + ts + MPI_Wtime()
  enddo

#if defined(DTFFT_WITH_CUDA)
  call checkAndReportComplexDouble(int(nx * ny * nz, int64), tf, tb, c_loc(inout), in_size, check, platform%val)
#else
  call checkAndReportComplexDouble(int(nx * ny * nz, int64), tf, tb, c_loc(inout), in_size, check)
#endif

  call plan%mem_free(inout, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(aux, error_code=ierr); DTFFT_CHECK(ierr)

  call mem_free_host(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d