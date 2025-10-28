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
program test_c2c_3d_float
use iso_fortran_env
use iso_c_binding, only: c_null_ptr, c_loc, c_ptr
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#endif
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  type(c_ptr) :: check
  complex(real32), pointer :: in(:,:,:), out(:)
  integer(int32), parameter :: nx = 13, ny = 45, nz = 29
  integer(int32) :: comm_size, comm_rank, i
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(3), out_counts(3), in_starts(3)
  integer(int64) :: alloc_size, in_size, out_size, element_size
  real(real64) :: tf, tb
  type(dtfft_executor_t) :: executor
  integer(int32) :: ierr
  type(dtfft_config_t) :: conf
  type(dtfft_backend_t) :: backend
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
  logical :: is_cuda_platform
#endif

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_3d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

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

    is_cuda_platform = .false.
    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
      is_cuda_platform = .true.
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

  backend = DTFFT_BACKEND_MPI_P2P
  conf = dtfft_config_t(backend=backend)

#if defined(DTFFT_WITH_CUDA)
    ! Can be redefined by environment variable
    conf%platform = DTFFT_PLATFORM_CUDA

    if ( is_cuda_platform ) then
#if defined(DTFFT_WITH_NVSHMEM)
      backend = DTFFT_BACKEND_CUFFTMP_PIPELINED
#elif defined(DTFFT_WITH_NCCL)
      backend = DTFFT_BACKEND_NCCL_PIPELINED
#endif
    endif
#endif
  conf%backend = backend

  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_starts=in_starts, in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%report()

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(ierr); DTFFT_CHECK(ierr)

  if ( platform == DTFFT_PLATFORM_CUDA ) then
    block
      type(dtfft_backend_t) :: real_backend

      real_backend = plan%get_backend(error_code=ierr); DTFFT_CHECK(ierr)
      if ( backend /= real_backend .and. comm_size > 1 ) error stop "backend /= real_backend"
    endblock
  endif
#endif

  element_size = plan%get_element_size(ierr); DTFFT_CHECK(ierr)
  if ( element_size /= 8_int64 ) error stop "element_size /= 8_int64"

  in_size = product(in_counts)
  out_size = product(out_counts)

  call plan%mem_alloc(alloc_size, in, in_counts, lbounds=in_starts, error_code=ierr)
  do i = 1, 3
    if ( lbound(in, dim=i) /= in_starts(i) ) error stop "invalid lbound, dim = "//to_str(i)//""
    if ( ubound(in, dim=i) /= in_starts(i) + in_counts(i) - 1 ) error stop "invalid ubound, dim = "//to_str(i)
  enddo
  call plan%mem_alloc(alloc_size, out, error_code=ierr)

  check = mem_alloc_host(in_size * element_size)
  call setTestValuesComplexFloat(check, in_size)

#if defined(DTFFT_WITH_CUDA)
  call complexFloatH2D(check, c_loc(in), in_size, platform%val)
#else
  call complexFloatH2D(check, c_loc(in), in_size)
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
  call scaleComplexFloat(executor%val, c_loc(out), int(product(out_counts), int64), int(nx * ny * nz, int64), platform%val, NULL_STREAM)
#else
  call scaleComplexFloat(executor%val, c_loc(out), int(product(out_counts), int64), int(nx * ny * nz, int64))
#endif

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call checkAndReportComplexFloat(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check, platform%val)
#else
  call checkAndReportComplexFloat(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check)
#endif

  call mem_free_host(check)

  call plan%mem_free(in, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(out, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d_float