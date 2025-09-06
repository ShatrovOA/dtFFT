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
program test_r2c_3d_float
use iso_fortran_env
use iso_c_binding, only: c_loc, c_ptr
use dtfft
use test_utils
#include "_dtfft_mpi.h"
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#if defined(__NVCOMPILER)
use cudafor, only: cuda_stream_kind
#endif
#endif
#include "_dtfft_cuda.h"
#include "dtfft.f03"
implicit none
#ifndef DTFFT_TRANSPOSE_ONLY
  type(c_ptr) :: check
  real(real32),     allocatable, target :: in(:,:,:)
  complex(real32),  allocatable, target :: out(:)
#if defined(DTFFT_WITH_CUDA)
  integer(int32), parameter :: nx = 513, ny = 711, nz = 33
#else
  integer(int32), parameter :: nx = 16, ny = 8, nz = 4
#endif
  integer(int32) :: comm_size, comm_rank, ierr
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2c_t) :: plan
  integer(int32) :: in_counts(3), out_counts(3)
  integer(int64) :: alloc_size, in_size, out_size
  real(real64) :: tf, tb
  type(dtfft_config_t) :: conf
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: cuda_stream
#endif
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_stream_t) :: dtfft_stream
  type(dtfft_platform_t) :: platform
#endif

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2c_3d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ', nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif
#ifdef DTFFT_TRANSPOSE_ONLY
  if ( comm_rank == 0 ) &
    write(output_unit, '(a)') "R2C Transpose plan not supported, skipping it"

  call MPI_Finalize(ierr)
  stop
#endif

  call attach_gpu_to_process()
  call dtfft_create_config(conf)

  executor = DTFFT_EXECUTOR_NONE

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined (DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#endif

#if defined(DTFFT_WITH_CUDA)
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
#if !defined(__NVCOMPILER)
      if ( comm_rank == 0 ) &
        write(output_unit, '(a)') "This test requires NVFortran in order to run on GPU"
      call MPI_Finalize(ierr)
      stop
#endif
#ifdef DTFFT_WITH_NCCL
      conf%backend = DTFFT_BACKEND_NCCL
#else
      conf%backend = DTFFT_BACKEND_MPI_P2P
#endif
# if defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT
      conf%enable_z_slab = .false.
# elif defined( DTFFT_WITH_CUFFT )
      executor = DTFFT_EXECUTOR_CUFFT
# endif
    endif
  endblock
#endif

  if ( executor == DTFFT_EXECUTOR_NONE ) then
    if ( comm_rank == 0 ) &
      write(output_unit, '(a)') "Could not find valid R2C FFT executor, skipping test"
    call MPI_Finalize(ierr)
    stop
  endif

  call dtfft_set_config(conf)

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report()

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    call plan%get_stream(dtfft_stream, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(__NVCOMPILER)
    ! cuda_stream can be used in CUDA API provided with nvhpc-sdk
    cuda_stream = dtfft_get_cuda_stream(dtfft_stream)
#endif
  endif
#endif

  allocate(in(in_counts(1),in_counts(2), in_counts(3)))
  allocate(out(alloc_size))
!$acc enter data create(in, out) if ( platform == DTFFT_PLATFORM_CUDA )

  in_size = product(in_counts)
  out_size = product(out_counts)

  check = mem_alloc_host(in_size * FLOAT_STORAGE_SIZE)
  call setTestValuesFloat(check, in_size)

#if defined(DTFFT_WITH_CUDA)
!$acc host_data use_device(in) if ( platform == DTFFT_PLATFORM_CUDA )
  call floatH2D(check, c_loc(in), in_size, platform%val)
!$acc end host_data
#else
  call floatH2D(check, c_loc(in), in_size)
#endif


  tf = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(in, out) if ( platform == DTFFT_PLATFORM_CUDA )
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(dtfft_stream) )
  endif
#endif
  tf = tf + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
!$acc host_data use_device(out) if ( platform == DTFFT_PLATFORM_CUDA )
  call scaleComplexFloat(executor%val, c_loc(out), out_size, int(nx * ny * nz, int64), platform%val, dtfft_stream)
!$acc end host_data
#else
  call scaleComplexFloat(executor%val, c_loc(out), out_size, int(nx * ny * nz, int64))
#endif

  tb = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(in, out) if ( platform == DTFFT_PLATFORM_CUDA )
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(dtfft_stream) )
  endif
#endif
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
!$acc host_data use_device(in) if ( platform == DTFFT_PLATFORM_CUDA )
  call checkAndReportFloat(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check, platform%val)
!$acc end host_data
#else
  call checkAndReportFloat(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check)
#endif

!$acc exit data delete(in, out) if ( platform == DTFFT_PLATFORM_CUDA )

  deallocate(in)
  deallocate(out)
  call mem_free_host(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
#endif
end program test_r2c_3d_float