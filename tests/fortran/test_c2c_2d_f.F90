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
program test_c2c_2d
use iso_fortran_env
use iso_c_binding
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA)
! # if defined(__NVCOMPILER)
! use cudafor
! # else
use dtfft_interface_cuda_runtime
! # endif
use dtfft_utils
#endif
#include "dtfft_cuda.h"
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none (type, external)
  complex(real64),  allocatable, target :: check(:,:)
  complex(real64),  pointer :: in(:,:), out(:,:)
  ! type(c_ptr) :: inptr, outptr
  real(real64) :: local_error, rnd1, rnd2
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  integer(int32), parameter :: nx = 1115, ny = 3333
#else
  integer(int32), parameter :: nx = 12, ny = 12
#endif
  integer(int32) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(2), out_counts(2)
  real(real64) :: tf, tb
  type(dtfft_executor_t) :: executor = DTFFT_EXECUTOR_NONE
  integer(int64) :: alloc_size, element_size, in_size
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
#endif


  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a,i0)') "dtFFT Version = ",dtfft_get_version()
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_2d             |"
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

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
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
    conf = dtfft_config_t(platform=DTFFT_PLATFORM_CUDA, backend=DTFFT_BACKEND_NCCL)
#ifndef DTFFT_WITH_NCCL
    conf%backend = DTFFT_BACKEND_MPI_A2A;
#endif
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

  ! call c_f_pointer(inptr, in, in_counts)
  ! call c_f_pointer(outptr, out, out_counts)

  allocate(check(in_counts(1), in_counts(2)))

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd1)
      call random_number(rnd2)
      check(i,j) = cmplx(rnd1, rnd2, kind=real64)
    enddo
  enddo

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)

  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(in), c_loc(check), in_size * element_size, cudaMemcpyHostToDevice) )
  else
    in(:,:) = check(:,:)
  endif
#else
  in(:,:) = check(:,:)
#endif


  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tf = tf + MPI_Wtime()

  if ( executor /= DTFFT_EXECUTOR_NONE ) then
  block
    integer(int64) :: out_size, scale

    out_size = product(out_counts)
    scale = nx * ny
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA )  then
      call scaleComplexDouble(c_loc(out), out_size, scale, dtfft_stream_t(c_null_ptr))
      CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
    else
      call scaleComplexDoubleHost(c_loc(out), out_size, scale)
    endif
#else
    call scaleComplexDoubleHost(c_loc(out), out_size, scale);
#endif
  end block
  endif

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr);  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
  block
    complex(real64), allocatable, target :: test(:)

    allocate(test(product(in_counts)))

    CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(test), c_loc(in), element_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkComplexDouble(c_loc(check), c_loc(test), in_size);
    deallocate(test)
  endblock
  else
    local_error = checkComplexDouble(c_loc(check), c_loc(in), in_size);
  endif
#else
  local_error = checkComplexDouble(c_loc(check), c_loc(in), in_size);
#endif

  call report(tf, tb, local_error, nx, ny)

  call plan%mem_free(in, ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(out, ierr); DTFFT_CHECK(ierr)

  deallocate(check)

  call plan%destroy(error_code=ierr); DTFFT_CHECK(ierr)
  call MPI_Finalize(ierr)
end program test_c2c_2d