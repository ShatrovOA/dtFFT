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
program test_c2c_3d
use iso_fortran_env
use dtfft
use iso_c_binding
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
use dtfft_utils
#endif
#include "dtfft_cuda.h"
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  complex(real64),  pointer :: inout(:), aux(:)
  complex(real64),  allocatable, target :: check(:,:,:)
  real(real64) :: local_error, rnd1, rnd2
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  integer(int32), parameter :: nx = 255, ny = 333, nz = 135
#else
  integer(int32), parameter :: nx = 129, ny = 123, nz = 33
#endif
  integer(int32) :: comm_size, comm_rank, i, j, k, ierr, ii, jj, kk, idx
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(3), out_counts(3), iter
  integer(int64)  :: alloc_size, element_size, in_size, out_size
  real(real64) :: ts, tf, tb
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_stream_t)  :: stream
  type(dtfft_backend_t) :: selected_backend
  type(dtfft_platform_t) :: platform
#endif

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

#if defined(DTFFT_WITH_CUDA)
  block
    type(dtfft_config_t) :: conf

    conf = dtfft_config_t(backend=DTFFT_BACKEND_NCCL_PIPELINED, platform=DTFFT_PLATFORM_CUDA)
    call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  endblock
#endif

  ! Setting effort=DTFFT_PATIENT will ignore value of `conf%backend` and will run autotune to find best backend
  ! Fastest backend will be selected
  call plan%create([nx, ny, nz], executor=executor, effort=DTFFT_ESTIMATE, error_code=ierr); DTFFT_CHECK(ierr)
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
  selected_backend = plan%get_backend(error_code=ierr); DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Selected backend: '"//dtfft_get_backend_string(selected_backend)//"'"
  endif
#endif

  call plan%mem_alloc(alloc_size, inout, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_size, aux, error_code=ierr); DTFFT_CHECK(ierr)

  allocate(check(in_counts(1), in_counts(2), in_counts(3)))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        check(i,j,k) = cmplx(rnd1, rnd1, real64)
      enddo
    enddo
  enddo

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaMemcpyAsync", cudaMemcpyAsync(c_loc(inout), c_loc(check), in_size * element_size, cudaMemcpyHostToDevice, stream) )
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  else
    do k = 1, in_counts(3)
      do j = 1, in_counts(2)
        do i = 1, in_counts(1)
          ii = i - 1; jj = j - 1; kk = k - 1
          idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
          inout(idx) = check(i,j,k)
        enddo
      enddo
    enddo
  endif
#else
  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        ii = i - 1; jj = j - 1; kk = k - 1
        idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
        inout(idx) = check(i,j,k)
      enddo
    enddo
  enddo
#endif


  tf = 0.0_real64
  tb = 0.0_real64
  do iter = 1, 3
    ts = - MPI_Wtime()

    call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, aux, error_code=ierr);  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
    endif
#endif
    tf = tf + ts + MPI_Wtime()

    if ( executor /= DTFFT_EXECUTOR_NONE ) then
      block
        integer(int64) :: scale

        scale = nx * ny * nz
#if defined(DTFFT_WITH_CUDA)
        if ( platform == DTFFT_PLATFORM_CUDA )  then
          call scaleComplexDouble(c_loc(inout), out_size, scale, stream)
          CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
        else
          call scaleComplexDoubleHost(c_loc(inout), out_size, scale)
        endif
#else
        call scaleComplexDoubleHost(c_loc(inout), out_size, scale)
#endif
      end block
    endif

    ts = 0.0_real64 - MPI_Wtime()
    call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD, aux, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
    endif
#endif
    tb = tb + ts + MPI_Wtime()
  enddo

  local_error = 0._real64

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
  block
    complex(real64), allocatable, target :: test(:)

    allocate(test(in_size))

    CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(test), c_loc(inout), element_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkComplexDouble(c_loc(check), c_loc(test), in_size)
    deallocate(test)
  endblock
  else
    local_error = checkComplexDouble(c_loc(check), c_loc(inout), in_size)
  endif
#else
  local_error = checkComplexDouble(c_loc(check), c_loc(inout), in_size)
#endif

  call report(tf, tb, local_error, nx, ny, nz)

  call plan%mem_free(inout, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(aux, error_code=ierr); DTFFT_CHECK(ierr)

  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d