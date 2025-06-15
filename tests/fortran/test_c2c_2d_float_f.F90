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
program test_c2c_2d_float
use iso_fortran_env
use iso_c_binding
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
use dtfft_utils
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  complex(real32), allocatable, target :: check(:,:)
  complex(real32), DEVICE_PTR pointer :: pin(:,:), pout(:,:)
  real(real32) :: local_error, rnd1, rnd2
  integer(int32), parameter :: nx = 64, ny = 32
  integer(int32) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_executor_t) :: executor
  integer(int64) :: alloc_size, element_size, alloc_bytes
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(2), out_counts(2)
  real(real64) :: tf, tb
  type(c_ptr) :: in, out
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
#endif

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_2d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
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
      executor = DTFFT_EXECUTOR_CUFFT;
# elif defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT;
# else
      executor = DTFFT_EXECUTOR_NONE;
# endif
    else
# if defined(__NVCOMPILER)
      if ( comm_rank == 0 ) then
        write(error_unit, "(a)") "This test can only run on a CUDA Device, due to `DEVICE_PTR` specification of pointers"
      endif
      call MPI_Finalize(ierr)
      stop
#endif
    endif
  endblock
#endif

    call attach_gpu_to_process()

#if defined(DTFFT_WITH_CUDA)
  block
    type(dtfft_config_t) :: conf

    conf = dtfft_config_t(backend=DTFFT_BACKEND_MPI_P2P, platform=DTFFT_PLATFORM_CUDA)
    call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  endblock
#endif

  call plan%create([nx, ny], precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  element_size = plan%get_element_size(ierr);  DTFFT_CHECK(ierr)

  alloc_bytes = alloc_size * element_size

  call plan%mem_alloc(alloc_bytes, in, ierr); DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_bytes, out, ierr); DTFFT_CHECK(ierr)

  allocate(check(in_counts(1),in_counts(2)))

  call c_f_pointer(in, pin, in_counts)
  call c_f_pointer(out, pout, out_counts)

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd1)
      call random_number(rnd2)
      check(i,j) = cmplx(rnd1, rnd2)
    enddo
  enddo

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
#endif

! Host to device copy
  pin(:,:) = check(:,:)

  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute(pin, pout, DTFFT_EXECUTE_FORWARD)
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
      call scaleComplexFloat(out, out_size, scale, dtfft_stream_t(c_null_ptr))
      CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
    else
      call scaleComplexFloatHost(out, out_size, scale)
    endif
#else
    call scaleComplexFloatHost(out, out_size, scale);
#endif
    end block
    endif

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(pout, pin, DTFFT_EXECUTE_BACKWARD)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

  block
    integer(int64) :: in_size

    in_size = product(in_counts)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
  block
    type(c_ptr) :: test

    call mem_alloc_host(element_size * in_size, test)

    CUDA_CALL( "cudaMemcpy", cudaMemcpy(test, in, element_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkComplexFloat(c_loc(check), test, in_size);

    call mem_free_host(test)
  endblock
  else
    local_error = checkComplexFloat(c_loc(check), in, in_size);
  endif
#else
  local_error = checkComplexFloat(c_loc(check), in, in_size);
#endif
  endblock
  call report(tf, tb, local_error, nx, ny)

  call plan%mem_free(in)
  call plan%mem_free(out)

  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_2d_float