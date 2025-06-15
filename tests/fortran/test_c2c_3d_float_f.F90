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
program test_c2c_3d_float
use iso_fortran_env
use iso_c_binding, only: c_null_ptr, c_loc
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
  complex(real32),  allocatable, target :: check(:,:,:)
  complex(real32), pointer :: in(:), out(:)
  real(real32) :: local_error, rnd1, rnd2
  integer(int32), parameter :: nx = 13, ny = 45, nz = 2
  integer(int32) :: comm_size, comm_rank, i, j, k, ii, jj, kk, idx
  type(dtfft_plan_c2c_t) :: plan
  integer(int32) :: in_counts(3), out_counts(3)
  integer(int64) :: alloc_size, in_size, out_size, element_size
  real(real64) :: tf, tb
  type(dtfft_executor_t) :: executor
  integer(int32) :: ierr
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_backend_t) :: backend
  type(dtfft_platform_t) :: platform
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

#if defined(DTFFT_WITH_CUDA)
  block
    type(dtfft_config_t) :: conf

    conf = dtfft_config_t(platform=DTFFT_PLATFORM_CUDA)
#if defined(DTFFT_WITH_NVSHMEM)
    backend = DTFFT_BACKEND_CUFFTMP
#else
    backend = DTFFT_BACKEND_MPI_P2P
#endif
    conf%backend = backend
    call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  endblock
#endif

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts = in_counts, out_counts = out_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

  call plan%report()

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(ierr); DTFFT_CHECK(ierr)

  if ( platform == DTFFT_PLATFORM_CUDA ) then
    block
      type(dtfft_backend_t) :: real_backend

      real_backend = plan%get_backend(error_code=ierr); DTFFT_CHECK(ierr)
      if ( backend /= real_backend ) error stop "backend /= real_backend"
    endblock
  endif
#endif

  element_size = plan%get_element_size(ierr); DTFFT_CHECK(ierr)
  if ( element_size /= 8_int64 ) error stop "element_size /= 8_int64"

  in_size = 1_int64 * product(in_counts)
  out_size = 1_int64 * product(out_counts)

  call plan%mem_alloc(alloc_size, in, error_code=ierr)
  call plan%mem_alloc(alloc_size, out, error_code=ierr)

  allocate(check(in_counts(1),in_counts(2),in_counts(3)))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        check(i,j,k) = cmplx(rnd1, rnd2, real32)
      enddo
    enddo
  enddo

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(in), c_loc(check), in_size * element_size, cudaMemcpyHostToDevice) )
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  else
    do k = 1, in_counts(3)
      do j = 1, in_counts(2)
        do i = 1, in_counts(1)
          ii = i - 1; jj = j - 1; kk = k - 1
          idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
          in(idx) = check(i,j,k)
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
        in(idx) = check(i,j,k)
      enddo
    enddo
  enddo
#endif


  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tf = tf + MPI_Wtime()
  DTFFT_CHECK(ierr)

  if ( executor /= DTFFT_EXECUTOR_NONE ) then
    block
      integer(int64) :: scale

      scale = nx * ny * nz
#if defined(DTFFT_WITH_CUDA)
      if ( platform == DTFFT_PLATFORM_CUDA )  then
        call scaleComplexFloat(c_loc(out), out_size, scale, dtfft_stream_t(c_null_ptr))
        CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
      else
        call scaleComplexFloatHost(c_loc(out), out_size, scale)
      endif
#else
      call scaleComplexFloatHost(c_loc(out), out_size, scale)
#endif
    end block
  endif

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaDeviceSynchronize", cudaDeviceSynchronize() )
  endif
#endif
  tb = tb + MPI_Wtime()

  local_error = 0._real32

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
  block
    complex(real32), allocatable, target :: test(:)

    allocate(test(in_size))

    CUDA_CALL( "cudaMemcpy", cudaMemcpy(c_loc(test), c_loc(in), element_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkComplexFloat(c_loc(check), c_loc(test), in_size)
    deallocate(test)
  endblock
  else
    local_error = checkComplexFloat(c_loc(check), c_loc(in), in_size)
  endif
#else
  local_error = checkComplexFloat(c_loc(check), c_loc(in), in_size)
#endif

  call report(tf, tb, local_error, nx, ny, nz)

  deallocate(check)

  call plan%mem_free(in, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_free(out, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d_float