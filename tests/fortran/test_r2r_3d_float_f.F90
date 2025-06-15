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
program test_r2r_3d_float
use iso_fortran_env
use dtfft
use dtfft_utils
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none (type, external)
  real(real32),  allocatable :: inout(:), check(:)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  real(real32), managed, allocatable :: d_inout(:)
#endif
  real(real32) :: local_error, rnd
  integer(int32), parameter :: nx = 512, ny = 64, nz = 16
  integer(int32) :: comm_size, comm_rank, ierr, in_counts(3), in_product
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2r_t) :: plan
  real(real64) :: tf, tb
  integer(int64)  :: alloc_size, i
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: stream
  integer(int32) :: host_rank, host_size, num_devices
  type(dtfft_backend_t) :: backend_to_use, actual_backend_used
#endif
  type(dtfft_config_t) :: conf

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_3d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
  endif

  executor = DTFFT_EXECUTOR_NONE

  call attach_gpu_to_process()

  call dtfft_create_config(conf)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  backend_to_use = DTFFT_BACKEND_NCCL_PIPELINED
  conf%backend = backend_to_use

  CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(stream) )
  conf%stream = dtfft_stream_t(stream)

  conf%platform = DTFFT_PLATFORM_CUDA
#endif

  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  actual_backend_used = plan%get_backend(error_code=ierr);  DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Using backend: "//dtfft_get_backend_string(actual_backend_used)
  endif
  if ( comm_size > 1 .and. actual_backend_used /= backend_to_use ) then
    error stop "Invalid backend: actual_backend_used /= backend_to_use"
  endif
#endif

  in_product = product(in_counts)
  allocate(inout(alloc_size))
  allocate(check(in_product))

  do i = 1, in_product
    call random_number(rnd)
    inout(i) = rnd
    check(i) = inout(i)
  enddo

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  allocate(d_inout(alloc_size))
  d_inout(:) = inout(:)
#endif

  tf = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%execute(d_inout, d_inout, DTFFT_EXECUTE_FORWARD, error_code=ierr)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#else
  call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, error_code=ierr)
#endif
  DTFFT_CHECK(ierr)
  tf = tf + MPI_Wtime()

  tb = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%execute(d_inout, d_inout, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  inout(:) = d_inout(:)
#else
  call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD)
#endif
  DTFFT_CHECK(ierr)
  tb = tb + MPI_Wtime()

  local_error = maxval(abs(inout(:in_product) - check(:in_product)))

  call report(tf, tb, local_error, nx, ny, nz)

  deallocate(inout, check)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  deallocate(d_inout)
  CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(stream) )
#endif

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_3d_float