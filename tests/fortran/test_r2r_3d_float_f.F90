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
use iso_fortran_env, only: R8P => real64, R4P => real32, I4P => int32, I8P => int64, I1P => int8, output_unit, error_unit, int32
use dtfft
use dtfft_utils
use test_utils
#ifdef DTFFT_WITH_CUDA
use cudafor
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  real(R4P),  allocatable :: inout(:), check(:)
#ifdef DTFFT_WITH_CUDA
  real(R4P), managed, allocatable :: d_inout(:)
#endif
  real(R4P) :: local_error, rnd
  integer(I4P), parameter :: nx = 512, ny = 64, nz = 16
  integer(I4P) :: comm_size, comm_rank, ierr, in_counts(3), in_product
  integer(I1P) :: executor_type
  type(dtfft_plan_r2r) :: plan
  real(R8P) :: tf, tb
  integer(I8P)  :: alloc_size, i
#ifdef DTFFT_WITH_CUDA
  integer(cuda_stream_kind) :: stream
  integer(I4P) :: host_rank, host_size, num_devices
#endif

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

  executor_type = DTFFT_EXECUTOR_NONE

  call attach_gpu_to_process()

#ifdef DTFFT_WITH_CUDA
  call dtfft_set_gpu_backend(DTFFT_GPU_BACKEND_NCCL_PIPELINED, error_code=ierr)
  DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Using backend: "//dtfft_get_gpu_backend_string(DTFFT_GPU_BACKEND_NCCL_PIPELINED)
  endif

  CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(stream) )
  call dtfft_set_stream(stream, error_code=ierr)
  DTFFT_CHECK(ierr)
#endif

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor_type=executor_type, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

  in_product = product(in_counts)
  allocate(inout(alloc_size))
  allocate(check(in_product))

  do i = 1, in_product
    call random_number(rnd)
    inout(i) = rnd
    check(i) = inout(i)
  enddo

#ifdef DTFFT_WITH_CUDA
  allocate(d_inout(alloc_size))
  d_inout(:) = inout(:)
#endif

  tf = 0.0_R8P - MPI_Wtime()
#ifdef DTFFT_WITH_CUDA
  call plan%execute(d_inout, d_inout, DTFFT_TRANSPOSE_OUT, error_code=ierr)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#else
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_OUT, error_code=ierr)
#endif
  DTFFT_CHECK(ierr)
  tf = tf + MPI_Wtime()

  tb = 0.0_R8P - MPI_Wtime()
#ifdef DTFFT_WITH_CUDA
  call plan%execute(d_inout, d_inout, DTFFT_TRANSPOSE_IN, error_code=ierr)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  inout(:) = d_inout(:)
#else
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_IN)
#endif
  DTFFT_CHECK(ierr)
  tb = tb + MPI_Wtime()

  local_error = maxval(abs(inout(:in_product) - check(:in_product)))

  call report("r2r_3d_float", tf, tb, local_error)

  deallocate(inout, check)
#ifdef DTFFT_WITH_CUDA
  deallocate(d_inout)
  CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(stream) )
#endif

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_3d_float