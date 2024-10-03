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
use iso_fortran_env, only: R8P => real64, R4P => real32, I4P => int32, I8P => int64, output_unit, error_unit
use dtfft
#include "dtfft_mpi.h"
implicit none
  real(R4P),  allocatable :: inout(:), check(:)
  real(R4P) :: local_error, global_error, rnd
  real(R4P) :: scaler
  integer(I4P), parameter :: nx = 32, ny = 64, nz = 16
  integer(I4P) :: comm_size, comm_rank, ierr, executor_type, in_counts(3), in_product
  type(dtfft_plan_r2r) :: plan
  real(R8P) :: tf, tb, t_sum
  integer(I8P)  :: alloc_size, i

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

! #ifdef DTFFT_WITH_KFR
!   executor_type = DTFFT_EXECUTOR_KFR
!   scaler = 8._R4P / real(nx * ny * nz, R4P)
#if !defined(DTFFT_WITHOUT_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
  scaler = 1._R4P / real(8 * (nx - 1) * ny * nz, R4P)
#else
  executor_type = DTFFT_EXECUTOR_NONE
  scaler = 1._R4P
#endif
  call plan%create([nx, ny, nz], [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3], precision=DTFFT_SINGLE, executor_type=executor_type)
  call plan%get_local_sizes(in_counts=in_counts, alloc_size=alloc_size)

  in_product = product(in_counts)
  allocate(inout(alloc_size))
  allocate(check(in_product))

  do i = 1, in_product
    call random_number(rnd)
    inout(i) = rnd
    check(i) = inout(i)
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()

  inout = inout * scaler

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()

  call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tf = t_sum / real(comm_size, R8P)
  call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tb = t_sum / real(comm_size, R8P)

  if(comm_rank == 0) then
    write(output_unit, '(a, f16.10)') "Forward transposition time: ", tf
    write(output_unit, '(a, f16.10)') "Backward transposition time: ", tb
  endif

  local_error = maxval(abs(inout(:in_product) - check(:in_product)))

  call MPI_Allreduce(local_error, global_error, 1, MPI_REAL, MPI_MAX, MPI_COMM_WORLD, ierr)
  if(comm_rank == 0) then
    if(global_error < 1.e-5) then
      write(output_unit, '(a)') "Test 'r2r_3d_float' PASSED!"
    else
      write(error_unit, '(a, d16.5)') "Test 'r2r_3d_float' FAILED... error = ", global_error
      error stop
    endif
  endif

  deallocate(inout, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_3d_float