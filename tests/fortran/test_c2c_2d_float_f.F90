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
use iso_fortran_env, only: R8P => real64, R4P => real32, I4P => int32, I8P => int64, output_unit, error_unit
use dtfft
use iso_c_binding
#include "dtfft_mpi.h"
implicit none
  complex(R4P),  allocatable, target :: in(:), out(:), check(:,:)
  complex(R4P),  pointer :: pin(:,:), pout(:,:)
  real(R4P) :: local_error, global_error, rnd1, rnd2
  integer(I4P), parameter :: nx = 64, ny = 32
  integer(I4P) :: comm_size, comm_rank, i, j, ierr, executor_type
  integer(I8P) :: alloc_size
  type(dtfft_plan_c2c) :: plan
  integer(I4P) :: in_counts(2), out_counts(2)
  real(R8P) :: tf, tb, t_sum

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
  executor_type = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL
! #elif defined(DTFFT_WITH_KFR)
!   executor_type = DTFFT_EXECUTOR_KFR
#else
  executor_type = DTFFT_EXECUTOR_NONE
#endif

  call plan%create([nx, ny], precision=DTFFT_SINGLE, executor_type=executor_type)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size)

  allocate(in(alloc_size))
  allocate(check(in_counts(1),in_counts(2)))
  allocate(out(alloc_size))

  pin(1:in_counts(1), 1:in_counts(2)) => in
  pout(1:out_counts(1), 1:out_counts(2)) => out

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd1)
      call random_number(rnd2)
      pin(i,j) = cmplx(rnd1, rnd2)
      check(i,j) = pin(i,j)
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()
#ifndef DTFFT_TRANSPOSE_ONLY
  pout(:,:) = pout(:,:) / real(nx * ny, R4P)
#endif
  ! Nullify recv buffer
  in = (-1._R4P, -1._R4P)

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()

  call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tf = t_sum / real(comm_size, R8P)
  call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tb = t_sum / real(comm_size, R8P)

  if(comm_rank == 0) then
    write(output_unit, '(a, f16.10)') "Forward execution time: ", tf
    write(output_unit, '(a, f16.10)') "Backward execution time: ", tb
    write(output_unit, '(a)') "----------------------------------------"
  endif

  local_error = maxval(abs(pin - check))

  call MPI_Allreduce(local_error, global_error, 1, MPI_REAL, MPI_MAX, MPI_COMM_WORLD, ierr)
  if(comm_rank == 0) then
    if(global_error < 1.e-6) then
      write(output_unit, '(a)') "Test 'c2c_2d_float' PASSED!"
    else
      write(error_unit, '(a, f16.10)') "Test 'c2c_2d_float' FAILED... error = ", global_error
      error stop
    endif
    write(output_unit, '(a)') "----------------------------------------"
  endif

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_2d_float