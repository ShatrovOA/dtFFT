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
use iso_fortran_env, only: R8P => real64, I4P => int32, output_unit, error_unit
use dtfft
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  complex(R8P),  allocatable :: in(:,:), out(:,:), check(:,:)
  real(R8P) :: local_error, global_error, rnd1, rnd2
  integer(I4P), parameter :: nx = 12, ny = 12
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_plan_c2c) :: plan
  integer(I4P) :: in_starts(2), in_counts(2), out_starts(2), out_counts(2)
  real(R8P) :: tf, tb, t_sum
  integer(I4P) :: executor_type = DTFFT_EXECUTOR_NONE

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined(DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL
! #elif defined(DTFFT_WITH_KFR)
!   executor_type = DTFFT_EXECUTOR_KFR
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
#endif

  call plan%create([nx, ny], effort_flag=DTFFT_PATIENT, executor_type=executor_type, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, error_code=ierr); DTFFT_CHECK(ierr)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1,       &
              in_starts(2):in_starts(2) + in_counts(2) - 1))

  allocate(check, source = in)

  allocate(out(out_starts(1):out_starts(1) + out_counts(1) - 1,    &
                out_starts(2):out_starts(2) + out_counts(2) - 1))

  do j = in_starts(2), in_starts(2) + in_counts(2) - 1
    do i = in_starts(1), in_starts(1) + in_counts(1) - 1
      call random_number(rnd1)
      call random_number(rnd2)
      in(i,j) = cmplx(rnd1, rnd2, kind=R8P)
      check(i,j) = in(i,j)
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT, error_code=ierr); DTFFT_CHECK(ierr)
  tf = tf + MPI_Wtime()
#ifndef DTFFT_TRANSPOSE_ONLY
  out(:,:) = out(:,:) / real(nx * ny, R8P)
#endif
  ! Clear recv buffer
  in = (-1._R8P, -1._R8P)


  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN, error_code=ierr); DTFFT_CHECK(ierr)
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

  local_error = maxval(abs(in - check))

  call MPI_Allreduce(local_error, global_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
  if(comm_rank == 0) then
    if(global_error < 1.d-10) then
      write(output_unit, '(a)') "Test 'c2c_2d' PASSED!"
    else
      write(error_unit, '(a, f16.10)') "Test 'c2c_2d' FAILED... error = ", global_error
      error stop
    endif
    write(output_unit, '(a)') "----------------------------------------"
  endif

  deallocate(in, out, check)

  call plan%destroy(error_code=ierr); DTFFT_CHECK(ierr)
  call MPI_Finalize(ierr)
end program test_c2c_2d