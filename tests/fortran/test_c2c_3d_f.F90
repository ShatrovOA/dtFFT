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
use iso_fortran_env, only: R8P => real64, I4P => int32, I8P => int64, output_unit, error_unit
use dtfft
use iso_c_binding
#include "dtfft_mpi.h"
implicit none
  complex(R8P),  allocatable :: inout(:), check(:,:,:), aux(:)
  real(R8P) :: err, local_error, global_error, rnd1, rnd2
  integer(I4P), parameter :: nx = 129, ny = 123, nz = 33
  integer(I4P) :: comm_size, comm_rank, i, j, k, ierr, ii, jj, kk, idx, executor_type
  type(dtfft_plan_c2c) :: plan
  integer(I4P) :: in_counts(3), out_counts(3)
  integer(I8P)  :: alloc_size
  real(R8P) :: tf, tb, t_sum, t_total

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
  executor_type = DTFFT_EXECUTOR_MKL
#elif !defined(DTFFT_WITHOUT_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
! #elif defined(DTFFT_WITH_KFR)
!   executor_type = DTFFT_EXECUTOR_KFR
#else
  executor_type = DTFFT_EXECUTOR_NONE
#endif

  call plan%create([nx, ny, nz], executor_type=executor_type, effort_flag=DTFFT_PATIENT)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size)

  allocate(inout(alloc_size))
  allocate(aux(alloc_size))
  allocate(check(in_counts(1), in_counts(2), in_counts(3)))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        ii = i - 1; jj = j - 1; kk = k - 1
        idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
        inout(idx) = cmplx(rnd1, rnd1, R8P)
        check(i,j,k) = inout(idx)
      enddo
    enddo
  enddo

  t_total = -MPI_Wtime()

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_OUT, aux)
  tf = tf + MPI_Wtime()
#ifndef DTFFT_TRANSPOSE_ONLY
  inout(:) = inout(:) / real(nx * ny * nz, R8P)
#endif
  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(inout, inout, DTFFT_TRANSPOSE_IN, aux)
  tb = tb + MPI_Wtime()

  t_total = t_total + MPI_Wtime()

  call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tf = t_sum / real(comm_size, R8P)
  call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tb = t_sum / real(comm_size, R8P)

  if(comm_rank == 0) then
    write(output_unit, '(a, f16.10)') "Forward execution time: ", tf
    write(output_unit, '(a, f16.10)') "Backward execution time: ", tb
  endif

  local_error = 0._R8P
  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        ii = i - 1; jj = j - 1; kk = k - 1
        idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
        err = abs(inout(idx) - check(i,j,k))
        if ( err > local_error ) local_error = err
      enddo
    enddo
  enddo

  call MPI_Allreduce(local_error, global_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)
  if(comm_rank == 0) then
    if(global_error < 1.d-10) then
      write(output_unit, '(a)') "Test 'c2c_3d' PASSED!"
    else
      write(error_unit, '(a, f16.10)') "Test 'c2c_3d' FAILED... error = ", global_error
      error stop
    endif
    write(output_unit, '(a)') "----------------------------------------"
  endif

  deallocate(inout, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d