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
program test_r2c_2d_float
use iso_fortran_env, only: R8P => real64, R4P => real32, I4P => int32, I8P => int64, output_unit, error_unit
use dtfft
use iso_c_binding
#include "dtfft_mpi.h"
implicit none
  real(R4P),     allocatable, target :: in(:), check(:,:)
  real(R4P),      pointer     :: pin(:,:)
  complex(R4P),  allocatable :: out(:)
  real(R4P) :: local_error, global_error, rnd
  integer(I4P), parameter :: nx = 17, ny = 19
  integer(I4P) :: comm_size, comm_rank, i, j, ierr, executor_type
  type(dtfft_plan_r2c) :: plan
  integer(I4P) :: in_counts(2), out_counts(2)
  real(R8P) :: tf, tb, t_sum
  integer(I8P) :: alloc_size

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2c_2d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif
#ifdef DTFFT_TRANSPOSE_ONLY
  if ( comm_rank == 0 ) &
    write(output_unit, '(a)') "R2C Transpose plan not supported, skipping it"

  call MPI_Finalize(ierr)
  stop
#endif

#if !defined(DTFFT_WITHOUT_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL
#endif
! #elif defined(DTFFT_WITH_KFR)
!   executor_type = DTFFT_EXECUTOR_KFR
! #endif

  call plan%create([nx, ny], precision=DTFFT_SINGLE, executor_type=executor_type)

  call plan%get_local_sizes(in_counts = in_counts, out_counts = out_counts, alloc_size=alloc_size)

  allocate(in(alloc_size))
  allocate(out(alloc_size / 2))
  allocate(check(1:in_counts(1), 1:in_counts(2)))

  pin(1:in_counts(1), 1:in_counts(2)) => in

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd)
      pin(i,j) = rnd
      check(i,j) = rnd
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()

  out(:) = out(:) / real(nx * ny, R4P)
  ! Nullify recv buffer
  in = -1._R4P

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
    if(global_error < 1.e-4) then
      write(output_unit, '(a)') "Test 'r2c_2d_float' PASSED!"
    else
      write(error_unit, '(a, f16.10)') "Test 'r2c_2d_float' FAILED... error = ", global_error
      error stop
    endif
    write(output_unit, '(a)') "----------------------------------------"
  endif


  deallocate(in, out, check)
  nullify( pin )
  call MPI_Finalize(ierr)
  print*,ierr, dtfft_get_error_string(ierr)
  !! Check that no error is raised when MPI is finalized
  call plan%destroy(ierr)
  print*,ierr, dtfft_get_error_string(ierr)
end program test_r2c_2d_float