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
program test_r2c_3d
use iso_fortran_env, only: R8P => real64, I4P => int32, I1P => int8, output_unit, error_unit
use iso_c_binding, only: c_size_t
use dtfft
use test_utils
#include "dtfft_mpi.h"
implicit none
#ifndef DTFFT_TRANSPOSE_ONLY
  real(R8P),     allocatable, target :: in(:), check(:,:,:)
  real(R8P),      pointer :: pin(:,:,:)
  complex(R8P),  allocatable :: out(:)
  real(R8P) :: local_error, rnd
  integer(I4P), parameter :: nx = 256, ny = 256, nz = 4
  integer(I4P) :: comm_size, comm_rank, i, j, k, ierr
  type(dtfft_executor_t) :: executor_type
  type(dtfft_plan_r2c_t) :: plan
  integer(I4P) :: in_counts(3)
  integer(c_size_t) :: alloc_size
  real(R8P) :: tf, tb

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2c_3d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ', nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif
#ifdef DTFFT_TRANSPOSE_ONLY
  if ( comm_rank == 0 ) &
    write(output_unit, '(a)') "R2C Transpose plan not supported, skipping it"

  call MPI_Finalize(ierr)
  stop
#endif

#if defined(DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
#endif

  call plan%create([nx, ny, nz], executor_type=executor_type, effort_type=DTFFT_MEASURE)

  call plan%get_local_sizes(in_counts = in_counts, alloc_size=alloc_size)

  allocate(in(alloc_size))
  allocate(check(1:in_counts(1), 1:in_counts(2), 1:in_counts(3)))
  allocate(out(alloc_size / 2), source = (0._R8P, 0._R8P))
  pin(1:in_counts(1), 1:in_counts(2), 1:in_counts(3)) => in

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd)
        pin(i,j,k) = 1.0d0
        check(i,j,k) = pin(i,j,k)
      enddo
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()

  ! Nullify recv buffer
  in = -1._R8P

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()
  in(:) = in(:) / real(nx * ny * nz, R8P)

  local_error = maxval(abs(pin - check))
  call report(tf, tb, local_error, nx, ny, nz)

  nullify(pin)
  deallocate(in)
  deallocate(out)
  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
#endif
end program test_r2c_3d