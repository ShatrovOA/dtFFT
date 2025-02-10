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
program test_r2r_2d
use iso_fortran_env, only: R4P => real32, R8P => real64, I4P => int32, I1P => int8, output_unit, error_unit
use dtfft
use test_utils
#include "dtfft_mpi.h"
implicit none
  real(R8P),  allocatable :: in(:,:), out(:,:), check(:,:)
  real(R8P) :: local_error, scaler, rnd
  integer(I4P), parameter :: nx = 8, ny = 12
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_plan_r2r_t) :: plan
  integer(I4P) :: in_starts(2), in_counts(2), out_starts(2), out_counts(2), in_vals
  real(R8P) :: tf, tb
  type(dtfft_r2r_kind_t) :: kinds(2)
  type(dtfft_executor_t) :: executor_type

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

  kinds = DTFFT_DCT_1
#if defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
  scaler = 1._R8P / real(4 * (nx - 1) * (ny - 1), R8P)
#else
  executor_type = DTFFT_EXECUTOR_NONE
  scaler = 1._R8P
#endif

  call plan%create([nx, ny], kinds=kinds, effort_type=DTFFT_PATIENT, executor_type=executor_type)

  call plan%get_local_sizes(in_starts, in_counts, out_starts, out_counts)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1, &
              in_starts(2):in_starts(2) + in_counts(2) - 1) )

  allocate(out(out_starts(1):out_starts(1) + out_counts(1) - 1,   &
               out_starts(2):out_starts(2) + out_counts(2) - 1)   )

  in_vals = product(in_counts)
  do j = in_starts(2), in_starts(2) + in_counts(2) - 1
    do i = in_starts(1), in_starts(1) + in_counts(1) - 1
      call random_number(rnd)
      in(i,j) = rnd
    enddo
  enddo

  allocate(check, source = in)

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()


  out(:,:) = out(:,:) * scaler
  ! Nullify recv buffer
  in = -1._R8P

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()

  local_error = maxval(abs(in - check))

  call report(tf, tb, local_error, nx, ny)

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_2d