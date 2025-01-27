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
program test_r2r_2d_float
use iso_fortran_env, only: R8P => real64, R4P => real32, int32, I4P => int32, I1P => int8, output_unit, error_unit
use dtfft
use test_utils
#include "dtfft_mpi.h"
implicit none
  real(R4P),  allocatable :: in(:,:), out(:,:), check(:,:)
  real(R4P) :: local_error, rnd
  integer(I4P), parameter :: nx = 17, ny = 4
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  integer(I1P) :: executor_type
  type(dtfft_plan_r2r) :: plan
  integer(I4P) :: in_starts(2), in_counts(2), out_starts(2), out_counts(2)
  real(R8P) :: tf, tb, t_sum
  TYPE_MPI_COMM :: comm_1d

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_2d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3
#else
  executor_type = DTFFT_EXECUTOR_NONE
#endif

  call MPI_Cart_create(MPI_COMM_WORLD, 1, [comm_size], [.false.], .true., comm_1d, ierr)

  call plan%create([nx, ny], [DTFFT_DST_2, DTFFT_DST_3], comm=comm_1d, precision=DTFFT_SINGLE, executor_type=executor_type)
  call plan%get_local_sizes(in_starts, in_counts, out_starts, out_counts)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1,                     &
              in_starts(2):in_starts(2) + in_counts(2) - 1),  source = 0._R4P)

  allocate(check, source = in)

  allocate(out(out_starts(1):out_starts(1) + out_counts(1) - 1,                 &
                out_starts(2):out_starts(2) + out_counts(2) - 1), source = 0._R4P)

  do j = in_starts(2), in_starts(2) + in_counts(2) - 1
    do i = in_starts(1), in_starts(1) + in_counts(1) - 1
      call random_number(rnd)
      in(i,j) = rnd
      check(i,j) = in(i,j)
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()

  if ( executor_type /= DTFFT_EXECUTOR_NONE ) then
    out(:,:) = out(:,:) / real(4 * nx * ny, R4P)
  endif
  ! Nullify recv buffer
  in = -1._R4P

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()

  local_error = maxval(abs(in - check))
  call report(tf, tb, local_error, nx, ny)

  call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tf = t_sum / real(comm_size, R8P)
  call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  tb = t_sum / real(comm_size, R8P)

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_2d_float