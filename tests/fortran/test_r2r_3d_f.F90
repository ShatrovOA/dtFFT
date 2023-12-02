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
program test_r2r_3d
!------------------------------------------------------------------------------------------------
!< This program shows how to use DTFFT with Real-to-Real 3d transform
!< It also tests user-defined 1d communicator
!------------------------------------------------------------------------------------------------
use iso_fortran_env, only: R8P => real64, I4P => int32, IP => int32, output_unit, error_unit, I8P => int64
use dtfft
#include "dtfft.i90"
implicit none
  real(R8P),  allocatable :: in(:,:,:), out(:), check(:,:,:)
  real(R8P) :: local_error, global_error, rnd
  integer(I4P), parameter :: nx = 256, ny = 256, nz = 128
  integer(I4P) :: comm_size, comm_rank, i, j, k
  type(dtfft_plan_r2r) :: plan
  integer(I4P) :: in_starts(3), in_counts(3), ierr
  real(R8P) :: tf, tb, t_sum
  TYPE_MPI_COMM :: comm_1d
  integer(I8P) :: alloc_size

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_3d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
  endif

  call MPI_Cart_create(MPI_COMM_WORLD, 1, [comm_size], [.false.], .true., comm_1d, ierr)

  call plan%create([nx, ny, nz], [DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_2], [DTFFT_DCT_3, DTFFT_DCT_2, DTFFT_DCT_3], comm=comm_1d)

  call plan%get_local_sizes(in_starts, in_counts, alloc_size=alloc_size)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1,                     &
              in_starts(2):in_starts(2) + in_counts(2) - 1,                     &
              in_starts(3):in_starts(3) + in_counts(3) - 1),  source = 0._R8P)

  allocate(check, source = in)

  allocate(out(alloc_size), source=0._R8P)

  do k = in_starts(3), in_starts(3) + in_counts(3) - 1
    do j = in_starts(2), in_starts(2) + in_counts(2) - 1
      do i = in_starts(1), in_starts(1) + in_counts(1) - 1
        call random_number(rnd)
        in(i,j,k) = real(k * ny * nx + j * nx + i, R8P) / real(nx * ny * nz, R8P)
        check(i,j,k) = in(i,j,k)
      enddo
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT)
  tf = tf + MPI_Wtime()

  out = out / real(8 * nx * ny * nz, R8P)

  ! Nullify recv buffer
  in = -1._R8P

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
  endif

  local_error = maxval(abs(in - check))

  call MPI_Allreduce(local_error, global_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD, ierr)

  if(comm_rank == 0) then
    if(global_error < 1.e-14) then
      write(output_unit, '(a)') "Test 'r2r_3d' PASSED!"
    else
      write(error_unit, '(a, d16.5)') "Test 'r2r_3d' FAILED... error = ", global_error
      error stop
    endif
  endif

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_3d