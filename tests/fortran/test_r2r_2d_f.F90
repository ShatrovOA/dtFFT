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
program test_r2r_2d
use iso_fortran_env, only: R8P => real64, I4P => int32, output_unit, error_unit
use dtfft
use mpi_f08
use iso_c_binding
implicit none
include 'fftw3.f03'
  real(R8P),  allocatable :: in(:,:), out(:,:), check(:,:)
  real(R8P) :: err, max_error, rnd
  integer(I4P), parameter :: nx = 2048, ny = 1024
  integer(I4P) :: comm_size, comm_rank, i, j
  type(dtfft_plan_r2r_2d) :: plan
  integer(I4P) :: in_starts(2), in_counts(2), out_starts(2), out_counts(2)
  real(R8P) :: tf, tb, t_sum
  integer(I4P) :: f_kinds(2), b_kinds(2)
  
  call MPI_Init()
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

  f_kinds = FFTW_REDFT10
  b_kinds = FFTW_REDFT01
  call plan%create(MPI_COMM_WORLD, nx, ny, f_kinds, b_kinds)
  
  call plan%get_local_sizes(in_starts, in_counts, out_starts, out_counts)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1,                     &
              in_starts(2):in_starts(2) + in_counts(2) - 1),  source = 0._R8P)

  allocate(check, source = in)

  allocate(out(out_starts(1):out_starts(1) + out_counts(1) - 1,                 &
                out_starts(2):out_starts(2) + out_counts(2) - 1), source = 0._R8P)

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

  out(:,:) = out(:,:) / real(4 * nx * ny, R8P)

  ! Nullify recv buffer
  in = -1._R8P

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN)
  tb = tb + MPI_Wtime()

  call MPI_Allreduce(tf, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD)
  tf = t_sum / real(comm_size, R8P)
  call MPI_Allreduce(tb, t_sum, 1, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD)
  tb = t_sum / real(comm_size, R8P)

  if(comm_rank == 0) then 
    write(output_unit, '(a, f16.10)') "Forward execution time: ", tf
    write(output_unit, '(a, f16.10)') "Backward execution time: ", tb
  endif

  err = maxval(abs(in - check))

  call MPI_Allreduce(err, max_error, 1, MPI_REAL8, MPI_MAX, MPI_COMM_WORLD)
  if(comm_rank == 0) then
    if(max_error < 1.d-10) then
      write(output_unit, '(a)') "Test 'r2r_2d' PASSED!"
    else
      write(error_unit, '(a, d16.5)') "Test 'r2r_2d' FAILED... error = ", max_error
    endif
  endif

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize()
end program test_r2r_2d