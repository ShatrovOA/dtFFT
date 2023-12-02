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
program test_c2c_3d
use iso_fortran_env, only: R8P => real64, I4P => int32, I8P => int64, output_unit, error_unit
use dtfft
use iso_c_binding
#include "dtfft.i90"
implicit none
  complex(R8P),  allocatable :: in(:,:,:), out(:,:,:), check(:,:,:), work(:)
  real(R8P) :: local_error, global_error, rnd1, rnd2
  integer(I4P), parameter :: nx = 512, ny = 128, nz = 64
  integer(I4P) :: comm_size, comm_rank, i, j, k, ierr
  type(dtfft_plan_c2c) :: plan
  integer(I4P) :: in_counts(3), out_counts(3)
  integer(I8P)  :: alloc_size
  real(R8P) :: tf, tb, t_sum

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

  call plan%create([nx, ny, nz])
  call plan%get_local_sizes(in_counts = in_counts, out_counts = out_counts)

  allocate(in(in_counts(1),in_counts(2),in_counts(3)), source = (0._R8P, 0._R8P))

  allocate(check, source = in)

  allocate(out(out_counts(1), out_counts(2), out_counts(3)), source = (0._R8P, 0._R8P))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        in(i,j,k) = cmplx(rnd1, rnd1, R8P)
        check(i,j,k) = in(i,j,k)
      enddo
    enddo
  enddo

  alloc_size = plan%get_aux_size()
  allocate(work(alloc_size), source = (-1._R8P, -1._R8P))

!$acc data copy(in) copyin(out, work)

  tf = 0.0_R8P - MPI_Wtime()

  call plan%execute(in, out, DTFFT_TRANSPOSE_OUT, work)
  tf = tf + MPI_Wtime()

  out(:,:,:) = out(:,:,:) / real(nx * ny * nz, R8P)

  ! Nullify recv buffer
  in = (-1._R8P, -1._R8P)

  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_TRANSPOSE_IN, work)
  tb = tb + MPI_Wtime()

!$acc end data

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
    if(global_error < 1.d-10) then
      write(output_unit, '(a)') "Test 'c2c_3d' PASSED!"
    else
      write(error_unit, '(a, f16.10)') "Test 'c2c_3d' FAILED... error = ", global_error
      error stop
    endif
    write(output_unit, '(a)') "----------------------------------------"
  endif

  deallocate(in, out, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d