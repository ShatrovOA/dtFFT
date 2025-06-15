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
use iso_fortran_env
use iso_c_binding
use dtfft
use test_utils
#include "dtfft_mpi.h"
implicit none (type, external)
#ifndef DTFFT_TRANSPOSE_ONLY
  real(real64),     allocatable, target :: in(:), check(:,:,:)
  real(real64),      pointer :: pin(:,:,:)
  complex(real64),  allocatable :: out(:)
  real(real64) :: local_error, rnd
  integer(int32), parameter :: nx = 256, ny = 256, nz = 4
  integer(int32) :: comm_size, comm_rank, i, j, k, ierr
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2c_t) :: plan
  integer(int32) :: in_counts(3)
  integer(int64) :: alloc_size, element_size
  real(real64) :: tf, tb
  type(dtfft_pencil_t) :: real_pencil, cmplx_pencil

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
  executor = DTFFT_EXECUTOR_MKL
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#endif

  call plan%create([nx, ny, nz], executor=executor, effort=DTFFT_MEASURE)

  call plan%get_local_sizes(in_counts = in_counts, alloc_size=alloc_size)
  element_size = plan%get_element_size()
  call plan%report()

  if ( element_size /= real64 ) error stop "element_size /= real64"

  real_pencil = plan%get_pencil(0)
  cmplx_pencil = plan%get_pencil(1)
  if ( any(in_counts /= real_pencil%counts) ) error stop "in_counts /= real_pencil%counts"
  if ( cmplx_pencil%counts(1) /= (nx / 2) + 1 ) error stop "cmplx_pencil%counts(1) /= (nx / 2) + 1"
  if ( any(real_pencil%counts(2:) /= cmplx_pencil%counts(2:)) ) error stop "cmplx_pencil%counts(1) /= (nx / 2) + 1"
  if ( comm_size == 1 ) then
    if ( any(real_pencil%counts /= [nx, ny, nz]) ) error stop "real_pencil%counts /= [nx, ny, nz]"
  endif

  allocate(in(alloc_size))
  allocate(check(1:in_counts(1), 1:in_counts(2), 1:in_counts(3)))
  allocate(out(alloc_size / 2), source = (0._real64, 0._real64))
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

  tf = 0.0_real64 - MPI_Wtime()
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD)
  tf = tf + MPI_Wtime()

  ! Nullify recv buffer
  in = -1._real64

  tb = 0.0_real64 - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD)
  tb = tb + MPI_Wtime()
  in(:) = in(:) / real(nx * ny * nz, real64)

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