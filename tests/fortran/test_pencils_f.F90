!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
use iso_fortran_env
use dtfft
use test_utils, only: attach_gpu_to_process
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  integer(int32) :: comm_size, comm_rank, ierr
  type(dtfft_plan_r2r_t) :: plan
  integer(int32) :: lbounds(3), sizes(3)
  type(dtfft_pencil_t) :: pencil
  type(dtfft_pencil_t) :: test, test2
  integer(int32), pointer :: dims(:)


  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)
  call attach_gpu_to_process()

  if ( comm_size /= 4 ) then
    if ( comm_rank == 0 ) write(output_unit, "(a)") "This test requires 4 MPI processes..."
    call MPI_Finalize(ierr)
    stop
  endif

  if ( comm_rank == 0 ) then
    lbounds = [-1, 0, 0]
    sizes = [4, 8, 12]
  elseif ( comm_rank == 1 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 8, 12]
  else if ( comm_rank == 2 ) then
    lbounds = [0, 0, 8]
    sizes = [4, 8, 12]
  else
    lbounds = [0, 0, 8]
    sizes = [4, 8, 12]
  endif

  allocate( pencil%counts, source=sizes )
  allocate( pencil%starts, source=lbounds )

  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_NOT_INITIALIZED ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_NOT_INITIALIZED"
  endif

  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_INVALID_STARTS ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_INVALID_STARTS"
  endif

  if ( comm_rank == 0 ) then
    lbounds = [0, 0, 0]
  endif
  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_OVERLAP ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_OVERLAP"
  endif

  if ( comm_rank == 0 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 8, 12]
  elseif ( comm_rank == 1 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 8, 12]
  else if ( comm_rank == 2 ) then
    lbounds = [0, 0, 12]
    sizes = [4, 8, 12]
  else
    lbounds = [0, 0, 13]
    sizes = [4, 8, 12]
  endif
  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_OVERLAP ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_OVERLAP"
  endif

  if ( comm_rank == 0 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 12, 12]
  elseif ( comm_rank == 1 ) then
    lbounds = [0, 8, 0]
    sizes = [4, 8, 12]
  else if ( comm_rank == 2 ) then
    lbounds = [0, 8, 12]
    sizes = [4, 8, 12]
  else
    lbounds = [0, 0, 12]
    sizes = [4, 8, 12]
  endif
  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_SHAPE_MISMATCH ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_SHAPE_MISMATCH"
  endif

  if ( comm_rank == 0 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 8, 12]
  elseif ( comm_rank == 1 ) then
    lbounds = [0, 10, 0]
    sizes = [4, 8, 12]
  else if ( comm_rank == 2 ) then
    lbounds = [0, 10, 12]
    sizes = [4, 8, 12]
  else
    lbounds = [0, 0, 12]
    sizes = [4, 8, 12]
  endif
  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_ERROR_PENCIL_NOT_CONTINUOUS ) then
    error stop "ierr /= DTFFT_ERROR_PENCIL_NOT_CONTINUOUS"
  endif

  ! Allowed since 3.0.0
  !
  ! if ( comm_rank == 0 ) then
  !   lbounds = [0, 0, 0]
  !   sizes = [4, 8, 12]
  ! elseif ( comm_rank == 1 ) then
  !   lbounds = [4, 0, 0]
  !   sizes = [4, 8, 12]
  ! else if ( comm_rank == 2 ) then
  !   lbounds = [8, 0, 0]
  !   sizes = [4, 8, 12]
  ! else
  !   lbounds = [12, 0, 0]
  !   sizes = [4, 8, 12]
  ! endif
  ! pencil = dtfft_pencil_t(lbounds, sizes)
  ! call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  ! if ( ierr /= DTFFT_ERROR_INVALID_COMM_FAST_DIM ) then
  !   error stop "ierr /= DTFFT_ERROR_INVALID_COMM_FAST_DIM"
  ! endif

  if ( comm_rank == 0 ) then
    lbounds = [0, 0, 0]
    sizes = [4, 8, 12]
  elseif ( comm_rank == 1 ) then
    lbounds = [0, 8, 0]
    sizes = [4, 8, 12]
  else if ( comm_rank == 2 ) then
    lbounds = [0, 8, 12]
    sizes = [4, 8, 12]
  else
    lbounds = [0, 0, 12]
    sizes = [4, 8, 12]
  endif
  pencil = dtfft_pencil_t(lbounds, sizes)
  call plan%create(pencil, effort=DTFFT_PATIENT, error_code=ierr)
  if ( ierr /= DTFFT_SUCCESS ) then
    error stop "ierr /= DTFFT_SUCCESS"
  endif

  test = plan%get_pencil(DTFFT_LAYOUT_X_PENCILS, error_code=ierr); DTFFT_CHECK(ierr)
  if ( any( lbounds /= test%starts ) ) error stop "any( lbounds /= test%starts )"
  if ( any( sizes /= test%counts ) ) error stop " any( sizes /= test%counts )"

  test = plan%get_pencil(DTFFT_LAYOUT_Y_PENCILS, error_code=ierr); DTFFT_CHECK(ierr)
  if ( test%starts(2) /= lbounds(3) ) error stop "test%starts(2) /= lbounds(3)"
  if ( test%counts(2) /= sizes(3) ) error stop "test%counts(2) /= sizes(3)"

  test2 = plan%get_pencil(DTFFT_LAYOUT_Z_PENCILS, error_code=ierr); DTFFT_CHECK(ierr)
  if ( test2%starts(2) /= test%starts(3) ) error stop "test2%starts(2) /= test%starts(2)"
  if ( test2%counts(2) /= test%counts(3) ) error stop "test2%counts(2) /= test%counts(2)"


  call plan%get_dims(dims, error_code=ierr); DTFFT_CHECK(ierr)
  if ( any( dims /= [4, 16, 24] ) ) error stop "any( dims /= [4, 16, 24] )"

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_2d