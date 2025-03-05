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
program test_c2c_2d_float
use iso_fortran_env, only: R8P => real64, R4P => real32, I4P => int32, I8P => int64, I1P => int8, output_unit, error_unit
use iso_c_binding, only: c_ptr, c_f_pointer
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none

  type(c_ptr) :: in, out
  complex(R4P), allocatable :: check(:,:)
  complex(R4P), DEVICE_PTR pointer :: pin(:,:), pout(:,:)
  real(R4P) :: local_error, rnd1, rnd2
  integer(I4P), parameter :: nx = 64, ny = 32
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_executor_t) :: executor
  integer(I8P) :: alloc_size, element_size
  type(dtfft_plan_c2c_t) :: plan
  integer(I4P) :: in_counts(2), out_counts(2)
  real(R8P) :: tf, tb
  type(dtfft_config_t) :: conf

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_2d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#else
  executor = DTFFT_EXECUTOR_NONE
#endif

  conf = dtfft_config_t()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  conf%gpu_backend = DTFFT_GPU_BACKEND_NCCL
  conf%platform = DTFFT_PLATFORM_CUDA
#endif
  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny], precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)
  element_size = plan%get_element_size()

  call plan%mem_alloc(alloc_size * element_size, in, ierr);  DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_size * element_size, out, ierr); DTFFT_CHECK(ierr)

  allocate(check(in_counts(1),in_counts(2)))

  call c_f_pointer(in, pin, in_counts)
  call c_f_pointer(out, pout, out_counts)

  do j = 1, in_counts(2)
    do i = 1, in_counts(1)
      call random_number(rnd1)
      call random_number(rnd2)
      pin(i,j) = cmplx(rnd1, rnd2)
      check(i,j) = pin(i,j)
    enddo
  enddo

  tf = 0.0_R8P - MPI_Wtime()
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD)
  tf = tf + MPI_Wtime()
#ifndef DTFFT_WITH_CUDA
  if ( executor /= DTFFT_EXECUTOR_NONE ) then
    pout(:,:) = pout(:,:) / real(nx * ny, R4P)
  endif
#endif
  ! Nullify recv buffer
#ifndef DTFFT_WITH_CUDA
  pin(:,:) = cmplx(-1._R4P, -1._R4P)
#endif
  tb = 0.0_R8P - MPI_Wtime()
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD)
  tb = tb + MPI_Wtime()

  local_error = maxval(abs(pin - check))
  call report(tf, tb, local_error, nx, ny)

  call plan%mem_free(in)
  call plan%mem_free(out)

  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_2d_float