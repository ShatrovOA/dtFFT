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
program test_c2c_2d
use iso_fortran_env, only: R8P => real64, I4P => int32, I1P => int8, output_unit, error_unit, int32, R4P => real32
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
#endif
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  complex(R8P),  allocatable :: in(:,:), out(:,:), check(:,:)
  real(R8P) :: local_error, rnd1, rnd2
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(I4P), parameter :: nx = 2048, ny = 2048
#else
  integer(I4P), parameter :: nx = 12, ny = 12
#endif
  integer(I4P) :: comm_size, comm_rank, i, j, ierr
  type(dtfft_plan_c2c_t) :: plan
  integer(I4P) :: in_starts(2), in_counts(2), out_starts(2), out_counts(2)
  real(R8P) :: tf, tb
  type(dtfft_executor_t) :: executor = DTFFT_EXECUTOR_NONE
  type(dtfft_config_t) :: conf


  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a,i0)') "dtFFT Version = ",dtfft_get_version()
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: c2c_2d             |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
    write(output_unit, '(a)') "----------------------------------------"
  endif

#if defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined (DTFFT_WITH_CUFFT)
  executor = DTFFT_EXECUTOR_CUFFT
#endif

#ifdef _OPENACC
  block
    use openacc
    integer(I4P) :: num_devices, my_device, host_rank, host_size
    TYPE_MPI_COMM :: host_comm

    call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank, MPI_INFO_NULL, host_comm, ierr)
    call MPI_Comm_rank(host_comm, host_rank, ierr)
    call MPI_Comm_size(host_comm, host_size, ierr)
    call MPI_Comm_free(host_comm, ierr)

    num_devices = acc_get_num_devices(acc_device_nvidia)
    if ( num_devices == 0 ) error stop "GPUs not found on host"
    if ( num_devices < host_size ) error stop "Number of GPU devices < Number of MPI processes"

    my_device = mod(host_rank, num_devices)
    ! print*,'setting device',comm_rank,my_device
    call acc_set_device_num(my_device, acc_device_nvidia)
  endblock
#endif

  conf = dtfft_config_t()

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  conf%gpu_backend = DTFFT_GPU_BACKEND_NCCL
  conf%platform = DTFFT_PLATFORM_CUDA
#endif
  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny], effort=DTFFT_MEASURE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_starts, in_counts, out_starts, out_counts, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)

  allocate(in(in_starts(1):in_starts(1) + in_counts(1) - 1,       &
              in_starts(2):in_starts(2) + in_counts(2) - 1))

  allocate(check, source = in)

  allocate(out(out_starts(1):out_starts(1) + out_counts(1) - 1,    &
                out_starts(2):out_starts(2) + out_counts(2) - 1))

!$acc enter data create(in, out)

  do j = in_starts(2), in_starts(2) + in_counts(2) - 1
    do i = in_starts(1), in_starts(1) + in_counts(1) - 1
      call random_number(rnd1)
      call random_number(rnd2)
      in(i,j) = cmplx(rnd1, rnd2, kind=R8P)
      check(i,j) = in(i,j)
    enddo
  enddo

!$acc update device(in)

  tf = 0.0_R8P - MPI_Wtime()
!$acc host_data use_device(in, out)
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
!$acc wait
  tf = tf + MPI_Wtime()
  if ( executor /= DTFFT_EXECUTOR_NONE ) then
!$acc kernels present(out)
    out(:,:) = out(:,:) / real(nx * ny, R8P)
!$acc end kernels
  endif
  ! Clear recv buffer
  in = (-1._R8P, -1._R8P)


  tb = 0.0_R8P - MPI_Wtime()
!$acc host_data use_device(in, out)
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
!$acc wait
  tb = tb + MPI_Wtime()

!$acc update self(in)

  local_error = maxval(abs(in - check))

  call report(tf, tb, local_error, nx, ny)

  deallocate(in, out, check)

  call plan%destroy(error_code=ierr); DTFFT_CHECK(ierr)
  call MPI_Finalize(ierr)
end program test_c2c_2d