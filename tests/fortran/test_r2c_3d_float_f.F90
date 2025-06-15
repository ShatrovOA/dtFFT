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
program test_r2c_3d_float
use iso_fortran_env
use iso_c_binding, only: c_size_t
use dtfft
use test_utils
#include "dtfft_mpi.h"
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
use dtfft_utils
#include "dtfft_cuda.h"
#endif
#include "dtfft.f03"
implicit none (type, external)
#ifndef DTFFT_TRANSPOSE_ONLY
  real(real32),     allocatable :: in(:,:,:), check(:,:,:)
  complex(real32),  allocatable :: out(:)
  real(real32) :: local_error, rnd
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(int32), parameter :: nx = 513, ny = 711, nz = 33
#else
  integer(int32), parameter :: nx = 16, ny = 8, nz = 4
#endif
  integer(int32) :: comm_size, comm_rank, i, j, k, ierr
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2c_t) :: plan
  integer(int32) :: in_counts(3)
  integer(c_size_t) :: alloc_size
  real(real64) :: tf, tb
  type(dtfft_config_t) :: conf
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: stream
#endif

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2c_3d_float       |"
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

  call dtfft_create_config(conf)

#ifdef _OPENACC
  block
    use openacc
    integer(int32) :: num_devices, my_device, host_rank, host_size
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

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#elif defined(DTFFT_WITH_CUFFT)
  executor = DTFFT_EXECUTOR_CUFFT
#elif defined(DTFFT_WITH_VKFFT)
  executor = DTFFT_EXECUTOR_VKFFT
  conf%enable_z_slab = .false.
#endif

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  conf%backend = DTFFT_BACKEND_NCCL
  conf%platform = DTFFT_PLATFORM_CUDA
#endif

  call dtfft_set_config(conf)

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts = in_counts, alloc_size = alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%get_stream(stream, error_code=ierr)
  DTFFT_CHECK(ierr)
#endif

  allocate(in(in_counts(1),in_counts(2), in_counts(3)), source = -33._real32)

  allocate(check, source = in)

  allocate(out(alloc_size), source = (0._real32, 0._real32))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd)
        in(i,j,k) = rnd
        check(i,j,k) = in(i,j,k)
      enddo
    enddo
  enddo

!$acc enter data create(out) copyin(in, check)

  tf = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(in, out)
  call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
  tf = tf + MPI_Wtime()
!$acc kernels present(out)
  out(:) = out(:) / real(nx * ny * nz, real32)
!$acc end kernels
  ! Nullify recv buffer
!$acc kernels present(in)
  in(:,:,:) = -1._real32
!$acc end kernels

  tb = 0.0_real64 - MPI_Wtime()
!$acc host_data use_device(in, out)
  call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
!$acc end host_data
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
  tb = tb + MPI_Wtime()

!$acc kernels present(in, check)
  local_error = maxval(abs(in - check))
!$acc end kernels

  call report(tf, tb, local_error, nx, ny, nz)

  deallocate(in)
  deallocate(out)
  deallocate(check)

  call plan%destroy()
  call MPI_Finalize(ierr)
#endif
end program test_r2c_3d_float