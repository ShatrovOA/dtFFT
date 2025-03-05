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
program test_c2c_3d
use iso_fortran_env, only: R8P => real64, I4P => int32, I8P => int64, I1P => int8, output_unit, error_unit, int32
use dtfft
use iso_c_binding
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
use dtfft_utils
#include "dtfft_cuda.h"
#endif
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  complex(R8P),  allocatable :: inout(:), check(:,:,:), aux(:)
  real(R8P) :: err, local_error, rnd1, rnd2
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(I4P), parameter :: nx = 2011, ny = 111, nz = 755
#else
  integer(I4P), parameter :: nx = 129, ny = 123, nz = 33
#endif
  integer(I4P) :: comm_size, comm_rank, i, j, k, ierr, ii, jj, kk, idx
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_c2c_t) :: plan
  integer(I4P) :: in_counts(3), out_counts(3), iter
  integer(I8P)  :: alloc_size
  real(R8P) :: ts, tf, tb
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: stream
  type(dtfft_gpu_backend_t) :: selected_backend
#endif
  type(dtfft_config_t) :: conf

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

#if defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined (DTFFT_WITH_CUFFT)
  executor = DTFFT_EXECUTOR_CUFFT
#else
  executor = DTFFT_EXECUTOR_NONE
#endif

  conf = dtfft_config_t()

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
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

    conf%gpu_backend = DTFFT_GPU_BACKEND_MPI_A2A
    conf%platform = DTFFT_PLATFORM_CUDA
  endblock
#endif

  call dtfft_set_config(conf)
  ! Setting effort=DTFFT_PATIENT will ignore value of `conf%gpu_backend` and will run autotune to find best backend
  ! Fastest backend will be selected
  call plan%create([nx, ny, nz], executor=executor, effort=DTFFT_PATIENT, error_code=ierr)
  DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, error_code=ierr)
  DTFFT_CHECK(ierr)
  alloc_size = plan%get_alloc_size(ierr)
  DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%get_stream(stream, error_code=ierr); DTFFT_CHECK(ierr)
  selected_backend = plan%get_gpu_backend(error_code=ierr); DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Selected backend: '"//dtfft_get_gpu_backend_string(selected_backend)//"'"
  endif
#endif

  allocate(inout(alloc_size))
  allocate(aux(alloc_size))
  allocate(check(in_counts(1), in_counts(2), in_counts(3)))

  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        call random_number(rnd1)
        call random_number(rnd2)
        ii = i - 1; jj = j - 1; kk = k - 1
        idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
        inout(idx) = cmplx(rnd1, rnd1, R8P)
        check(i,j,k) = inout(idx)
      enddo
    enddo
  enddo
!$acc enter data copyin(inout, check) create(aux)

  tf = 0.0_R8P
  tb = 0.0_R8P
  do iter = 1, 3
    ts = - MPI_Wtime()
!$acc host_data use_device(inout, aux)
    call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, aux,  error_code=ierr)
!$acc end host_data
    DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif

    tf = tf + ts + MPI_Wtime()

    if ( executor /= DTFFT_EXECUTOR_NONE ) then
    !$acc kernels present(inout)
      inout(:) = inout(:) / real(nx * ny * nz, R8P)
    !$acc end kernels
    endif
    ts = 0.0_R8P - MPI_Wtime()
  !$acc host_data use_device(inout, aux)
    call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD, aux, error_code=ierr)
  !$acc end host_data
    DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    tb = tb + ts + MPI_Wtime()
  enddo

  local_error = 0._R8P
!$acc parallel loop collapse(3) present(inout, check) copyin(in_counts) reduction(max:local_error)
  do k = 1, in_counts(3)
    do j = 1, in_counts(2)
      do i = 1, in_counts(1)
        ii = i - 1; jj = j - 1; kk = k - 1
        idx = kk * in_counts(2) * in_counts(1) + jj * in_counts(1) + ii + 1
        err = abs(inout(idx) - check(i,j,k))
        if ( err > local_error ) local_error = err
      enddo
    enddo
  enddo

  call report(tf, tb, local_error, nx, ny, nz)

!$acc exit data delete(inout, aux)

  deallocate(inout, check)

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_c2c_3d