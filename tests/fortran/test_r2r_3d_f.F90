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
program test_r2r_3d
!------------------------------------------------------------------------------------------------
!< This program shows how to use DTFFT with Real-to-Real 3d transform
!< It also tests user-defined 1d communicator
!------------------------------------------------------------------------------------------------
use iso_fortran_env
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor
use dtfft_utils
#include "dtfft_cuda.h"
#endif
#include "dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  real(real64), allocatable :: in(:), out(:), check(:)
  real(real64) :: local_error, rnd
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(int32), parameter :: nx = 1024, ny = 1024, nz = 16
#else
  integer(int32), parameter :: nx = 512, ny = 32, nz = 8
#endif
  integer(int32) :: comm_size, comm_rank, i, j, k, out_size, in_size
  class(dtfft_plan_t), allocatable :: plan
  integer(int32) :: in_starts(3), in_counts(3), out_counts(3), ierr, ijk
  integer(int32) :: iter
  type(dtfft_executor_t) :: executor
  real(real64) :: tf, tb, temp
  integer(int64) :: alloc_size
  TYPE_MPI_COMM :: comm
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: stream
#endif
  type(dtfft_config_t) :: conf

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

  call attach_gpu_to_process()

  call MPI_Cart_create(MPI_COMM_WORLD, 1, [comm_size], [.false.], .false., comm, ierr)

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#elif defined(DTFFT_WITH_VKFFT)
  executor = DTFFT_EXECUTOR_VKFFT
#else
  executor = DTFFT_EXECUTOR_NONE
#endif

  conf = dtfft_config_t()
  conf%enable_z_slab = .false.
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  conf%backend = DTFFT_BACKEND_MPI_P2P
  conf%platform = DTFFT_PLATFORM_CUDA
#endif

  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  allocate( dtfft_plan_r2r_t :: plan )
  select type (plan)
  class is ( dtfft_plan_r2r_t )
    call plan%create([nx, ny, nz], [DTFFT_DCT_2, DTFFT_DCT_2, DTFFT_DCT_2], comm=comm, effort=DTFFT_ESTIMATE, executor=executor, error_code=ierr)
  endselect
  DTFFT_CHECK(ierr)

  call plan%get_local_sizes(in_starts, in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr)
  DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%get_stream(stream, error_code=ierr)
  DTFFT_CHECK(ierr)
#endif

  allocate(in(alloc_size))
  allocate(check(alloc_size))
  allocate(out(alloc_size))

!$acc enter data create(out)
  in_size = product(in_counts)
  out_size = product(out_counts)

  do k = 0, in_counts(3) - 1
    do j = 0, in_counts(2) - 1
      do i = 0, in_counts(1) - 1
        call random_number(rnd)
        ijk = (k * in_counts(2) + j) * in_counts(1) + i + 1
        in(ijk) = rnd
        ! in(ijk) = real(1000 * comm_rank + ijk, real64)
        check(ijk) = in(ijk)
      enddo
    enddo
  enddo

!$acc enter data copyin(in, check)

  tf = 0.0_real64
  tb = 0.0_real64
  do iter = 1, 10
    temp = 0.0_real64 - MPI_Wtime()
  !$acc host_data use_device(in, out)
    call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr)
  !$acc end host_data
    DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    tf = tf + temp + MPI_Wtime()

    if ( executor /= DTFFT_EXECUTOR_NONE ) then
    !$acc kernels present(out)
      out(:out_size) = out(:out_size) / real(8 * nx * ny * nz, real64)
    !$acc end kernels
    endif

  !$acc kernels present(in)
    in(:) = -1._real64
  !$acc end kernels

    temp = 0.0_real64 - MPI_Wtime()
  !$acc host_data use_device(in, out)
    call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
  !$acc end host_data
    DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
#endif
    tb = tb + temp + MPI_Wtime()
  enddo

!$acc kernels present(in, check)
  local_error = maxval(abs(in(:in_size) - check(:in_size)))
!$acc end kernels

  call report(tf, tb, local_error, nx, ny, nz)

  deallocate(in, out, check)
  call plan%destroy(error_code=ierr)
  DTFFT_CHECK(ierr)
  deallocate( plan )
  call MPI_Finalize(ierr)
end program test_r2r_3d