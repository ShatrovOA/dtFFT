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
program test_r2r_3d
!------------------------------------------------------------------------------------------------
!< This program shows how to use DTFFT with Real-to-Real 3d transform
!< It also tests user-defined 1d communicator
!------------------------------------------------------------------------------------------------
use iso_fortran_env
use iso_c_binding, only: c_loc, c_ptr
use dtfft
use test_utils
#if defined(DTFFT_WITH_CUDA)
use dtfft_interface_cuda_runtime
#include "_dtfft_cuda.h"
#endif
#include "_dtfft_mpi.h"
#include "dtfft.f03"
implicit none
  type(c_ptr) :: check
  real(real64), pointer :: in(:), out(:)
#if defined(DTFFT_WITH_CUDA)
  integer(int32), parameter :: nx = 1024, ny = 1024, nz = 16
#else
  integer(int32), parameter :: nx = 512, ny = 32, nz = 8
#endif
  integer(int32) :: comm_size, comm_rank
  class(dtfft_plan_t), allocatable :: plan
  integer(int32) :: in_counts(3), out_counts(3), ierr
  type(dtfft_r2r_kind_t) :: kinds(3)
  integer(int32) :: iter
  type(dtfft_executor_t) :: executor
  real(real64) :: tf, tb, temp
  integer(int64) :: alloc_size, in_size, out_size
  TYPE_MPI_COMM :: comm
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_stream_t) :: stream
  type(dtfft_platform_t) :: platform
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


  kinds(:) = DTFFT_DCT_2
  executor = DTFFT_EXECUTOR_NONE

#if defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3
#endif

  conf = dtfft_config_t()
  conf%enable_z_slab = .true.
  conf%enable_y_slab = .true.
  conf%enable_mpi_backends = .true.

#if defined(DTFFT_WITH_CUDA)
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
      conf%backend = DTFFT_BACKEND_MPI_P2P
      conf%platform = DTFFT_PLATFORM_CUDA
# if defined( DTFFT_WITH_VKFFT )
      executor = DTFFT_EXECUTOR_VKFFT
# endif
    endif
  endblock
#endif

  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  allocate( dtfft_plan_r2r_t :: plan )
  select type (plan)
  class is ( dtfft_plan_r2r_t )
    call plan%create([nx, ny, nz], kinds, comm=comm, effort=DTFFT_PATIENT, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  endselect
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, out_counts=out_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)

  call MPI_Comm_free(comm, ierr)

  call plan%mem_alloc(alloc_size, in, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%mem_alloc(alloc_size, out, error_code=ierr); DTFFT_CHECK(ierr)

  in_size = product(in_counts)
  out_size = product(out_counts)

  check = mem_alloc_host(in_size * DOUBLE_STORAGE_SIZE)
  call setTestValuesDouble(check, in_size)

#if defined(DTFFT_WITH_CUDA)
  platform = plan%get_platform(error_code=ierr); DTFFT_CHECK(ierr)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    call plan%get_stream(stream, error_code=ierr); DTFFT_CHECK(ierr)
  endif
  call doubleH2D(check, c_loc(in), in_size, platform%val)
#else
  call doubleH2D(check, c_loc(in), in_size)
#endif

  tf = 0.0_real64
  tb = 0.0_real64
  do iter = 1, 1
    temp = 0.0_real64 - MPI_Wtime()
    call plan%execute(in, out, DTFFT_EXECUTE_FORWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
    endif
#endif
    tf = tf + temp + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
    call scaleDouble(executor%val, c_loc(out), out_size, int(8 * nx * ny * nz, int64), platform%val, stream)
#else
    call scaleDouble(executor%val, c_loc(out), out_size, int(8 * nx * ny * nz, int64))
#endif

    temp = 0.0_real64 - MPI_Wtime()
    call plan%execute(out, in, DTFFT_EXECUTE_BACKWARD, error_code=ierr); DTFFT_CHECK(ierr)
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) then
      CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
    endif
#endif
    tb = tb + temp + MPI_Wtime()
  enddo


#if defined(DTFFT_WITH_CUDA)
  call checkAndReportDouble(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check, platform%val)
#else
  call checkAndReportDouble(int(nx * ny * nz, int64), tf, tb, c_loc(in), in_size, check)
#endif

  call plan%mem_free(in)
  call plan%mem_free(out)
  call mem_free_host(check)

  call plan%destroy(error_code=ierr)
  DTFFT_CHECK(ierr)
  deallocate( plan )
  call MPI_Finalize(ierr)
end program test_r2r_3d