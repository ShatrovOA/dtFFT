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
program test_r2r_3d_float
use iso_fortran_env
use iso_c_binding, only: c_loc, c_ptr, c_f_pointer
use dtfft
use dtfft_utils
use test_utils
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
use cudafor, only: cuda_stream_kind, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, cudaSuccess, cudaGetErrorString
#endif
#include "dtfft_mpi.h"
#include "dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  type(c_ptr) :: check
  real(real32),  allocatable, target :: inout(:)
  real(real32), pointer :: check_(:)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  real(real32), managed, allocatable :: inout_m(:)
#endif
  integer(int32), parameter :: nx = 512, ny = 64, nz = 16
  integer(int32) :: comm_size, comm_rank, ierr, in_counts(3)
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2r_t) :: plan
  real(real64) :: tf, tb
  integer(int64)  :: alloc_size, in_size
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_backend_t) :: backend_to_use
  type(dtfft_platform_t) :: platform
# if defined(__NVCOMPILER)
  type(dtfft_backend_t) :: actual_backend_used
  integer(cuda_stream_kind) :: stream
# endif
#endif
  type(dtfft_config_t) :: conf

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       DTFFT test: r2r_3d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
  endif

  executor = DTFFT_EXECUTOR_NONE

  call attach_gpu_to_process()

  call dtfft_create_config(conf)

#if defined(DTFFT_WITH_CUDA)
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
      backend_to_use = DTFFT_BACKEND_MPI_A2A
# if defined (DTFFT_WITH_NCCL)
      backend_to_use = DTFFT_BACKEND_NCCL_PIPELINED
# endif
      conf%backend = backend_to_use

# if defined(__NVCOMPILER)
      CUDA_CALL( "cudaStreamCreate", cudaStreamCreate(stream) )
      conf%stream = dtfft_stream_t(stream)
# else
      if ( comm_rank == 0 ) &
        write(output_unit, '(a)') "This test requires NVFortran in order to run on GPU"
      call MPI_Finalize(ierr)
      stop
# endif
    endif
  endblock
#endif

  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)

  call plan%create([nx, ny, nz], precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(in_counts=in_counts, alloc_size=alloc_size, error_code=ierr); DTFFT_CHECK(ierr)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  actual_backend_used = plan%get_backend(error_code=ierr);  DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Using backend: "//dtfft_get_backend_string(actual_backend_used)
  endif
  platform = plan%get_platform()
  if ( platform == DTFFT_PLATFORM_CUDA .and. comm_size > 1 .and. actual_backend_used /= backend_to_use ) then
    error stop "Invalid backend: actual_backend_used /= backend_to_use"
  endif
#endif

  in_size = product(in_counts)
  allocate(inout(alloc_size))
  call mem_alloc_host(in_size * FLOAT_STORAGE_SIZE, check)
  call setTestValuesFloat(check, in_size)

  call c_f_pointer(check, check_, [in_size])

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  allocate(inout_m(alloc_size))
  inout_m(:in_size) = check_(:)
#else
  inout(:in_size) = check_(:)
#endif

  tf = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%execute(inout_m, inout_m, DTFFT_EXECUTE_FORWARD, error_code=ierr)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  endif
#else
  call plan%execute(inout, inout, DTFFT_EXECUTE_FORWARD, error_code=ierr)
#endif
  DTFFT_CHECK(ierr)
  tf = tf + MPI_Wtime()

  tb = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call plan%execute(inout_m, inout_m, DTFFT_EXECUTE_BACKWARD, error_code=ierr)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaStreamSynchronize", cudaStreamSynchronize(stream) )
  endif
  inout(:) = inout_m(:)
#else
  call plan%execute(inout, inout, DTFFT_EXECUTE_BACKWARD)
#endif
  DTFFT_CHECK(ierr)
  tb = tb + MPI_Wtime()

#if defined(DTFFT_WITH_CUDA)
  call checkAndReportFloat(int(nx * ny * nz, int64), tf, tb, c_loc(inout), in_size, check, DTFFT_PLATFORM_HOST%val)
#else
  call checkAndReportFloat(int(nx * ny * nz, int64), tf, tb, c_loc(inout), in_size, check)
#endif

  deallocate(inout)
  nullify(check_)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  deallocate(inout_m)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( "cudaStreamDestroy", cudaStreamDestroy(stream) )
  endif
#endif

  call plan%destroy()
  call MPI_Finalize(ierr)
end program test_r2r_3d_float