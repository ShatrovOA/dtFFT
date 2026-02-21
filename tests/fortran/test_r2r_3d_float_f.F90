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
#if defined (DTFFT_WITH_CUDA) && !defined(__NVCOMPILER) && !defined(DTFFT_WITH_MOCK_ENABLED)
! #error "nvcompiler is required or mock build"
#endif
program test_r2r_3d_float
use iso_fortran_env
use iso_c_binding, only: c_loc, c_ptr, c_f_pointer, c_int
use dtfft
use dtfft_utils
use test_utils
#if defined(DTFFT_WITH_CUDA)
# if defined(__NVCOMPILER)
use cudafor, only: cuda_stream_kind, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize, cudaSuccess, cudaGetErrorString
# elif defined (DTFFT_WITH_MOCK_ENABLED)
use dtfft_interface_cuda_runtime
# endif
#endif
#include "_dtfft_mpi.h"
#include "_dtfft_cuda.h"
#include "dtfft.f03"
implicit none
  type(c_ptr) :: check
  real(real32),  allocatable, target :: r(:), f(:)
  real(real32), pointer :: check_(:)
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  real(real32), managed, allocatable :: r_m(:), f_m(:)
#endif
  integer(int32), parameter :: nx = 512, ny = 64, nz = 32
  integer(int32) :: comm_size, comm_rank, ierr, in_counts(3), in_starts(3), comm_dims(3), out_counts(3), iter
  type(dtfft_executor_t) :: executor
  type(dtfft_plan_r2r_t) :: plan
  real(real64) :: tf, tb
  integer(int64)  :: alloc_size, in_size
  type(dtfft_backend_t) :: backend_to_use
  type(dtfft_backend_t) :: reshape_backend_to_use
  type(dtfft_backend_t) :: actual_backend_used
#if defined(DTFFT_WITH_CUDA)
  type(dtfft_platform_t) :: platform
# if defined(__NVCOMPILER)
  integer(cuda_stream_kind) :: stream
# elif defined (DTFFT_WITH_MOCK_ENABLED)
  type(dtfft_stream_t) :: stream
# endif
#endif
  type(dtfft_config_t) :: conf
  type(dtfft_pencil_t) :: pencil
  TYPE_MPI_COMM :: comm
  integer(int32), pointer :: dims(:) => null()
  type(dtfft_request_t) :: request
  ! interface
  !   subroutine cali_flush(val) bind(C)
  !     import
  !     integer(c_int), intent(in), value :: val
  !   end subroutine cali_flush
  ! end interface


  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, comm_size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, comm_rank, ierr)

  if(comm_rank == 0) then
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a)') "|       dtFFT test: r2r_3d_float       |"
    write(output_unit, '(a)') "----------------------------------------"
    write(output_unit, '(a, i0, a, i0, a, i0)') 'Nx = ',nx, ', Ny = ',ny, ', Nz = ',nz
    write(output_unit, '(a, i0)') 'Number of processors: ', comm_size
  endif

  executor = DTFFT_EXECUTOR_NONE

  call attach_gpu_to_process()

  call dtfft_create_config(conf)

  backend_to_use = DTFFT_BACKEND_MPI_P2P_FUSED
  reshape_backend_to_use = DTFFT_BACKEND_MPI_P2P_FUSED

#if defined(DTFFT_WITH_CUDA)
  block
    character(len=5) :: platform_env
    integer(int32) :: env_len

    call get_environment_variable("DTFFT_PLATFORM", platform_env, env_len)

    if ( env_len == 0 .or. trim(adjustl(platform_env)) == "cuda" ) then
# if defined (DTFFT_WITH_NCCL)
      backend_to_use = DTFFT_BACKEND_NCCL_PIPELINED
# endif

# if defined(__NVCOMPILER)
      CUDA_CALL( cudaStreamCreate(stream) )
      conf%stream = dtfft_stream_t(stream)
# elif defined (DTFFT_WITH_MOCK_ENABLED)
      CUDA_CALL( cudaStreamCreate(stream) )
      conf%stream = stream
# else
      if ( comm_rank == 0 ) &
        write(output_unit, '(a)') "This test requires NVFortran or MOCK build in order to run on GPU"
      call MPI_Finalize(ierr)
      stop
# endif
    endif
  endblock
#endif

  conf%backend = backend_to_use
  conf%reshape_backend = reshape_backend_to_use
  conf%enable_y_slab = .true.
! #ifdef DTFFT_WITH_COMPRESSION
!   conf%backend = DTFFT_BACKEND_ADAPTIVE
!   conf%reshape_backend = DTFFT_BACKEND_ADAPTIVE
!   conf%compression_config_transpose%compression_mode = DTFFT_COMPRESSION_MODE_FIXED_ACCURACY
!   conf%compression_config_transpose%tolerance = 1.e-6_real64
!   conf%compression_config_reshape%compression_mode = DTFFT_COMPRESSION_MODE_FIXED_PRECISION
!   conf%compression_config_reshape%precision = 20
! #endif


  call dtfft_set_config(conf, error_code=ierr); DTFFT_CHECK(ierr)
  do iter = 2, 4
    comm_dims(:) = 0
    if ( iter < 4 ) comm_dims(iter) = 1
    call createGridDims(3, [nx, ny, nz], comm_dims, in_starts, in_counts)
    pencil = dtfft_pencil_t(in_starts, in_counts)

    comm = MPI_COMM_WORLD
    if ( iter == 4 ) then
      block
        logical :: periods(3)
        integer(int32) :: temp_dims(3)

        temp_dims(:) = comm_dims(:)
        ! Would like that number of processes in X direction be redistributed into Y direction
        temp_dims(2) = temp_dims(2) * temp_dims(1)
        temp_dims(1) = 1
        periods = .false.
        call MPI_Cart_create(MPI_COMM_WORLD, 3, temp_dims, periods, .false., comm, ierr)
      endblock
    endif

  call plan%create(pencil, comm=comm, effort=DTFFT_ESTIMATE, precision=DTFFT_SINGLE, executor=executor, error_code=ierr); DTFFT_CHECK(ierr)
  alloc_size = plan%get_alloc_size(error_code=ierr); DTFFT_CHECK(ierr)
  call plan%get_local_sizes(out_counts=out_counts, error_code=ierr); DTFFT_CHECK(ierr)
  call plan%report(error_code=ierr); DTFFT_CHECK(ierr)

  actual_backend_used = plan%get_backend(error_code=ierr);  DTFFT_CHECK(ierr)
  if(comm_rank == 0) then
    write(output_unit, '(a)') "Using backend: "//dtfft_get_backend_string(actual_backend_used)
  endif
  if ( comm_dims(1) > 1 ) then
    actual_backend_used = plan%get_reshape_backend(error_code=ierr);  DTFFT_CHECK(ierr)
    if(comm_rank == 0) then
      write(output_unit, '(a)') "Using reshape backend: "//dtfft_get_backend_string(actual_backend_used)
    endif
  endif

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  platform = plan%get_platform()
#endif

  in_size = product(in_counts)
  allocate(r(alloc_size), f(alloc_size))
  check = mem_alloc_host(alloc_size * FLOAT_STORAGE_SIZE)
  call setTestValuesFloat(check, in_size)

  call c_f_pointer(check, check_, [in_size])

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  allocate(r_m(alloc_size), f_m(alloc_size))
  r_m(:) = check_(:)
#else
  r(:in_size) = check_(:)
#endif

  tf = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  if ( comm_dims(1) > 1 ) then
    call plan%reshape(r_m, f_m, DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(f_m, r_m, DTFFT_TRANSPOSE_X_TO_Y, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(r_m, f_m, DTFFT_TRANSPOSE_Y_TO_Z, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%reshape(f_m, r_m, DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS, error_code=ierr);  DTFFT_CHECK(ierr)
  else
    call plan%transpose(r_m, f_m, DTFFT_TRANSPOSE_X_TO_Y, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(f_m, r_m, DTFFT_TRANSPOSE_Y_TO_Z, error_code=ierr);  DTFFT_CHECK(ierr)
  endif
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaStreamSynchronize(stream) )
  endif
#else
  if ( comm_dims(1) > 1 ) then
    call plan%reshape(r, f, DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(f, r, DTFFT_TRANSPOSE_X_TO_Y, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(r, f, DTFFT_TRANSPOSE_Y_TO_Z, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%reshape(f, r, DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS, error_code=ierr);  DTFFT_CHECK(ierr)
  else
    call plan%transpose(r, f, DTFFT_TRANSPOSE_X_TO_Y, error_code=ierr);  DTFFT_CHECK(ierr)
    call plan%transpose(f, r, DTFFT_TRANSPOSE_Y_TO_Z, error_code=ierr);  DTFFT_CHECK(ierr)
  endif
#endif
  DTFFT_CHECK(ierr)
  tf = tf + MPI_Wtime()

  tb = 0.0_real64 - MPI_Wtime()
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  if ( comm_dims(1) > 1 ) then
    call plan%reshape(r_m, f_m, DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose(f_m, r_m, DTFFT_TRANSPOSE_Z_TO_Y, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose(r_m, f_m, DTFFT_TRANSPOSE_Y_TO_X, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%reshape(f_m, r_m, DTFFT_RESHAPE_X_PENCILS_TO_BRICKS, error_code=ierr); DTFFT_CHECK(ierr)
  else
    call plan%transpose(r_m, f_m, DTFFT_TRANSPOSE_Z_TO_Y, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose(f_m, r_m, DTFFT_TRANSPOSE_Y_TO_X, error_code=ierr); DTFFT_CHECK(ierr)
  endif
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaStreamSynchronize(stream) )
  endif
  r(:) = r_m(:)
#else
  if ( comm_dims(1) > 1 ) then
    request = plan%reshape_start(r, f, DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%reshape_end(request, error_code=ierr); DTFFT_CHECK(ierr)
    request = plan%transpose_start(f, r, DTFFT_TRANSPOSE_Z_TO_Y, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose_end(request, error_code=ierr); DTFFT_CHECK(ierr)
    request = plan%transpose_start(r, f, DTFFT_TRANSPOSE_Y_TO_X, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose_end(request, error_code=ierr); DTFFT_CHECK(ierr)
    request = plan%reshape_start(f, r, DTFFT_RESHAPE_X_PENCILS_TO_BRICKS, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%reshape_end(request, error_code=ierr); DTFFT_CHECK(ierr)
  else
    call plan%transpose(r, f, DTFFT_TRANSPOSE_Z_TO_Y, error_code=ierr); DTFFT_CHECK(ierr)
    call plan%transpose(f, r, DTFFT_TRANSPOSE_Y_TO_X, error_code=ierr); DTFFT_CHECK(ierr)
  endif
#endif

  DTFFT_CHECK(ierr)
  tb = tb + MPI_Wtime()

  call plan%get_dims(dims, error_code=ierr); DTFFT_CHECK(ierr)

#ifdef DTFFT_WITH_COMPRESSION
  call plan%report_compression()
#endif

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  call checkAndReportFloat(int(product(dims), int64), tf, tb, c_loc(r_m), in_size, check, DTFFT_PLATFORM_HOST%val)
#else
  call checkAndReportFloat(int(product(dims), int64), tf, tb, c_loc(r), in_size, check)
#endif

  deallocate(r, f)
  call mem_free_host(check)
  nullify(check_)

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  deallocate(r_m, f_m)
#endif

  call plan%destroy()

  if ( iter == 4 ) then
    call MPI_Comm_free(comm, ierr)
  endif

  enddo

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  if ( platform == DTFFT_PLATFORM_CUDA ) then
    CUDA_CALL( cudaStreamDestroy(stream) )
  endif
#endif

  call MPI_Finalize(ierr)
end program test_r2r_3d_float