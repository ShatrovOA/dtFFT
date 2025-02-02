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
module dtfft_api
!! This module is a Fortran part of C/C++ interface
use iso_c_binding,    only: c_int8_t, c_int32_t, c_size_t, &
                            c_float, c_bool, c_char,        &
                            c_null_ptr, c_ptr, c_loc,       &
                            c_f_pointer, c_associated
use iso_fortran_env,  only: int8, int32
use dtfft_parameters
use dtfft_pencil,     only: dtfft_pencil_t
use dtfft_plan
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use cudafor,          only: cuda_stream_kind
#endif
#include "dtfft_cuda.h"
#include "dtfft_mpi.h"
implicit none
private

#define CHECK_PLAN_CREATED(plan)                \
  if(.not.c_associated(plan)) then;             \
    error_code = DTFFT_ERROR_PLAN_NOT_CREATED;  \
    return;                                     \
  endif

  type :: dtfft_plan_c
  !! C pointer to Fortran plan
    class(dtfft_abstract_plan),  allocatable :: p                         !< Actual Fortran plan
  end type dtfft_plan_c

contains

  pure TYPE_MPI_COMM function get_comm(c_comm)
    integer(c_int32_t),  intent(in) :: c_comm

    DTFFT_GET_MPI_VALUE(get_comm) = c_comm
  end function get_comm

  function dtfft_create_plan_r2r_c(ndims, dims, kinds, comm, precision, effort_flag, executor_type, plan_ptr)       &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates R2R dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
    integer(c_int8_t),  intent(in)                :: ndims                !< Rank of transform. Can be 2 or 3.
    type(c_ptr),                        value     :: dims                 !< Global sizes of transform
    type(c_ptr),                        value     :: kinds                !< FFT R2R kinds
    integer(c_int32_t),                 value     :: comm                 !< Communicator
    integer(c_int8_t),  intent(in)                :: precision            !< Precision of transform
    integer(c_int8_t),  intent(in)                :: effort_flag          !< DTFFT planner effort flag
    integer(c_int8_t),  intent(in)                :: executor_type        !< Type of External FFT Executor
    type(c_ptr),        intent(out)               :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !< Fortran dims
    integer(int8),                      pointer   :: fkinds(:)            !< Fortran R2R kinds
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_r2r :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])
    call c_f_pointer(kinds, fkinds, [ndims])

    select type( p => plan%p )
    type is ( dtfft_plan_r2r )
      call p%create(fdims, fkinds, get_comm(comm), precision, effort_flag, executor_type, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2r_c

  function dtfft_create_plan_c2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr)              &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates C2C dtFFT Plan, allocates all structures and prepares FFT, C/C++ interface
    integer(c_int8_t),  intent(in)                :: ndims                !< Rank of transform. Can be 2 or 3.
    type(c_ptr),                        value     :: dims                 !< Global sizes of transform
    integer(c_int32_t),                 value     :: comm                 !< Communicator
    integer(c_int8_t),  intent(in)                :: precision            !< Precision of transform
    integer(c_int8_t),  intent(in)                :: effort_flag          !< DTFFT planner effort flag
    integer(c_int8_t),  intent(in)                :: executor_type        !< Type of External FFT Executor
    type(c_ptr),        intent(out)               :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !< Fortran dims
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_c2c :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_c2c)
      call p%create(fdims, get_comm(comm), precision, effort_flag, executor_type, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_c2c_c

#ifndef DTFFT_TRANSPOSE_ONLY
  function dtfft_create_plan_r2c_c(ndims, dims, comm, precision, effort_flag, executor_type, plan_ptr)              &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates R2C dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
    integer(c_int8_t),  intent(in)                :: ndims                !< Rank of transform. Can be 2 or 3.
    type(c_ptr),                        value     :: dims                 !< Global sizes of transform
    integer(c_int32_t),                 value     :: comm                 !< Communicator
    integer(c_int8_t),  intent(in)                :: precision            !< Precision of transform
    integer(c_int8_t),  intent(in)                :: effort_flag          !< DTFFT planner effort flag
    integer(c_int8_t),  intent(in)                :: executor_type        !< Type of External FFT Executor
    type(c_ptr),        intent(out)               :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !< Fortran dims
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_r2c :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_r2c)
      call p%create(fdims, get_comm(comm), precision, effort_flag, executor_type, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2c_c
#endif

  function dtfft_get_z_slab_c(plan_ptr, is_zlab)                                                                    &
    result(error_code)                                                                                              &
    bind(C)
  !! Checks if dtFFT Plan is using Z-slab optimization
    type(c_ptr),        intent(in),     value     :: plan_ptr             !< C pointer to Fortran plan
    logical(c_bool),    intent(out)               :: is_zlab              !< Is plan internally using Z-slab optimization
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    is_zlab = plan%p%get_z_slab(error_code)
  end function dtfft_get_z_slab_c

  function dtfft_execute_c(plan_ptr, in, out, execute_type, aux)                                                  &
    result(error_code)                                                                                              &
    bind(C)
  !! Executes dtFFT Plan, C/C++/Python interface. `Aux` can be NULL. If `in` or `out` are NULL, bad things will happen.
    type(c_ptr),        intent(in),     value     :: plan_ptr             !< C pointer to Fortran plan
    real(c_float),  DEVICE_PTR    intent(inout)   :: in(*)                !< Incomming buffer, not NULL
    real(c_float),  DEVICE_PTR    intent(inout)   :: out(*)               !< Outgoing buffer
    integer(c_int8_t),  intent(in)                :: execute_type       !< Type of execution,
                                                                          !< one of ``DTFFT_TRANSPOSE_OUT``, ``DTFFT_TRANSPOSE_IN``
    real(c_float),      intent(inout),  optional  :: aux(*)               !< Aux buffer, can be NULL
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%execute(in, out, execute_type, aux, error_code)
  end function dtfft_execute_c

  function dtfft_transpose_c(plan_ptr, in, out, transpose_type)                                                     &
    result(error_code)                                                                                              &
    bind(C)
  !! Executes single transposition, C interface.
    type(c_ptr),        intent(in),     value     :: plan_ptr             !< C pointer to Fortran plan
    real(c_float),      DEVICE_PTR  intent(inout) :: in(*)                !< Incomming buffer, not NULL
    real(c_float),      DEVICE_PTR  intent(inout) :: out(*)               !< Outgoing buffer, not NULL
    integer(c_int8_t),  intent(in)                :: transpose_type       !< Type of transposition. One of the:
                                                                          !< - `DTFFT_TRANSPOSE_X_TO_Y`
                                                                          !< - `DTFFT_TRANSPOSE_Y_TO_X`
                                                                          !< - `DTFFT_TRANSPOSE_Y_TO_Z` (only for 3d plan)
                                                                          !< - `DTFFT_TRANSPOSE_Z_TO_Y` (only for 3d plan)
                                                                          !< - `DTFFT_TRANSPOSE_X_TO_Z` (only 3D and slab decomposition in Z direction)
                                                                          !< - `DTFFT_TRANSPOSE_Z_TO_X` (only 3D and slab decomposition in Z direction)
                                                                          !<
                                                                          !< [//]: # (ListBreak)
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%transpose(in, out, transpose_type, error_code)
  end function dtfft_transpose_c

  function dtfft_destroy_c(plan_ptr)                                                                                &
    result(error_code)                                                                                              &
    bind(C)
  !! Destroys dtFFT Plan, C/C++ interface
    type(c_ptr)                                   :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%destroy(error_code)
    deallocate( plan%p )
    deallocate( plan )
    plan_ptr = c_null_ptr
  end function dtfft_destroy_c

  function dtfft_get_local_sizes_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size)              &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns local sizes, counts in real and Fourier spaces and number of elements to be allocated for `in` and `out` buffers,
  !! C/C++ interface. It is necessary for R2C transform  to allocate `out` buffer with `alloc_size` number of elements.
  !! Total number of bytes to be allocated :
  !!    - C2C: 2 * alloc_size * sizeof(double/float)
  !!    - R2R: alloc_size * sizeof(double/float)
  !!    - R2C: alloc_size * sizeof(double/float)
    type(c_ptr),                         value    :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int32_t),  intent(out),    optional :: in_starts(3)         !< Starts of local portion of data in 'real' space
    integer(c_int32_t),  intent(out),    optional :: in_counts(3)         !< Counts of local portion of data in 'real' space
    integer(c_int32_t),  intent(out),    optional :: out_starts(3)        !< Starts of local portion of data in 'fourier' space
    integer(c_int32_t),  intent(out),    optional :: out_counts(3)        !< Counts of local portion of data in 'fourier' space
    integer(c_size_t),   intent(out),    optional :: alloc_size           !< Minimum data needs to be allocated
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    call plan%p%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size, error_code)
  end function dtfft_get_local_sizes_c

  subroutine dtfft_get_error_string_c(error_code, error_string, error_string_size) bind(C)
  !! Returns an explaination of ``error_code`` that could have been previously returned by one of dtFFT API calls
    integer(c_int32_t),  intent(in)               :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    character(c_char),   intent(out)              :: error_string(*)      !< Explanation of error
    integer(c_size_t),   intent(out)              :: error_string_size    !< Size of ``error_string``

    call string_f2c(dtfft_get_error_string(error_code), error_string, error_string_size)
  end subroutine dtfft_get_error_string_c

  function dtfft_get_pencil_c(plan_ptr, dim, pencil)                                                                &
    result(error_code)                                                                                              &
    bind(C)
    type(c_ptr),                         value    :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int8_t),  intent(in)                :: dim                  !< Dimension requested
    type(dtfft_pencil_t)                          :: pencil               !< Pencil pointer
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    pencil = plan%p%get_pencil(dim, error_code)
  end function dtfft_get_pencil_c

#ifdef DTFFT_WITH_CUDA
  function dtfft_set_stream_c(stream)                                                                               &
    result(error_code)                                                                                              &
    bind(C)
  !! Sets CUDA stream that should be used in dtFFT
  !! In order for this call to take effect, must be called before creation of plan
    integer(cuda_stream_kind),        intent(in)  :: stream               !< CUDA stream
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.

    call dtfft_set_stream(stream, error_code)
  end function dtfft_set_stream_c

  function dtfft_set_gpu_backend_c(backend_id)                                                                      &
    result(error_code)                                                                                              &
    bind(C)
  !! Sets backend that should be used when ``effort_flag`` parameter of create subroutine is ``DTFFT_ESTIMATE`` or ``DTFFT_MEASURE``.
  !! In order for this call to take effect, must be called before creation of plan
    integer(c_int8_t),                intent(in)  :: backend_id           !< The enumerated type dtfft_gpu_backend_t
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.

    call dtfft_set_gpu_backend(backend_id, error_code)
  end function dtfft_set_gpu_backend_c

  function dtfft_get_stream_c(plan_ptr, stream)                                                                     &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns CUDA stream associated with plan
    type(c_ptr),                         value    :: plan_ptr             !< C pointer to Fortran plan
    integer(cuda_stream_kind),  intent(out)       :: stream               !< CUDA stream
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    stream = plan%p%get_stream(error_code)
  end function dtfft_get_stream_c

  function dtfft_get_gpu_backend_c(plan_ptr, backend_id)                                                             &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns selected dtfft_gpu_backend_t during autotuning
    type(c_ptr),                         value    :: plan_ptr             !< C pointer to Fortran plan
    integer(c_int8_t),          intent(out)       :: backend_id           !< The enumerated type dtfft_gpu_backend_t
    integer(c_int32_t)                            :: error_code           !< The enumerated type dtfft_error_code_t
                                                                          !< defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !< Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr)
    call c_f_pointer(plan_ptr, plan)
    backend_id = plan%p%get_gpu_backend(error_code)
  end function dtfft_get_gpu_backend_c

  subroutine dtfft_get_gpu_backend_string_c(backend_id, backend_string, backend_string_size) bind(C)
  !! Returns string representation of ``dtfft_gpu_backend_t``
    integer(c_int8_t),   intent(in)               :: backend_id           !< The enumerated type dtfft_gpu_backend_t
    character(c_char),   intent(out)              :: backend_string(*)    !< Resulting string
    integer(c_size_t),   intent(out)              :: backend_string_size  !< Size of string

    call string_f2c(dtfft_get_gpu_backend_string(backend_id), backend_string, backend_string_size)
  end subroutine dtfft_get_gpu_backend_string_c
#endif
end module dtfft_api