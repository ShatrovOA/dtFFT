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
                            c_f_pointer
use iso_fortran_env,  only: int8, int32
use dtfft_config
use dtfft_parameters
use dtfft_pencil,     only: dtfft_pencil_t
use dtfft_plan
use dtfft_utils
#include "dtfft_mpi.h"
implicit none
private

#define CHECK_PLAN_CREATED(c_plan, f_plan)      \
  if(is_null_ptr(c_plan)) then;                 \
    error_code = DTFFT_ERROR_PLAN_NOT_CREATED;  \
    return;                                     \
  endif;                                        \
  call c_f_pointer(c_plan, f_plan)

  type :: dtfft_plan_c
  !! C pointer to Fortran plan
    class(dtfft_plan_t),  allocatable :: p
      !! Actual Fortran plan
  end type dtfft_plan_c

  type, bind(C) :: dtfft_pencil_c
  !! Structure to hold pencil decomposition info
    integer(c_int8_t)   :: dim        !! Aligned dimension id
    integer(c_int8_t)   :: ndims      !! Number of dimensions
    integer(c_int32_t)  :: starts(3)  !! Local starts, starting from 0 for both C and Fortran
    integer(c_int32_t)  :: counts(3)  !! Local counts of data, in elements
    integer(c_size_t)   :: size       !! Total number of elements in a pencil
  end type dtfft_pencil_c

contains

  pure TYPE_MPI_COMM function get_comm(c_comm)
    integer(c_int32_t),  intent(in) :: c_comm

    DTFFT_GET_MPI_VALUE(get_comm) = c_comm
  end function get_comm

  function dtfft_create_plan_r2r_c(ndims, dims, kinds, comm, precision, effort, executor, plan_ptr)                 &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates R2R dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
    integer(c_int8_t),              intent(in)    :: ndims                !! Rank of transform. Can be 2 or 3.
    type(c_ptr),            value,  intent(in)    :: dims                 !! Global sizes of transform
    type(c_ptr),            value,  intent(in)    :: kinds                !! FFT R2R kinds
    integer(c_int32_t),     value,  intent(in)    :: comm                 !! Communicator
    type(dtfft_precision_t),        intent(in)    :: precision            !! Precision of transform
    type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner effort type
    type(dtfft_executor_t),         intent(in)    :: executor             !! Type of External FFT Executor
    type(c_ptr),                    intent(out)   :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !! Fortran dims
    type(dtfft_r2r_kind_t),             pointer   :: fkinds(:)            !! Fortran R2R kinds
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_r2r_t :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])
    call c_f_pointer(kinds, fkinds, [ndims])

    select type( p => plan%p )
    type is ( dtfft_plan_r2r_t )
      call p%create(fdims, fkinds, get_comm(comm), precision, effort, executor, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2r_c

  function dtfft_create_plan_c2c_c(ndims, dims, comm, precision, effort, executor, plan_ptr)                        &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates C2C dtFFT Plan, allocates all structures and prepares FFT, C/C++ interface
    integer(c_int8_t),              intent(in)    :: ndims                !! Rank of transform. Can be 2 or 3.
    type(c_ptr),            value,  intent(in)    :: dims                 !! Global sizes of transform
    integer(c_int32_t),     value,  intent(in)    :: comm                 !! Communicator
    type(dtfft_precision_t),        intent(in)    :: precision            !! Precision of transform
    type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner effort type
    type(dtfft_executor_t),         intent(in)    :: executor             !! Type of External FFT Executor
    type(c_ptr),                    intent(out)   :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !! Fortran dims
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_c2c_t :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_c2c_t)
      call p%create(fdims, get_comm(comm), precision, effort, executor, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_c2c_c

#ifndef DTFFT_TRANSPOSE_ONLY
  function dtfft_create_plan_r2c_c(ndims, dims, comm, precision, effort, executor, plan_ptr)                        &
    result(error_code)                                                                                              &
    bind(C)
  !! Creates R2C dtFFT Plan, allocates all structures and prepares FFT, C/C++/Python interface
    integer(c_int8_t),              intent(in)    :: ndims                !! Rank of transform. Can be 2 or 3.
    type(c_ptr),            value,  intent(in)    :: dims                 !! Global sizes of transform
    integer(c_int32_t),     value,  intent(in)    :: comm                 !! Communicator
    type(dtfft_precision_t),        intent(in)    :: precision            !! Precision of transform
    type(dtfft_effort_t),           intent(in)    :: effort               !! ``dtFFT`` planner effort type
    type(dtfft_executor_t),         intent(in)    :: executor             !! Type of External FFT Executor
    type(c_ptr),                    intent(out)   :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    integer(int32),                     pointer   :: fdims(:)             !! Fortran dims
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    allocate(plan)
    allocate( dtfft_plan_r2c_t :: plan%p )

    call c_f_pointer(dims, fdims, [ndims])

    select type(p => plan%p)
    class is (dtfft_plan_r2c_t)
      call p%create(fdims, get_comm(comm), precision, effort, executor, error_code)
    endselect
    plan_ptr = c_loc(plan)
  end function dtfft_create_plan_r2c_c
#endif

  function dtfft_get_z_slab_enabled_c(plan_ptr, is_z_slab_enabled)                                                  &
    result(error_code)                                                                                              &
    bind(C)
  !! Checks if dtFFT Plan is using Z-slab optimization
    type(c_ptr),        intent(in),     value     :: plan_ptr             !! C pointer to Fortran plan
    logical(c_bool),    intent(out)               :: is_z_slab_enabled    !! Is plan internally using Z-slab optimization
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    is_z_slab_enabled = plan%p%get_z_slab_enabled(error_code)
  end function dtfft_get_z_slab_enabled_c

  function dtfft_execute_c(plan_ptr, in, out, execute_type, aux)                                                    &
    result(error_code)                                                                                              &
    bind(C)
  !! Executes dtFFT Plan, C/C++ interface. `aux` can be NULL.
    type(c_ptr),         value,   intent(in)      :: plan_ptr             !! C pointer to Fortran plan
    real(c_float),                intent(inout)   :: in(*)                !! Incomming buffer, not NULL
    real(c_float),                intent(inout)   :: out(*)               !! Outgoing buffer
    type(dtfft_execute_t),        intent(in)      :: execute_type         !! Type of execution
    real(c_float),      optional, intent(inout)   :: aux(*)               !! Aux buffer, can be NULL
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%execute(in, out, execute_type, aux, error_code)
  end function dtfft_execute_c

  function dtfft_transpose_c(plan_ptr, in, out, transpose_type)                                                     &
    result(error_code)                                                                                              &
    bind(C)
  !! Executes single transposition, C/C++ interface.
    type(c_ptr),        value,      intent(in)    :: plan_ptr             !! C pointer to Fortran plan
    real(c_float),                  intent(inout) :: in(*)                !! Incomming buffer, not NULL
    real(c_float),                  intent(inout) :: out(*)               !! Outgoing buffer, not NULL
    type(dtfft_transpose_t),        intent(in)    :: transpose_type       !! Type of transposition.
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%transpose(in, out, transpose_type, error_code)
  end function dtfft_transpose_c

  function dtfft_destroy_c(plan_ptr)                                                                                &
    result(error_code)                                                                                              &
    bind(C)
  !! Destroys dtFFT Plan, C/C++ interface
    type(c_ptr)                                   :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%destroy(error_code)
    deallocate( plan%p )
    deallocate( plan )
    plan_ptr = c_null_ptr
  end function dtfft_destroy_c

  function dtfft_get_local_sizes_c(plan_ptr, in_starts, in_counts, out_starts, out_counts, alloc_size)              &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns local sizes, counts in real and Fourier spaces and number of elements to be allocated for `in` and `out` buffers,
  !! C/C++ interface.
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t),  intent(out),    optional :: in_starts(3)         !! Starts of local portion of data in 'real' space
    integer(c_int32_t),  intent(out),    optional :: in_counts(3)         !! Counts of local portion of data in 'real' space
    integer(c_int32_t),  intent(out),    optional :: out_starts(3)        !! Starts of local portion of data in 'fourier' space
    integer(c_int32_t),  intent(out),    optional :: out_counts(3)        !! Counts of local portion of data in 'fourier' space
    integer(c_size_t),   intent(out),    optional :: alloc_size           !! Minimum data needs to be allocated
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%get_local_sizes(in_starts, in_counts, out_starts, out_counts, alloc_size, error_code)
  end function dtfft_get_local_sizes_c

  function dtfft_get_alloc_size_c(plan_ptr, alloc_size)                                                             &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns minimum number of bytes to be allocated for `in` and `out` buffers, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_size_t),   intent(out)              :: alloc_size           !! Minimum data needs to be allocated
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    alloc_size = plan%p%get_alloc_size(error_code)
  end function dtfft_get_alloc_size_c

  subroutine dtfft_get_error_string_c(error_code, error_string, error_string_size) bind(C)
  !! Returns an explaination of ``error_code`` that could have been previously returned by one of dtFFT API calls,
  !! C/C++ interface
    integer(c_int32_t),  intent(in)               :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    character(c_char),   intent(out)              :: error_string(*)      !! Explanation of error
    integer(c_size_t),   intent(out)              :: error_string_size    !! Size of ``error_string``

    call string_f2c(dtfft_get_error_string(error_code), error_string, error_string_size)
  end subroutine dtfft_get_error_string_c

  function dtfft_get_pencil_c(plan_ptr, dim, pencil)                                                                &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns pencil decomposition info, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int8_t),  intent(in)                :: dim                  !! Dimension requested
    type(dtfft_pencil_c)                          :: pencil               !! Pencil pointer
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object
    type(dtfft_pencil_t) :: pencil_

    CHECK_PLAN_CREATED(plan_ptr, plan)
    pencil_ = plan%p%get_pencil(dim, error_code)
    pencil%dim = pencil_%dim
    pencil%ndims = pencil_%ndims
    pencil%size = pencil_%size
    pencil%starts(1:pencil%ndims) = pencil_%starts(:)
    pencil%counts(1:pencil%ndims) = pencil_%counts(:)
  end function dtfft_get_pencil_c

  function dtfft_get_element_size_c(plan_ptr, element_size)                                                         &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns size of element in bytes, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_size_t),    intent(out)             :: element_size         !! Size of element in bytes
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    element_size = plan%p%get_element_size(error_code)
  end function dtfft_get_element_size_c

  function dtfft_set_config_c(config)                                                                               &
    result(error_code)                                                                                              &
    bind(C)
  !! Sets dtFFT configuration, C/C++ interface
    type(dtfft_config_t),             intent(in)  :: config               !! Configuration to set
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    call dtfft_set_config(config, error_code)
  end function dtfft_set_config_c

  function dtfft_report_c(plan_ptr)                                                                                 &
    result(error_code)                                                                                              &
    bind(C)
  !! Reports dtFFT Plan, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%report(error_code)
  end function dtfft_report_c

  function dtfft_mem_alloc_c(plan_ptr, alloc_bytes, ptr)                                                            &
    result(error_code)                                                                                              &
    bind(C)
  !! Allocates memory for dtFFT Plan, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    integer(c_size_t),                   value    :: alloc_bytes          !! Number of bytes to allocate
    type(c_ptr)                                   :: ptr                  !! Allocated pointer
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    ptr = plan%p%mem_alloc(alloc_bytes, error_code)
  end function dtfft_mem_alloc_c

  function dtfft_mem_free_c(plan_ptr, ptr)                                                                          &
    result(error_code)                                                                                              &
    bind(C)
  !! Frees memory for dtFFT Plan, C/C++ interface
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    type(c_ptr),                         value    :: ptr                  !! Pointer to deallocate
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%mem_free(ptr, error_code)
  end function dtfft_mem_free_c

#ifdef DTFFT_WITH_CUDA
  function dtfft_get_stream_c(plan_ptr, stream)                                                                     &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns Stream associated with plan
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    type(dtfft_stream_t),       intent(out)       :: stream               !! CUDA stream
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    call plan%p%get_stream(stream, error_code)
  end function dtfft_get_stream_c

  function dtfft_get_backend_c(plan_ptr, backend)                                                                   &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns selected [[dtfft_backend_t]] during autotuning
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    type(dtfft_backend_t),      intent(out)       :: backend              !! The enumerated type dtfft_backend_t
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    backend = plan%p%get_backend(error_code)
  end function dtfft_get_backend_c

  function dtfft_get_platform_c(plan_ptr, platform)                                                                 &
    result(error_code)                                                                                              &
    bind(C)
  !! Returns selected [[dtfft_platform_t]] during autotuning
    type(c_ptr),                         value    :: plan_ptr             !! C pointer to Fortran plan
    type(dtfft_platform_t),  intent(out)          :: platform             !! The enumerated type dtfft_platform_t
    integer(c_int32_t)                            :: error_code           !! The enumerated type dtfft_error_code_t
                                                                          !! defines API call result codes.
    type(dtfft_plan_c),                 pointer   :: plan                 !! Pointer to Fortran object

    CHECK_PLAN_CREATED(plan_ptr, plan)
    platform = plan%p%get_platform(error_code)
  end function dtfft_get_platform_c

  subroutine dtfft_get_backend_string_c(backend, backend_string, backend_string_size) bind(C)
  !! Returns string representation of ``dtfft_backend_t``
    type(dtfft_backend_t),  intent(in)       :: backend             !! The enumerated type dtfft_backend_t
    character(c_char),      intent(out)      :: backend_string(*)   !! Resulting string
    integer(c_size_t),      intent(out)      :: backend_string_size !! Size of string

    call string_f2c(dtfft_get_backend_string(backend), backend_string, backend_string_size)
  end subroutine dtfft_get_backend_string_c
#endif
end module dtfft_api