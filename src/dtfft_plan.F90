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
#include "dtfft_config.h"
#include "dtfft.f03"
!------------------------------------------------------------------------------------------------
module dtfft_plan
!! This module describes [[dtfft_plan_t]], [[dtfft_plan_c2c_t]], [[dtfft_plan_r2c_t]] and [[dtfft_plan_r2r_t]] types
use iso_c_binding,                    only: c_loc, c_f_pointer, c_ptr, c_bool, c_null_ptr
use iso_fortran_env,                  only: int8, int32, int64, real32, real64, output_unit, error_unit
use dtfft_abstract_executor,          only: abstract_executor, FFT_1D, FFT_2D, FFT_C2C, FFT_R2C, FFT_R2R
#ifdef DTFFT_WITH_FFTW
use dtfft_executor_fftw_m,            only: fftw_executor
#endif
#ifdef DTFFT_WITH_MKL
use dtfft_executor_mkl_m,             only: mkl_executor
#endif
#ifdef DTFFT_WITH_CUFFT
use dtfft_executor_cufft_m,           only: cufft_executor
#endif
#ifdef DTFFT_WITH_VKFFT
use dtfft_executor_vkfft_m,           only: vkfft_executor
#endif
use dtfft_config
use dtfft_errors
use dtfft_pencil,                     only: pencil, pencil_init, get_local_sizes_private => get_local_sizes, dtfft_pencil_t
use dtfft_parameters
use dtfft_transpose_plan,             only: transpose_plan
use dtfft_utils
#ifdef DTFFT_WITH_CUDA
use dtfft_interface_cuda_runtime
#endif
#ifdef DTFFT_WITH_NVSHMEM
use dtfft_interface_nvshmem,          only: is_nvshmem_ptr
#endif
#include "_dtfft_cuda.h"
#include "_dtfft_mpi.h"
#include "_dtfft_private.h"
#include "_dtfft_profile.h"
implicit none
private
public :: dtfft_plan_t
public :: dtfft_plan_c2c_t
public :: dtfft_plan_r2c_t
public :: dtfft_plan_r2r_t

  type :: transpose_request
  !! Handle for async transpose operation
    type(dtfft_transpose_t) :: transpose_type       !! Type of transposition requested
    logical                 :: is_started = .false. !! Flag that indicates if transpose was started
    type(c_ptr)             :: in                   !! Input pointer
    type(c_ptr)             :: out                  !! Output pointer
  end type transpose_request

  type :: fft_executor
  !! FFT handle
    class(abstract_executor), allocatable :: fft
    !! Executor
  end type fft_executor

  type, abstract :: dtfft_plan_t
  !! Abstract class for all ``dtFFT`` plans
  private
    integer(int8)                                 :: ndims
      !! Number of global dimensions
    integer(int32),                   allocatable :: dims(:)
      !! Global dimensions
    integer(int32),                   allocatable :: grid_dims(:)
      !! Grid decomposition dimensions
    type(dtfft_precision_t)                       :: precision
      !! Precision of transform
    logical                                       :: is_created = .false.
      !! Plan creation flag
    logical                                       :: is_transpose_plan = .false.
      !! Plan is transpose only
    logical                                       :: is_aux_alloc = .false.
      !! Auxiliary buffer is allocated internally
    logical                                       :: is_z_slab = .false.
      !! Using Z-slab optimization
      !!
      !! Only 3D plan.
      !!
      !! When .true., then data is distributed only Z direction and it creates a possibility for optimization:
      !!
      !! - 2 dimensional FFT plan is created for both X and Y dimensions
      !! - Single call to MPI_Alltoall is required to transpose data from X-align to Z-align
      !!
      !! For CUDA build this optimization means single CUDA kernel that tranposes data directly from X to Z
    logical                                       :: is_y_slab = .false.
      !! Using Y-slab optimization
      !!
      !! Only 3D plan.
      !!
      !! When .true., then data is distributed only Y direction and it creates a possibility for optimization:
      !!
      !! - 2 dimensional FFT plan is created for both Y and Z dimensions
      !! - Transpose from Y-align to Z-align is skipped
    type(dtfft_effort_t)                          :: effort
      !! User defined type of effort
    integer(int64)                                :: storage_size
      !! Single element size in bytes
    type(dtfft_executor_t)                        :: executor
      !! FFT executor type
    TYPE_MPI_COMM                                 :: comm
      !! Grid communicator
    TYPE_MPI_COMM,                    allocatable :: comms(:)
      !! Local 1d communicators
    type(transpose_plan)                          :: plan
      !! Transpose plan handle
    type(pencil),                     allocatable :: pencils(:)
      !! Information about data aligment and datatypes
    type(dtfft_platform_t)                        :: platform
      !! Execution platform
#ifdef DTFFT_WITH_CUDA
    type(dtfft_stream_t)                          :: stream
      !! CUDA Stream associated with current plan
#endif
    type(c_ptr)                                   :: aux_ptr
      !! Auxiliary pointer
    type(fft_executor),               allocatable :: fft(:)
      !! Internal fft runners
    integer(int32),                   allocatable :: fft_mapping(:)
      !! Memory and plan creation optimization.
      !! In case same FFTs needs to be run in different dimensions
      !! only single FFT plan needs to be created
  contains
  private
    procedure,  pass(self), non_overridable, public :: transpose          !! Performs single transposition
    procedure,  pass(self), non_overridable, public :: transpose_ptr      !! Performs single transposition using type(c_ptr) pointers instead of buffers
    procedure,  pass(self), non_overridable, public :: transpose_start    !! Starts an asynchronous transpose operation
    procedure,  pass(self), non_overridable, public :: transpose_start_ptr!! Starts an asynchronous transpose operation using type(c_ptr) pointers instead of buffers
    procedure,  pass(self), non_overridable, public :: transpose_end      !! Ends previously started transposition
    procedure,  pass(self), non_overridable, public :: execute            !! Executes plan
    procedure,  pass(self), non_overridable, public :: execute_ptr        !! Executes plan using type(c_ptr) pointers instead of buffers
    procedure,  pass(self), non_overridable, public :: destroy            !! Destroys plan
    procedure,  pass(self), non_overridable, public :: get_local_sizes    !! Returns local starts and counts in `real` and `fourier` spaces
    procedure,  pass(self), non_overridable, public :: get_alloc_size     !! Wrapper around ``get_local_sizes`` to obtain number of elements only
    procedure,  pass(self), non_overridable, public :: get_z_slab_enabled !! Returns logical value is Z-slab optimization is enabled
    procedure,  pass(self), non_overridable, public :: get_y_slab_enabled !! Returns logical value is Y-slab optimization is enabled
    procedure,  pass(self), non_overridable, public :: get_pencil         !! Returns pencil decomposition
    procedure,  pass(self), non_overridable, public :: get_element_size   !! Returns number of bytes required to store single element.
    procedure,  pass(self), non_overridable, public :: get_alloc_bytes    !! Returns minimum number of bytes required to execute plan
    procedure,  pass(self), non_overridable, public :: get_executor       !! Returns FFT Executor associated with plan
    procedure,  pass(self), non_overridable, public :: get_dims           !! Returns global dimensions
    procedure,  pass(self), non_overridable, public :: get_grid_dims      !! Returns grid decomposition dimensions
    procedure,  pass(self), non_overridable, public :: get_precision      !! Returns precision of plan
    procedure,  pass(self), non_overridable, public :: report             !! Prints plan details
    procedure,  pass(self), non_overridable, public :: mem_alloc_ptr      !! Allocates memory for type(c_ptr)
    generic,                                 public :: mem_alloc =>       &
                                                       mem_alloc_r32_1d,  &
                                                       mem_alloc_r64_1d,  &
                                                       mem_alloc_r32_2d,  &
                                                       mem_alloc_r64_2d,  &
                                                       mem_alloc_r32_3d,  &
                                                       mem_alloc_r64_3d,  &
                                                       mem_alloc_c32_1d,  &
                                                       mem_alloc_c64_1d,  &
                                                       mem_alloc_c32_2d,  &
                                                       mem_alloc_c64_2d,  &
                                                       mem_alloc_c32_3d,  &
                                                       mem_alloc_c64_3d
      !! Allocates memory specific for this plan
    procedure,  pass(self), non_overridable, public :: mem_free_ptr       !! Frees previously allocated memory for type(c_ptr)
    generic,                                 public :: mem_free =>        &
                                                       mem_free_r32_1d,   &
                                                       mem_free_r32_2d,   &
                                                       mem_free_r32_3d,   &
                                                       mem_free_r64_1d,   &
                                                       mem_free_r64_2d,   &
                                                       mem_free_r64_3d,   &
                                                       mem_free_c32_1d,   &
                                                       mem_free_c32_2d,   &
                                                       mem_free_c32_3d,   &
                                                       mem_free_c64_1d,   &
                                                       mem_free_c64_2d,   &
                                                       mem_free_c64_3d
    !! Frees previously allocated memory specific for this plan
    procedure,  pass(self), non_overridable, public :: get_backend        !! Returns selected backend during autotuning
#ifdef DTFFT_WITH_CUDA
    procedure,  pass(self), non_overridable, public :: get_platform       !! Returns plan execution platform
    generic,                                 public :: get_stream       & !! Returns CUDA stream associated with plan
                                                    => get_stream_ptr,  &
                                                       get_stream_int64
    procedure,  pass(self), non_overridable         :: get_stream_ptr     !! Returns CUDA stream associated with plan
    procedure,  pass(self), non_overridable         :: get_stream_int64   !! Returns CUDA stream associated with plan
#endif
    procedure,  pass(self), non_overridable         :: execute_private    !! Executes plan
    procedure,  pass(self), non_overridable         :: execute_2d         !! Executes 2d plan
    procedure,  pass(self), non_overridable         :: execute_z_slab     !! Executes Z slab plan
    procedure,  pass(self), non_overridable         :: execute_generic    !! Executes plan with specified auxiliary buffer
    procedure,  pass(self), non_overridable         :: transpose_private  !! Performs single transposition using type(c_ptr) pointers instead of buffers
    procedure,  pass(self), non_overridable         :: check_create_args  !! Check arguments provided to `create` subroutines
    procedure,  pass(self), non_overridable         :: create_private     !! Creates core
    procedure,  pass(self), non_overridable         :: alloc_fft_plans    !! Allocates `fft_executor` classes
    procedure,  pass(self), non_overridable         :: check_aux          !! Checks if aux buffer was passed
                                                                          !! and if not will allocate one internally
    procedure,  pass(self), non_overridable         :: mem_alloc_r32_1d   !! Allocates memory for 1d real32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_r64_1d   !! Allocates memory for 1d real64 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_r32_2d   !! Allocates memory for 2d real32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_r64_2d   !! Allocates memory for 2d real64 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_r32_3d   !! Allocates memory for 2d real32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_r64_3d   !! Allocates memory for 2d real64 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c32_1d   !! Allocates memory for 1d complex32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c64_1d   !! Allocates memory for 1d complex64 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c32_2d   !! Allocates memory for 2d complex32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c64_2d   !! Allocates memory for 2d complex64 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c32_3d   !! Allocates memory for 3d complex32 pointer
    procedure,  pass(self), non_overridable         :: mem_alloc_c64_3d   !! Allocates memory for 3d complex64 pointer
    procedure,  pass(self), non_overridable         :: mem_free_r32_1d    !! Frees real32 1d pointer
    procedure,  pass(self), non_overridable         :: mem_free_r64_1d    !! Frees real64 1d pointer
    procedure,  pass(self), non_overridable         :: mem_free_r32_2d    !! Frees real32 2d pointer
    procedure,  pass(self), non_overridable         :: mem_free_r64_2d    !! Frees real64 2d pointer
    procedure,  pass(self), non_overridable         :: mem_free_r32_3d    !! Frees real32 3d pointer
    procedure,  pass(self), non_overridable         :: mem_free_r64_3d    !! Frees real64 3d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c32_1d    !! Frees complex32 1d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c64_1d    !! Frees complex64 1d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c32_2d    !! Frees complex32 2d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c64_2d    !! Frees complex64 2d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c32_3d    !! Frees complex32 3d pointer
    procedure,  pass(self), non_overridable         :: mem_free_c64_3d    !! Frees complex64 3d pointer
  end type dtfft_plan_t

  type, abstract, extends(dtfft_plan_t) :: dtfft_core_c2c
  !! Abstract C2C Plan
  private
  contains
  private
    procedure, pass(self), non_overridable          :: create_c2c_core  !! Creates plan for both C2C and R2C
  end type dtfft_core_c2c

  type, extends(dtfft_core_c2c) :: dtfft_plan_c2c_t
  !! C2C Plan
  private
  contains
  private
    generic,              public  :: create => create_c2c,  &
                                               create_c2c_pencil
    !! Creates C2C plan
    procedure, pass(self)         :: create_c2c              !! Creates C2C plan using global dimensions
    procedure, pass(self)         :: create_c2c_pencil       !! Creates C2C plan using Pencil of local data
    procedure, pass(self)         :: create_c2c_internal     !! Private method that combines common logic for C2C plan creation
  end type dtfft_plan_c2c_t

  type, extends(dtfft_core_c2c) :: dtfft_plan_r2c_t
  !! R2C Plan
  private
    type(pencil)  :: real_pencil
      !! "Real" pencil decomposition info
  contains
  private
    generic,              public  :: create => create_r2c,  &
                                               create_r2c_pencil
    !! Creates R2C plan
    procedure, pass(self)         :: create_r2c              !! Creates R2C plan using global dimensions
    procedure, pass(self)         :: create_r2c_pencil       !! Creates R2C plan using Pencil of local data
    procedure, pass(self)         :: create_r2c_internal     !! Private method that combines common logic for R2C plan creation
  end type dtfft_plan_r2c_t

  type, extends(dtfft_plan_t) :: dtfft_plan_r2r_t
  !! R2R Plan
  private
  contains
  private
    generic,              public  :: create => create_r2r, &
                                               create_r2r_pencil
    !! Creates R2R plan
    procedure, pass(self)         :: create_r2r               !! Creates R2R plan using global dimensions
    procedure, pass(self)         :: create_r2r_pencil        !! Creates R2R plan using Pencil of local data
    procedure, pass(self)         :: create_r2r_internal      !! Private method that combines common logic for R2R plan creation
  end type dtfft_plan_r2r_t

contains

  subroutine transpose(self, in, out, transpose_type, error_code)
  !! Performs single transposition
  !!
  !! @note
  !! Buffers `in` and `out` cannot be the same
  !! @endnote
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(*),  target,           intent(inout) :: in(..)
      !! Incoming buffer of any rank and kind. Note that this buffer
      !! will be modified in GPU build
    type(*),  target,           intent(inout) :: out(..)
      !! Resulting buffer of any rank and kind
    type(dtfft_transpose_t),    intent(in)    :: transpose_type
      !! Type of transposition.
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    call self%transpose_ptr(c_loc(in), c_loc(out), transpose_type, error_code)
  end subroutine transpose

  subroutine transpose_ptr(self, in, out, transpose_type, error_code)
  !! Performs single transposition using type(c_ptr) pointers instead of buffers
  !!
  !! @note
  !! Buffers `in` and `out` cannot be the same
  !! @endnote
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Incoming pointer. Note that values of this pointer
      !! will be modified in GPU build
    type(c_ptr),                intent(in)    :: out
      !! Resulting pointer
    type(dtfft_transpose_t),    intent(in)    :: transpose_type
      !! Type of transposition.
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    PHASE_BEGIN("dtfft_transpose", COLOR_TRANSPOSE)
    call self%transpose_private(in, out, transpose_type, EXEC_BLOCKING, error_code)
    PHASE_END("dtfft_transpose")
  end subroutine transpose_ptr

  function transpose_start(self, in, out, transpose_type, error_code) result(request)
  !! Starts an asynchronous transpose operation
  !!
  !! @note
  !! Buffers `in` and `out` cannot be the same
  !! @endnote
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(*),  target,           intent(inout) :: in(..)
      !! Incoming buffer of any rank and kind. Note that this buffer
      !! will be modified in GPU build
    type(*),  target,           intent(inout) :: out(..)
      !! Resulting buffer of any rank and kind
    type(dtfft_transpose_t),    intent(in)    :: transpose_type
      !! Type of transposition.
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    type(dtfft_request_t)                      :: request
      !! Asynchronous handle describing started transpose operation
    request = self%transpose_start_ptr(c_loc(in), c_loc(out), transpose_type, error_code)
  end function transpose_start

  function transpose_start_ptr(self, in, out, transpose_type, error_code) result(request)
  !! Starts an asynchronous transpose operation using type(c_ptr) pointers instead of buffers
  !!
  !! @note
  !! Buffers `in` and `out` cannot be the same
  !! @endnote
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Incoming pointer. Note that values of this pointer
      !! will be modified in GPU build
    type(c_ptr),                intent(in)    :: out
      !! Resulting pointer
    type(dtfft_transpose_t),    intent(in)    :: transpose_type
      !! Type of transposition.
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    type(dtfft_request_t)                     :: request
      !! Asynchronous handle describing started transpose operation
    integer(int32)  :: ierr     !! Error code
    type(transpose_request),     pointer      :: internal_handle
      !! Handle to internal transpose structure

    PHASE_BEGIN("dtfft_transpose_start", COLOR_TRANSPOSE)
    request = dtfft_request_t(c_null_ptr)
    call self%transpose_private(in, out, transpose_type, EXEC_NONBLOCKING, ierr)
    if( ierr == DTFFT_SUCCESS ) then
      allocate(internal_handle)
      internal_handle%transpose_type = transpose_type
      internal_handle%is_started = .true.
      internal_handle%in = in
      internal_handle%out = out
      request%val = c_loc(internal_handle)
    endif
    if ( present( error_code ) ) error_code = ierr
    PHASE_END("dtfft_transpose_start")
  end function transpose_start_ptr

  subroutine transpose_end(self, request, error_code)
  !! Ends previously started transposition
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(dtfft_request_t),      intent(inout) :: request
      !! Handle obtained from [[dtfft_plan_t(type):transpose_start]] or [[dtfft_plan_t(type):transpose_start_ptr]]
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    type(transpose_request),    pointer       :: internal_handle
      !! Handle to internal transpose structure
    integer(int32)  :: ierr    !! Error code

    ierr = DTFFT_SUCCESS
#ifdef ENABLE_INPUT_CHECK
    if( is_null_ptr(request%val) ) ierr = DTFFT_ERROR_INVALID_REQUEST
    CHECK_ERROR_AND_RETURN
#endif
    call c_f_pointer(request%val, internal_handle)
#ifdef ENABLE_INPUT_CHECK
    if ( .not. internal_handle%is_started                                 &
      .or. .not. is_valid_transpose_type(internal_handle%transpose_type)  &
      .or. is_null_ptr(internal_handle%in)                                &
      .or. is_null_ptr(internal_handle%out)) ierr = DTFFT_ERROR_INVALID_REQUEST
    CHECK_ERROR_AND_RETURN
#endif

    PHASE_BEGIN("dtfft_transpose_end", COLOR_TRANSPOSE)
    call self%plan%execute_end(internal_handle%in, internal_handle%out, internal_handle%transpose_type, ierr)
    PHASE_END("dtfft_transpose_end")
#ifdef ENABLE_INPUT_CHECK
    CHECK_ERROR_AND_RETURN
#endif
    deallocate(internal_handle)
    request = dtfft_request_t(c_null_ptr)
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine transpose_end

  subroutine transpose_private(self, in, out, transpose_type, exec_type, error_code)
  !! Performs single transposition using type(c_ptr) pointers instead of buffers
  !!
  !! @note
  !! Buffers `in` and `out` cannot be the same
  !! @endnote
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Incoming pointer. Note that values of this pointer
      !! will be modified in GPU build
    type(c_ptr),                intent(in)    :: out
      !! Resulting pointer
    type(dtfft_transpose_t),    intent(in)    :: transpose_type
      !! Type of transposition.
    type(async_exec_t),         intent(in)    :: exec_type
      !! Type of asynchronous execution.
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr    !! Error code

    ierr = DTFFT_SUCCESS
#ifdef ENABLE_INPUT_CHECK
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( .not.is_valid_transpose_type(transpose_type)                                           &
         .or. ( self%ndims == 2 .and. abs(transpose_type%val) > 1 )                             &
         .or. (abs(transpose_type%val) == DTFFT_TRANSPOSE_X_TO_Z%val .and..not.self%is_z_slab)) &
      ierr = DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
    CHECK_ERROR_AND_RETURN
    if ( is_same_ptr(in, out) )                                                                 &
      ierr = DTFFT_ERROR_INPLACE_TRANSPOSE
    CHECK_ERROR_AND_RETURN
    select type( self )
    class is ( dtfft_plan_r2c_t )
      ierr = DTFFT_ERROR_R2C_TRANSPOSE_CALLED
      CHECK_ERROR_AND_RETURN
    endselect
# ifdef DTFFT_WITH_CUDA
    if ( self%platform == DTFFT_PLATFORM_CUDA  ) then
      ierr = check_device_pointers(in, out, self%plan%get_backend(), c_null_ptr)
      CHECK_ERROR_AND_RETURN
    endif
# endif
#endif
    call self%plan%execute(in, out, transpose_type, exec_type, ierr)
#ifdef ENABLE_INPUT_CHECK
    CHECK_ERROR_AND_RETURN
#endif
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine transpose_private

  subroutine execute(self, in, out, execute_type, aux, error_code)
  !! Executes plan
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(*),  target,           intent(inout) :: in(..)
      !! Incoming buffer of any rank and kind
    type(*),  target,           intent(inout) :: out(..)
      !! Resulting buffer of any rank and kind
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(*),  target, optional, intent(inout) :: aux(..)
      !! Optional auxiliary buffer.
      !! Size of buffer must be greater than value
      !! returned by `alloc_size` parameter of [[dtfft_plan_t(type):get_local_sizes]] subroutine
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    type(c_ptr)     :: aux_ptr

    aux_ptr = c_null_ptr; if( present(aux) ) aux_ptr = c_loc(aux)
    call self%execute_ptr(c_loc(in), c_loc(out), execute_type, aux_ptr, error_code)
  end subroutine execute

  subroutine execute_ptr(self, in, out, execute_type, aux, error_code)
  !! Executes plan using type(c_ptr) pointers instead of buffers
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Incoming pointer. Note that values of this pointer
      !! will be modified in GPU build
    type(c_ptr),                intent(in)    :: out
      !! Resulting pointer
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(c_ptr),                intent(in)    :: aux
      !! Auxiliary buffer. Not optional. If not required, c_null_ptr must be passed.
      !! Size of buffer must be greater than value
      !! returned by `alloc_size` parameter of [[dtfft_plan_t(type):get_local_sizes]] subroutine
    integer(int32),   optional, intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code
    logical         :: inplace  !! Inplace execution flag

    inplace = is_same_ptr(in, out)
    ierr = DTFFT_SUCCESS
#ifdef ENABLE_INPUT_CHECK
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
      CHECK_ERROR_AND_RETURN
    if ( .not.is_valid_execute_type(execute_type) ) ierr = DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
      CHECK_ERROR_AND_RETURN
    if ( self%is_transpose_plan .and. (self%ndims == 2 .or. self%is_y_slab) .and. inplace ) ierr = DTFFT_ERROR_INPLACE_TRANSPOSE
      CHECK_ERROR_AND_RETURN
    if ( is_same_ptr(in, aux) .or. is_same_ptr(out, aux) ) ierr = DTFFT_ERROR_INVALID_AUX
      CHECK_ERROR_AND_RETURN
    if ( self%plan%get_async_active() ) ierr = DTFFT_ERROR_TRANSPOSE_ACTIVE
      CHECK_ERROR_AND_RETURN
# ifdef DTFFT_WITH_CUDA
    if ( self%platform == DTFFT_PLATFORM_CUDA  ) then
      ierr = check_device_pointers(in, out, self%plan%get_backend(), aux)
        CHECK_ERROR_AND_RETURN
    endif
# endif
#endif

    PHASE_BEGIN("dtfft_execute", COLOR_EXECUTE)
    call self%check_aux(aux)
    if ( .not.is_null_ptr(aux) ) then
      call self%execute_private( in, out, execute_type, aux, inplace )
    else
      call self%execute_private( in, out, execute_type, self%aux_ptr, inplace )
    endif
    PHASE_END("dtfft_execute")
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine execute_ptr

  subroutine execute_private(self, in, out, execute_type, aux, inplace)
  !! Executes plan with specified auxiliary buffer
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Source pointer
    type(c_ptr),                intent(in)    :: out
      !! Target pointer
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(c_ptr),                intent(in)    :: aux
      !! Auxiliary pointer.
    logical,                    intent(in)    :: inplace
      !! Inplace execution flag

    if ( self%ndims == 2 .or. self%is_y_slab ) then
      call self%execute_2d(in, out, execute_type, aux)
    else if ( self%is_z_slab ) then
      call self%execute_z_slab(in, out, execute_type, aux, inplace)
    else
      call self%execute_generic(in, out, execute_type, aux)
    endif
  end subroutine execute_private

  subroutine execute_2d(self, in, out, execute_type, aux)
  !! Executes plan with specified auxiliary buffer
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Source pointer
    type(c_ptr),                intent(in)    :: out
      !! Target pointer
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(c_ptr),                intent(in)    :: aux
      !! Auxiliary pointer.

    if ( self%is_transpose_plan ) then
      if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
        call self%plan%execute(in, out, DTFFT_TRANSPOSE_X_TO_Y, EXEC_BLOCKING)
      else
        call self%plan%execute(in, out, DTFFT_TRANSPOSE_Y_TO_X, EXEC_BLOCKING)
      endif
      return
    endif

    if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
      ! 1d direct FFT X direction
      call self%fft(1)%fft%execute(in, aux, FFT_FORWARD)
      ! Transpose X -> Y
      call self%plan%execute(aux, out, DTFFT_TRANSPOSE_X_TO_Y, EXEC_BLOCKING)
      ! 1d FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(out, out, FFT_FORWARD)
    else
      ! 1d inverse FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(in, in, FFT_BACKWARD)
      ! Transpose Y -> X
      call self%plan%execute(in, aux, DTFFT_TRANSPOSE_Y_TO_X, EXEC_BLOCKING)
      ! 1d inverse FFT X direction
      call self%fft(1)%fft%execute(aux, out, FFT_BACKWARD)
    endif
  end subroutine execute_2d

  subroutine execute_z_slab(self, in, out, execute_type, aux, inplace)
  !! Executes plan with specified auxiliary buffer
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Source pointer
    type(c_ptr),                intent(in)    :: out
      !! Target pointer
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(c_ptr),                intent(in)    :: aux
      !! Auxiliary pointer.
    logical,                    intent(in)    :: inplace
      !! Inplace execution flag

    if ( self%is_transpose_plan ) then
      if ( inplace ) then
        call self%execute_generic(in, out, execute_type, aux)
        return
      endif
      if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
        call self%plan%execute(in, out, DTFFT_TRANSPOSE_X_TO_Z, EXEC_BLOCKING)
      else
        call self%plan%execute(in, out, DTFFT_TRANSPOSE_Z_TO_X, EXEC_BLOCKING)
      endif
      return
    endif

    if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
      ! 2d direct FFT X-Y directions
      call self%fft(1)%fft%execute(in, aux, FFT_FORWARD)
      ! Transpose X -> Z
      call self%plan%execute(aux, out, DTFFT_TRANSPOSE_X_TO_Z, EXEC_BLOCKING)
      ! 1d direct FFT Z direction
      call self%fft(3)%fft%execute(out, out, FFT_FORWARD)
    else
      ! 1d inverse FFT Z direction
      call self%fft(3)%fft%execute(in, in, FFT_BACKWARD)
      ! Transpose Z -> X
      call self%plan%execute(in, aux, DTFFT_TRANSPOSE_Z_TO_X, EXEC_BLOCKING)
      ! 2d inverse FFT X-Y direction
      call self%fft(1)%fft%execute(aux, out, FFT_BACKWARD)
    endif
  end subroutine execute_z_slab

  subroutine execute_generic(self, in, out, execute_type, aux)
  !! Executes plan with specified auxiliary buffer
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: in
      !! Source pointer
    type(c_ptr),                intent(in)    :: out
      !! Target pointer
    type(dtfft_execute_t),      intent(in)    :: execute_type
      !! Type of execution.
    type(c_ptr),                intent(in)    :: aux
      !! Auxiliary pointer.

    if ( self%is_transpose_plan ) then
      if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
        call self%plan%execute(in, aux, DTFFT_TRANSPOSE_X_TO_Y, EXEC_BLOCKING)
        call self%plan%execute(aux, out, DTFFT_TRANSPOSE_Y_TO_Z, EXEC_BLOCKING)
      else
        call self%plan%execute(in, aux, DTFFT_TRANSPOSE_Z_TO_Y, EXEC_BLOCKING)
        call self%plan%execute(aux, out, DTFFT_TRANSPOSE_Y_TO_X, EXEC_BLOCKING)
      endif
      return
    endif

    if ( execute_type == DTFFT_EXECUTE_FORWARD ) then
      ! 1d direct FFT X direction
      call self%fft(1)%fft%execute(in, aux, FFT_FORWARD)
      ! Transpose X -> Y
      call self%plan%execute(aux, out, DTFFT_TRANSPOSE_X_TO_Y, EXEC_BLOCKING)
      ! 1d FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(out, out, FFT_FORWARD)
      ! Transpose Y -> Z
      call self%plan%execute(out, aux, DTFFT_TRANSPOSE_Y_TO_Z, EXEC_BLOCKING)
      ! 1d direct FFT Z direction
      call self%fft(self%fft_mapping(3))%fft%execute(aux, out, FFT_FORWARD)
    else
      ! 1d inverse FFT Z direction
      call self%fft(self%fft_mapping(3))%fft%execute(in, aux, FFT_BACKWARD)
      ! Transpose Z -> Y
      call self%plan%execute(aux, in, DTFFT_TRANSPOSE_Z_TO_Y, EXEC_BLOCKING)
      ! 1d inverse FFT Y direction
      call self%fft(self%fft_mapping(2))%fft%execute(in, in, FFT_BACKWARD)
      ! Transpose Y -> X
      call self%plan%execute(in, aux, DTFFT_TRANSPOSE_Y_TO_X, EXEC_BLOCKING)
      ! 1d inverse FFT X direction
      call self%fft(1)%fft%execute(aux, out, FFT_BACKWARD)
    endif
  end subroutine execute_generic

  subroutine destroy(self, error_code)
  !! Destroys plan, frees all memory
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)  :: d        !! Counter
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    PHASE_BEGIN("dtfft_destroy", COLOR_DESTROY)

    if ( allocated(self%dims) ) deallocate(self%dims)
    if ( allocated(self%grid_dims) ) deallocate( self%grid_dims )

    select type ( self )
    class is ( dtfft_plan_r2c_t )
      call self%real_pencil%destroy()
    endselect

    if ( allocated(self%pencils) ) then
      do d = 1, self%ndims
        call self%pencils(d)%destroy()
      enddo
      deallocate(self%pencils)
    endif

    if ( self%is_aux_alloc ) then
      call self%mem_free_ptr(self%aux_ptr)
      self%is_aux_alloc = .false.
    endif

    if ( allocated(self%fft) ) then
      do d = 1, self%ndims
        call self%fft(d)%fft%destroy()
        if ( allocated(self%fft(d)%fft) ) deallocate(self%fft(d)%fft)
      enddo
      deallocate(self%fft)
    endif

    self%is_created = .false.
    self%is_aux_alloc = .false.

#ifdef DTFFT_WITH_CUDA
    call destroy_stream()
#endif

    block
      logical     :: is_finalized
      ! Following calls may contain calls to MPI
      ! Must make sure that MPI is still enabled
      call MPI_Finalized(is_finalized, ierr)

      if ( is_finalized ) ierr = DTFFT_ERROR_MPI_FINALIZED
      CHECK_ERROR_AND_RETURN
    end block

    call self%plan%destroy()

    if ( allocated(self%comms) ) then
      do d = 1, self%ndims
        call MPI_Comm_free(self%comms(d), ierr)
      enddo
      deallocate(self%comms)
    endif
    call MPI_Comm_free(self%comm, ierr)

    self%ndims = -1
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
    PHASE_END("dtfft_destroy")
    ! call cali_flush(0)
  end subroutine destroy

  logical function get_z_slab_enabled(self, error_code)
  !! Returns logical value is Z-slab optimization enabled internally
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_z_slab_enabled = .false.
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_z_slab_enabled = self%is_z_slab
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_z_slab_enabled

  logical function get_y_slab_enabled(self, error_code)
  !! Returns logical value is Y-slab optimization enabled internally
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_y_slab_enabled = .false.
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_y_slab_enabled = self%is_y_slab
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_y_slab_enabled

  type(dtfft_pencil_t) function get_pencil(self, layout, error_code)
  !! Returns pencil decomposition
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32),             intent(in)    :: layout
      !! Required layout:
      !!
      !!  - 0 for XYZ layout (real space, R2C only)
      !!  - 1 for XYZ layout
      !!  - 2 for YZX layout
      !!  - 3 for ZXY layout
      !!
      !! [//]: # (ListBreak)
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code
    integer(int32)  :: min_layout

    ierr = DTFFT_SUCCESS
    ! get_pencil = dtfft_pencil_t(-1,-1,-1,-1)
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    select type (self)
    class is ( dtfft_plan_r2c_t )
      min_layout = 0
    class default
      min_layout = 1
    endselect

    if ( layout < min_layout .or. layout > self%ndims ) ierr = DTFFT_ERROR_INVALID_DIM
    CHECK_ERROR_AND_RETURN
    if ( layout == 0 ) then
      select type (self)
      class is ( dtfft_plan_r2c_t )
        get_pencil = self%real_pencil%make_public()
      endselect
    else
      get_pencil = self%pencils(layout)%make_public()
    endif
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_pencil

  integer(int64) function get_element_size(self, error_code)
  !! Returns number of bytes required to store single element.
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_element_size = 0
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    select type (self)
    class is ( dtfft_plan_r2c_t )
      get_element_size = self%storage_size / 2
    class default
      get_element_size = self%storage_size
    endselect
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_element_size

  integer(int64) function get_alloc_bytes(self, error_code)
  !! Returns minimum number of bytes required to execute plan
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr         !! Error code
    integer(int64)  :: alloc_size   !! Number of elements required
    integer(int64)  :: element_size !! Size of each element

    ierr = DTFFT_SUCCESS
    get_alloc_bytes = 0
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    alloc_size = self%get_alloc_size()
    element_size = self%get_element_size()
    get_alloc_bytes = alloc_size * element_size
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_alloc_bytes

  type(dtfft_executor_t) function get_executor(self, error_code)
  !! Returns FFT Executor associated with plan
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_executor = dtfft_executor_t(VARIABLE_NOT_SET)
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_executor = self%executor
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_executor

  subroutine get_dims(self, dims, error_code)
  !! Returns global dimensions
    class(dtfft_plan_t), target,  intent(in)  :: self
      !! Abstract plan
    integer(int32),     pointer,  intent(out) :: dims(:)
      !! Global dimensions
    integer(int32),    optional,  intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    dims => self%dims
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine get_dims

  subroutine get_grid_dims(self, grid_dims, error_code)
  !! Returns grid decomposition dimensions
    class(dtfft_plan_t), target,  intent(in)  :: self
      !! Abstract plan
    integer(int32),     pointer,  intent(out) :: grid_dims(:)
      !! Grid dimensions
    integer(int32),    optional,  intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    grid_dims => self%grid_dims
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine get_grid_dims

  type(dtfft_precision_t) function get_precision(self, error_code)
  !! Returns precision of the plan
    class(dtfft_plan_t),        intent(in)    :: self
      !! Abstract plan
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_precision = dtfft_precision_t(VARIABLE_NOT_SET)
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_precision = self%precision
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function get_precision

  subroutine report(self, error_code)
  !! Prints plan-related information to stdout
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr                   !! Error code
    integer(int32)  :: comm_dims(self%ndims)  !! Communicator dimensions
    integer(int32)  :: d                      !! Counter

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    WRITE_REPORT("**Plan report**")
    WRITE_REPORT("  dtFFT Version        :  "//to_str(DTFFT_VERSION_MAJOR)//"."//to_str(DTFFT_VERSION_MINOR)//"."//to_str(DTFFT_VERSION_PATCH))
    WRITE_REPORT("  Number of dimensions :  "//to_str(self%ndims))

    do d = 2, self%ndims
      call MPI_Comm_size(self%comms(d), comm_dims(d), ierr)
    enddo
    if ( self%ndims == 2 ) then
      WRITE_REPORT("  Global dimensions    :  "//to_str(self%dims(1))//"x"//to_str(self%dims(2)))
      WRITE_REPORT("  Grid decomposition   :  1x"//to_str(comm_dims(2)))
    else
      WRITE_REPORT("  Global dimensions    :  "//to_str(self%dims(1))//"x"//to_str(self%dims(2))//"x"//to_str(self%dims(3)))
      WRITE_REPORT("  Grid decomposition   :  1x"//to_str(comm_dims(2))//"x"//to_str(comm_dims(3)))
    endif
#ifdef DTFFT_WITH_CUDA
    if ( self%platform == DTFFT_PLATFORM_HOST ) then
      WRITE_REPORT("  Execution platform   :  HOST")
    else
      WRITE_REPORT("  Execution platform   :  CUDA")
    endif
#endif
    select type( self )
    class is ( dtfft_plan_c2c_t )
      WRITE_REPORT("  Plan type            :  Complex-to-Complex")
    class is ( dtfft_plan_r2r_t )
      WRITE_REPORT("  Plan type            :  Real-to-Real")
    class is ( dtfft_plan_r2c_t )
      WRITE_REPORT("  Plan type            :  Real-to-Complex")
    endselect
      WRITE_REPORT("  Plan precision       :  "//dtfft_get_precision_string(self%precision))
      WRITE_REPORT("  FFT Executor type    :  "//dtfft_get_executor_string(self%executor))
    if ( self%ndims == 3 ) then
      if ( self%is_z_slab ) then
        WRITE_REPORT("  Z-slab enabled       :  True")
      else
        WRITE_REPORT("  Z-slab enabled       :  False")
      endif
      if ( self%is_y_slab ) then
        WRITE_REPORT("  Y-slab enabled       :  True")
      else
        WRITE_REPORT("  Y-slab enabled       :  False")
      endif
    endif
    WRITE_REPORT("  Backend              :  "//dtfft_get_backend_string(self%plan%get_backend()))
    WRITE_REPORT("**End of report**")
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine report

  type(dtfft_backend_t) function get_backend(self, error_code)
  !! Returns selected GPU backend during autotuning
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_backend = BACKEND_NOT_SET
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_backend = self%plan%get_backend()
    if ( present( error_code ) ) error_code = ierr
  end function get_backend

#ifdef DTFFT_WITH_CUDA
  type(dtfft_platform_t) function get_platform(self, error_code)
    !! Returns execution platform of the plan (HOST or CUDA)
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    get_platform = PLATFORM_NOT_SET
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    get_platform = self%platform
    if ( present( error_code ) ) error_code = ierr
  end function get_platform

  subroutine get_stream_ptr(self, stream, error_code)
  !! Returns CUDA stream associated with plan
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    type(dtfft_stream_t),       intent(out) :: stream
      !! dtFFT Stream
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    stream = NULL_STREAM
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( self%platform == DTFFT_PLATFORM_HOST ) ierr = DTFFT_ERROR_INVALID_USAGE
    CHECK_ERROR_AND_RETURN

    stream = self%stream
    if ( present( error_code ) ) error_code = ierr
  end subroutine get_stream_ptr

  subroutine get_stream_int64(self, stream, error_code)
  !! Returns CUDA stream associated with plan
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int64),             intent(out) :: stream
      !! CUDA-Fortran Stream
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)        :: ierr     !! Error code
    type(dtfft_stream_t)  :: stream_  !! dtFFT Stream

    call self%get_stream(stream_, error_code=ierr)
    if ( ierr == DTFFT_SUCCESS) stream = dtfft_get_cuda_stream(stream_)
    if ( present( error_code ) ) error_code = ierr
  end subroutine get_stream_int64

  integer(int32) function check_device_pointers(in, out, backend, aux) result(error_code)
  !! Checks if device pointers are provided by user
    type(c_ptr),            intent(in)  :: in
      !! First pointer
    type(c_ptr),            intent(in)  :: out
      !! Second pointer
    type(dtfft_backend_t),  intent(in)  :: backend
      !! Backend. Required to check for `nvshmem` pointer
    type(c_ptr),            intent(in)  :: aux
      !! Optional auxiliary pointer.
    logical(c_bool) :: is_devptr    !! Are pointers device pointers?

    error_code = DTFFT_SUCCESS

    if ( is_backend_nvshmem(backend) ) then
#ifdef DTFFT_WITH_NVSHMEM
      is_devptr = is_nvshmem_ptr(in) .and. is_nvshmem_ptr(out)
      if ( .not. is_null_ptr(aux) ) is_devptr = is_devptr .and. is_nvshmem_ptr(aux)
      if ( .not. is_devptr ) error_code = DTFFT_ERROR_NOT_NVSHMEM_PTR
#endif
    else
      is_devptr = is_device_ptr(in) .and. is_device_ptr(out)
      if ( .not. is_null_ptr(aux) ) is_devptr = is_devptr .and. is_device_ptr(aux)
      if ( .not. is_devptr ) error_code = DTFFT_ERROR_NOT_DEVICE_PTR
    endif
  end function check_device_pointers
#endif

  subroutine get_local_sizes(self, in_starts, in_counts, out_starts, out_counts, alloc_size, error_code)
  !! Obtain local starts and counts in ``real`` and ``fourier`` spaces
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int32), optional,   intent(out) :: in_starts(:)
      !! Starts of local portion of data in ``real`` space (0-based)
    integer(int32), optional,   intent(out) :: in_counts(:)
      !! Number of elements of local portion of data in 'real' space
    integer(int32), optional,   intent(out) :: out_starts(:)
      !! Starts of local portion of data in ``fourier`` space (0-based)
    integer(int32), optional,   intent(out) :: out_counts(:)
      !! Number of elements of local portion of data in ``fourier`` space
    integer(int64), optional,   intent(out) :: alloc_size
      !! Minimal number of elements required to execute plan
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr                 !! Error code

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( .not. present(in_starts)       &
    .and..not. present(in_counts)       &
    .and..not. present(out_starts)      &
    .and..not. present(out_counts)      &
    .and..not. present(alloc_size)) ierr = DTFFT_ERROR_INVALID_USAGE
    CHECK_ERROR_AND_RETURN

    select type ( self )
    class is (dtfft_plan_r2c_t)
      if (present( in_starts ) )    in_starts(1:self%ndims)   = self%real_pencil%starts(1:self%ndims)
      if (present( in_counts ) )    in_counts(1:self%ndims)   = self%real_pencil%counts(1:self%ndims)
      call get_local_sizes_private(self%pencils, out_starts=out_starts, out_counts=out_counts, alloc_size=alloc_size, is_y_slab=self%is_y_slab)
      if ( present( alloc_size ) ) alloc_size = max(int(product(self%real_pencil%counts), int64), 2 * alloc_size)
    class default
      call get_local_sizes_private(self%pencils, in_starts, in_counts, out_starts, out_counts, alloc_size, is_y_slab=self%is_y_slab)
    endselect
#ifdef DTFFT_WITH_CUDA
    if ( is_backend_nvshmem( self%plan%get_backend() ) .and. present(alloc_size) ) then
      block
        integer(int64) :: aux_size

        ! cufftMp pipelined may require aux buffer that is larger than
        ! required by dtfft. in such case we must make sure that
        ! buffer allocated by user is large enough
        aux_size = self%plan%get_aux_size()
        alloc_size = max(alloc_size, aux_size / self%storage_size )

        ALL_REDUCE(alloc_size, MPI_INTEGER8, MPI_MAX, self%comm, ierr)
      endblock
    endif
#endif
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine get_local_sizes

  function get_alloc_size(self, error_code) result(alloc_size)
  !! Wrapper around ``get_local_sizes`` to obtain number of elements only
    class(dtfft_plan_t),        intent(in)  :: self
      !! Abstract plan
    integer(int32), optional,   intent(out) :: error_code
      !! Optional error code returned to user
    integer(int64)                          :: alloc_size
      !! Minimal number of elements required to execute plan

    call self%get_local_sizes(alloc_size=alloc_size, error_code=error_code)
  end function get_alloc_size

  integer(int32) function create_private(self, sngl_type, sngl_storage_size, dbl_type, dbl_storage_size, dims, pencil, comm, precision, effort, executor, kinds)
#define __FUNC__ create_private
  !! Creates core
    class(dtfft_plan_t),              intent(inout) :: self
      !! Abstract plan
    TYPE_MPI_DATATYPE,                intent(in)    :: sngl_type
      !! MPI_Datatype for single precision plan
    integer(int64),                   intent(in)    :: sngl_storage_size
      !! Number of bytes needed to store single element (single precision)
    TYPE_MPI_DATATYPE,                intent(in)    :: dbl_type
      !! MPI_Datatype for double precision plan
    integer(int64),                   intent(in)    :: dbl_storage_size
      !! Number of bytes needed to store single element (double precision)
    integer(int32),         optional, intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional, intent(in)    :: pencil
      !! Pencil of local portion of data
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! User-defined communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    type(dtfft_r2r_kind_t), optional, intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    TYPE_MPI_DATATYPE       :: base_dtype           !! MPI_Datatype for current precision
    integer(int64)          :: base_storage         !! Number of bytes needed to store single element
    TYPE_MPI_COMM           :: comm_                !! MPI Communicator
    integer(int8)           :: d                    !! Counter
    integer(int32)          :: ierr

    create_private = DTFFT_SUCCESS
    CHECK_INTERNAL_CALL( self%check_create_args(dims, pencil, comm, precision, effort, executor, kinds) )

    select case ( self%precision%val )
    case ( DTFFT_SINGLE%val )
      base_storage = sngl_storage_size
      base_dtype = sngl_type
    case ( DTFFT_DOUBLE%val )
      base_storage = dbl_storage_size
      base_dtype = dbl_type
    case default
      INTERNAL_ERROR("unknown precision")
    endselect
    self%storage_size = base_storage

    if ( allocated(self%pencils) ) then
      do d = 1, size(self%pencils, kind=int8)
        call self%pencils(d)%destroy()
      enddo
      deallocate( self%pencils )
    endif

    if ( allocated(self%comms) ) then
      ! Can potentially leak some memory
      deallocate( self%comms )
    endif

    call self%plan%destroy()

    allocate(self%pencils(self%ndims))
    allocate(self%comms(self%ndims))

    comm_ = MPI_COMM_WORLD; if ( present( comm ) ) comm_ = comm
#ifdef DTFFT_WITH_CUDA
    if ( self%platform == DTFFT_PLATFORM_CUDA  ) then
      block
        TYPE_MPI_COMM   :: local_comm
        integer(int32)  :: n_devices, current_device, local_rank, local_size
        integer(int32), allocatable :: local_devices(:)

        call MPI_Comm_split_Type(comm_, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, local_comm, ierr)
        call MPI_Comm_size(local_comm, local_size, ierr)
        call MPI_Comm_rank(local_comm, local_rank, ierr)

        allocate( local_devices(local_size) )

        CUDA_CALL( "cudaGetDeviceCount", cudaGetDeviceCount(n_devices) )
        CUDA_CALL( "cudaGetDevice", cudaGetDevice(current_device) )

        call MPI_Allgather(current_device, 1, MPI_INTEGER4, local_devices, 1, MPI_INTEGER4, local_comm, ierr)
        call MPI_Comm_free(local_comm, ierr)
        if ( count_unique(local_devices) /= local_size ) then
          create_private = DTFFT_ERROR_GPU_NOT_SET
          return
        endif

        deallocate( local_devices )

        self%stream = get_conf_stream()
      endblock
    endif
#endif

    if ( present(pencil) ) then
      block
        integer(int32), allocatable :: fixed_dims(:)
        type(pencil_init) :: ipencil

        CHECK_INTERNAL_CALL( ipencil%create(pencil, comm_) )
        ! After creating internal pencil and validating user passed pencil
        ! We finally know global dimensions `dims`
        allocate( self%dims, source=ipencil%dims )
        allocate( fixed_dims, source=self%dims )
        select type( self )
        class is ( dtfft_plan_r2c_t )
          fixed_dims(1) = fixed_dims(1) / 2 + 1
          ipencil%counts(1) = ipencil%counts(1) / 2 + 1
        endselect
        CHECK_INTERNAL_CALL( self%plan%create(self%platform, fixed_dims, comm_, self%effort, base_dtype, base_storage, self%comm, self%comms, self%pencils, ipencil) )

        select type( self )
        class is ( dtfft_plan_r2c_t )
          ipencil%counts(1) = (ipencil%counts(1) - 1) * 2
          call self%real_pencil%create(self%ndims, 1_int8, self%dims, self%comms, ipencil%starts, ipencil%counts)
        endselect
        deallocate( fixed_dims )
        call ipencil%destroy()
      endblock
    else
      CHECK_INTERNAL_CALL( self%plan%create(self%platform, dims, comm_, self%effort, base_dtype, base_storage, self%comm, self%comms, self%pencils) )
    endif
    self%is_z_slab = self%plan%get_z_slab()

    if ( allocated( self%grid_dims ) ) deallocate( self%grid_dims )
    allocate( self%grid_dims(self%ndims) )
    do d = 1, self%ndims
      call MPI_Comm_size(self%comms(d), self%grid_dims(d), ierr)
    enddo

    self%is_y_slab = .false.
    if( self%ndims == 3 .and. .not. self%is_z_slab .and. self%grid_dims(self%ndims) == 1 .and. get_conf_y_slab_enabled() ) then
      self%is_y_slab = .true.
    endif
    call self%alloc_fft_plans(kinds)

    self%is_aux_alloc = .false.
#undef __FUNC__
  end function create_private

  integer(int32) function check_create_args(self, dims, pencil, comm, precision, effort, executor, kinds)
#define __FUNC__ check_create_args
  !! Check arguments provided by user and sets private variables
    class(dtfft_plan_t),                intent(inout) :: self
      !! Abstract plan
    integer(int32),         optional,   intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional,   intent(in)    :: pencil
      !! Pencil of local portion of data
    TYPE_MPI_COMM,          optional,   intent(in)    :: comm
      !! Optional MPI Communicator
    type(dtfft_precision_t),optional,   intent(in)    :: precision
      !! Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
    type(dtfft_effort_t),   optional,   intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional,   intent(in)    :: executor
      !! Type of External FFT Executor
    type(dtfft_r2r_kind_t), optional,   intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    integer(int32)          :: ierr             !! Error code
    integer(int32)          :: top_type         !! MPI Comm topology type
    integer(int32)          :: dim              !! Counter

    check_create_args = DTFFT_SUCCESS
    CHECK_INTERNAL_CALL( init_internal() )

    self%platform = get_conf_platform()
#ifdef DTFFT_DEBUG
    if ( .not.present(dims) .and. .not.present(pencil) ) INTERNAL_ERROR(".not.present(dims) .and. .not.present(pencil)")
    if ( present(dims) .and. present(pencil) ) INTERNAL_ERROR("present(dims) .and. present(pencil)")
#endif
    if ( allocated(self%dims) ) deallocate(self%dims)
    if ( present(dims) ) then
      self%ndims = size(dims, kind=int8)
      CHECK_INPUT_PARAMETER(self%ndims, is_valid_dimension, DTFFT_ERROR_INVALID_N_DIMENSIONS)
      if ( any(dims <= 0) ) then
        check_create_args = DTFFT_ERROR_INVALID_DIMENSION_SIZE
        return
      endif
      allocate( self%dims, source=dims )
    else
      self%ndims = pencil%ndims
      if ( self%ndims == 0 ) then
        check_create_args = DTFFT_ERROR_PENCIL_NOT_INITIALIZED
        return
      endif
    endif

    if ( present(comm) ) then
      call MPI_Topo_test(comm, top_type, ierr)
      CHECK_INPUT_PARAMETER(top_type, is_valid_comm_type, DTFFT_ERROR_INVALID_COMM_TYPE)
    endif

    self%precision = DTFFT_DOUBLE
    if ( present(precision) ) then
      CHECK_INPUT_PARAMETER(precision, is_valid_precision, DTFFT_ERROR_INVALID_PRECISION)
      self%precision = precision
    endif

    self%effort = DTFFT_ESTIMATE
    if ( present(effort) ) then
      CHECK_INPUT_PARAMETER(effort, is_valid_effort, DTFFT_ERROR_INVALID_EFFORT)
      self%effort = effort
    endif

    self%is_transpose_plan = .false.
    self%executor = DTFFT_EXECUTOR_NONE
    if ( present(executor) ) then
      CHECK_INPUT_PARAMETER(executor, is_valid_executor, DTFFT_ERROR_INVALID_EXECUTOR)
#ifdef DTFFT_WITH_CUDA
      if ( self%platform == DTFFT_PLATFORM_HOST ) then
        CHECK_INPUT_PARAMETER(executor, is_host_executor, DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR)
      else if ( self%platform == DTFFT_PLATFORM_CUDA  ) then
        CHECK_INPUT_PARAMETER(executor, is_cuda_executor, DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR)
      endif
#endif
      self%executor = executor
    endif
    if ( self%executor == DTFFT_EXECUTOR_NONE ) self%is_transpose_plan = .true.

#ifdef DTFFT_WITH_CUDA
    ! Platform is used to check correctness of both executor and backend
    ! DTFFT_PATIENT ignores backend set by user:
    if ( self%effort%val < DTFFT_PATIENT%val ) then
      block
        type(dtfft_backend_t) :: backend

        backend = get_conf_backend()
        if ( self%platform == DTFFT_PLATFORM_HOST ) then
          if ( is_backend_nccl(backend) .or. is_backend_nvshmem(backend) ) then
            __FUNC__ = DTFFT_ERROR_INVALID_PLATFORM_BACKEND
            return
          endif
        endif
      endblock
    endif
#endif

    if ( present(kinds) .and. .not. self%is_transpose_plan ) then
      do dim = 1, self%ndims
        CHECK_INPUT_PARAMETER(kinds(dim), is_valid_r2r_kind, DTFFT_ERROR_INVALID_R2R_KINDS)
      enddo
    endif
#undef __FUNC__
  end function check_create_args

  subroutine alloc_fft_plans(self, kinds)
  !! Allocates [[abstract_executor]] with required FFT class
  !! and populates [[dtfft_plan_t(type):fft_mapping]] with similar FFT ids
    class(dtfft_plan_t),              intent(inout) :: self
      !! Abstract plan
    type(dtfft_r2r_kind_t), optional, intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    integer(int8)                                   :: dim, dim2
      !! Counters
    type(dtfft_r2r_kind_t),           allocatable   :: kinds_(:)
      !! Dummy kinds

    if ( self%is_transpose_plan ) return

    allocate(self%fft(self%ndims))
    allocate(self%fft_mapping(self%ndims))

    do dim = 1, self%ndims
      self%fft_mapping(dim) = dim

      select case(self%executor%val)
#ifdef DTFFT_WITH_FFTW
      case (DTFFT_EXECUTOR_FFTW3%val)
        allocate(fftw_executor :: self%fft(dim)%fft)
#endif
#ifdef DTFFT_WITH_MKL
      case (DTFFT_EXECUTOR_MKL%val)
        allocate(mkl_executor :: self%fft(dim)%fft)
#endif
#ifdef DTFFT_WITH_CUFFT
      case (DTFFT_EXECUTOR_CUFFT%val)
        allocate(cufft_executor :: self%fft(dim)%fft)
#endif
#ifdef DTFFT_WITH_VKFFT
      case (DTFFT_EXECUTOR_VKFFT%val)
        allocate(vkfft_executor :: self%fft(dim)%fft)
#endif
      case default
        INTERNAL_ERROR("Executor type unrecognized")
      endselect
    enddo
    if( self%is_z_slab .or. self%is_y_slab ) return

    allocate(kinds_(self%ndims))
    kinds_(:) = dtfft_r2r_kind_t(-1); if ( present(kinds) ) kinds_(:) = kinds(:)
    ! Searching for similar FFT transforms in order to reduce time of plan creation
    ! and reduce memory usage
    ! Most profitable in GPU build
    do dim = 1_int8, self%ndims
      do dim2 = 1_int8, dim - 1_int8
        if ( dim == dim2 ) cycle
        select type ( self )
        class is ( dtfft_plan_r2c_t )
          if ( dim == 1 ) cycle
        endselect
        if ( self%pencils(dim)%counts(1) == self%pencils(dim2)%counts(1)                    &
             .and. product(self%pencils(dim)%counts) == product(self%pencils(dim2)%counts)  &
             .and. kinds_(dim) == kinds_(dim2) ) then
          self%fft_mapping(dim) = dim2
        endif
      enddo
    enddo
    deallocate(kinds_)
  end subroutine alloc_fft_plans

  subroutine check_aux(self, aux)
  !! Checks if aux buffer was passed by user and if not will allocate one internally
    class(dtfft_plan_t),            intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                    intent(in)    :: aux
      !! Optional auxiliary buffer.
    integer(int64)                                :: alloc_size
      !! Number of elements to be allocated
    integer(int32) :: ierr

    if ( self%is_aux_alloc .or. .not.is_null_ptr(aux) ) return

    alloc_size = self%get_alloc_size() * self%get_element_size()
    WRITE_DEBUG("Allocating auxiliary buffer of "//to_str(alloc_size)//" bytes")
    self%aux_ptr = self%mem_alloc_ptr(alloc_size, ierr);  DTFFT_CHECK(ierr)
    self%is_aux_alloc = .true.
  end subroutine check_aux

  subroutine create_r2r(self, dims, kinds, comm, precision, effort, executor, error_code)
  !! R2R Plan Constructor
    class(dtfft_plan_r2r_t),          intent(inout) :: self
      !! R2R Plan
    integer(int32),                   intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_r2r_kind_t), optional, intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)          :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_r2r_internal(dims=dims, kinds=kinds, comm=comm,precision=precision, effort=effort, executor=executor) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_r2r

  subroutine create_r2r_pencil(self, pencil, kinds, comm, precision, effort, executor, error_code)
  !! R2R Plan Constructor
    class(dtfft_plan_r2r_t),          intent(inout) :: self
      !! R2R Plan
    type(dtfft_pencil_t),             intent(in)    :: pencil
      !! Local pencil of data to be transformed
    type(dtfft_r2r_kind_t), optional, intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)          :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_r2r_internal(pencil=pencil, kinds=kinds, comm=comm, precision=precision, effort=effort, executor=executor) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_r2r_pencil

  integer(int32) function create_r2r_internal(self, dims, pencil, kinds, comm, precision, effort, executor)
  !! Creates plan for R2R plans
#define __FUNC__ create_r2r_internal
    class(dtfft_plan_r2r_t),          intent(inout) :: self
      !! R2R Plan
    integer(int32),         optional, intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional, intent(in)    :: pencil
      !! Pencil of data to be transformed
    type(dtfft_r2r_kind_t), optional, intent(in)    :: kinds(:)
      !! Kinds of R2R transform
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int8)           :: fft_rank           !! Rank of FFT transform
    integer(int32)          :: dim                !! Counter
    type(dtfft_r2r_kind_t)  :: r2r_kinds(2)       !! Transposed Kinds of R2R transform

    create_r2r_internal = DTFFT_SUCCESS
    if ( self%is_created ) then
      create_r2r_internal = DTFFT_ERROR_PLAN_IS_CREATED
      return
    endif

    PHASE_BEGIN("dtfft_create_r2r", COLOR_CREATE)
    CHECK_INTERNAL_CALL( self%create_private(MPI_REAL, FLOAT_STORAGE_SIZE, MPI_REAL8, DOUBLE_STORAGE_SIZE, dims=dims, pencil=pencil, comm=comm, precision=precision, effort=effort, executor=executor, kinds=kinds) )

    if ( .not. self%is_transpose_plan ) then
      if ( .not. present( kinds ) ) then
        create_r2r_internal = DTFFT_ERROR_MISSING_R2R_KINDS
        return
      endif

      do dim = 1, self%ndims
        r2r_kinds(1) = kinds(dim)
        fft_rank = FFT_1D
        if ( self%is_z_slab .and. dim == 1 ) then
          r2r_kinds(1) = kinds(2)
          r2r_kinds(2) = kinds(1)
          fft_rank = FFT_2D
        else if ( self%is_y_slab .and. dim == 2 ) then
          r2r_kinds(1) = kinds(3)
          r2r_kinds(2) = kinds(2)
          fft_rank = FFT_2D
        endif
        if ( self%is_z_slab .and. dim == 2 ) cycle
        if ( self%is_y_slab .and. dim == 3 ) cycle
        CHECK_INTERNAL_CALL( self%fft(self%fft_mapping(dim))%fft%create(fft_rank, FFT_R2R, self%precision, real_pencil=self%pencils(dim), r2r_kinds=r2r_kinds) )
      enddo
    endif
    self%is_created = .true.
    PHASE_END("dtfft_create_r2r")
#undef __FUNC__
  end function create_r2r_internal

  subroutine create_c2c(self, dims, comm, precision, effort, executor, error_code)
  !! C2C Plan Constructor
    class(dtfft_plan_c2c_t),          intent(inout) :: self
      !! C2C Plan
    integer(int32),                   intent(in)    :: dims(:)
      !! Global dimensions of transform
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)                    :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_c2c_internal(dims=dims, comm=comm, precision=precision, effort=effort, executor=executor) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_c2c

  subroutine create_c2c_pencil(self, pencil, comm, precision, effort, executor, error_code)
  !! C2C Plan Constructor
    class(dtfft_plan_c2c_t),          intent(inout) :: self
      !! C2C Plan
    type(dtfft_pencil_t),             intent(in)    :: pencil
      !! Local pencil of data to be transformed
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)                    :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_c2c_internal(pencil=pencil, comm=comm, precision=precision, effort=effort, executor=executor) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_c2c_pencil

  integer(int32) function create_c2c_internal(self, dims, pencil, comm, precision, effort, executor)
  !! Private method that combines common logic for C2C plan creation
#define __FUNC__ create_c2c_internal
    class(dtfft_plan_c2c_t),          intent(inout) :: self
      !! C2C Plan
    integer(int32),         optional, intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional, intent(in)    :: pencil
      !! Pencil of data to be transformed
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor

    create_c2c_internal = DTFFT_SUCCESS
    if ( self%is_created ) then
      create_c2c_internal = DTFFT_ERROR_PLAN_IS_CREATED
      return
    endif

    PHASE_BEGIN("dtfft_create_c2c", COLOR_CREATE)
    CHECK_INTERNAL_CALL( self%create_c2c_core(dims, pencil, comm, precision, effort, executor) )
    self%is_created = .true.
    PHASE_END("dtfft_create_c2c")
#undef __FUNC__
  end function create_c2c_internal

  integer(int32) function create_c2c_core(self, dims, pencil, comm, precision, effort, executor)
  !! Creates plan for both C2C and R2C
#define __FUNC__ create_c2c_core
    class(dtfft_core_c2c),            intent(inout) :: self
      !! C2C Plan
    integer(int32),         optional, intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional, intent(in)    :: pencil
      !! Pencil of data to be transformed
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    type(dtfft_executor_t), optional, intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int8)           :: dim                  !! Counter
    integer(int8)           :: fft_start            !! 1 for c2c, 2 for r2c
    integer(int8)           :: fft_rank             !! Rank of FFT transform

    CHECK_INTERNAL_CALL( self%create_private(MPI_COMPLEX, COMPLEX_STORAGE_SIZE, MPI_DOUBLE_COMPLEX, DOUBLE_COMPLEX_STORAGE_SIZE, dims=dims, pencil=pencil, comm=comm, precision=precision, effort=effort, executor=executor) )

    if ( self%is_transpose_plan ) return
    fft_start = 1
    select type ( self )
    class is (dtfft_plan_r2c_t)
      fft_start = 2
    endselect
    do dim = fft_start, self%ndims
      fft_rank = FFT_1D
      if ( (self%is_z_slab .and. dim == 1) .or. (self%is_y_slab .and. dim == 2) ) fft_rank = FFT_2D
      if ( (self%is_z_slab .and. dim == 2) .or. (self%is_y_slab .and. dim == 3) ) cycle
      CHECK_INTERNAL_CALL( self%fft(self%fft_mapping(dim))%fft%create(fft_rank, FFT_C2C, self%precision, complex_pencil=self%pencils(dim)) )
    enddo
#undef __FUNC__
  end function create_c2c_core

  subroutine create_r2c(self, dims, executor, comm, precision, effort, error_code)
  !! R2C Generic Plan Constructor
    class(dtfft_plan_r2c_t),          intent(inout) :: self
      !! C2C Plan
    integer(int32),                   intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_executor_t),           intent(in)    :: executor
    !! Type of External FFT Executor
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)                    :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_r2c_internal(executor, dims=dims, comm=comm, precision=precision, effort=effort) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_r2c

  subroutine create_r2c_pencil(self, pencil, executor, comm, precision, effort, error_code)
  !! R2C Plan Constructor with pencil
    class(dtfft_plan_r2c_t),          intent(inout) :: self
      !! R2C Plan
    type(dtfft_pencil_t),             intent(in)    :: pencil
      !! Local pencil of data to be transformed
    type(dtfft_executor_t),           intent(in)    :: executor
      !! Type of External FFT Executor
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    integer(int32),         optional, intent(out)   :: error_code
      !! Optional Error Code returned to user
    integer(int32)                    :: ierr               !! Error code

    CHECK_OPTIONAL_CALL( self%create_r2c_internal(executor, pencil=pencil, comm=comm, precision=precision, effort=effort) )
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine create_r2c_pencil

  integer(int32) function create_r2c_internal(self, executor, dims, pencil, comm, precision, effort)
  !! Private method that combines common logic for R2C plan creation
#define __FUNC__ create_r2c_internal
    class(dtfft_plan_r2c_t),          intent(inout) :: self
      !! R2C Plan
    type(dtfft_executor_t),           intent(in)    :: executor
      !! Type of External FFT Executor
    integer(int32),         optional, intent(in)    :: dims(:)
      !! Global dimensions of transform
    type(dtfft_pencil_t),   optional, intent(in)    :: pencil
      !! Local pencil of data to be transformed
    TYPE_MPI_COMM,          optional, intent(in)    :: comm
      !! Communicator
    type(dtfft_precision_t),optional, intent(in)    :: precision
      !! Presicion of Transform
    type(dtfft_effort_t),   optional, intent(in)    :: effort
      !! How thoroughly `dtFFT` searches for the optimal plan
    integer(int32),   allocatable     :: fixed_dims(:)      !! Fixed dimensions for R2C
    integer(int8)                     :: fft_rank           !! Rank of FFT transform

    create_r2c_internal = DTFFT_SUCCESS
    if ( self%is_created ) then
      create_r2c_internal = DTFFT_ERROR_PLAN_IS_CREATED
      return
    endif

    PHASE_BEGIN("dtfft_create_r2c", COLOR_CREATE)
    if ( present(dims) ) then
      allocate(fixed_dims, source=dims)
      fixed_dims(1) = int(dims(1) / 2, int32) + 1
      CHECK_INTERNAL_CALL( self%create_c2c_core(dims=fixed_dims, comm=comm, precision=precision, effort=effort, executor=executor) )
      deallocate( fixed_dims )

      call self%real_pencil%create(self%ndims, 1_int8, dims, self%comms)
    else
      ! Do not know global dimensions
      ! They are computed when private pencil is created
      CHECK_INTERNAL_CALL( self%create_c2c_core(pencil=pencil, comm=comm, precision=precision, effort=effort, executor=executor) )
    endif
    if ( self%is_transpose_plan ) then
      create_r2c_internal = DTFFT_ERROR_R2C_TRANSPOSE_PLAN
      return
    endif

    fft_rank = FFT_1D;  if ( self%is_z_slab ) fft_rank = FFT_2D
    CHECK_INTERNAL_CALL( self%fft(1)%fft%create(fft_rank, FFT_R2C, self%precision, real_pencil=self%real_pencil, complex_pencil=self%pencils(1)) )
    PHASE_END("dtfft_create_r2c")
    self%is_created = .true.
#undef __FUNC__
  end function create_r2c_internal

  function mem_alloc_ptr(self, alloc_bytes, error_code) result(ptr)
  !! Allocates memory specific for this plan
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    integer(int64),             intent(in)    :: alloc_bytes
      !! Number of bytes to allocate
    type(c_ptr)                               :: ptr
      !! Allocated pointer
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    ptr = c_null_ptr
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN
    if ( alloc_bytes < FLOAT_STORAGE_SIZE ) ierr = DTFFT_ERROR_INVALID_ALLOC_BYTES
    CHECK_ERROR_AND_RETURN

    if ( self%platform == DTFFT_PLATFORM_HOST ) then
      if( self%is_transpose_plan ) then
        ptr = mem_alloc_host(alloc_bytes)
      else
        call self%fft(1)%fft%mem_alloc(alloc_bytes, ptr)
      endif
      if ( is_null_ptr(ptr) ) ierr = DTFFT_ERROR_ALLOC_FAILED
#ifdef DTFFT_WITH_CUDA
    else
      call self%plan%mem_alloc(self%comm, alloc_bytes, ptr, ierr)
#endif
    endif
    CHECK_ERROR_AND_RETURN
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end function mem_alloc_ptr

  subroutine mem_free_ptr(self, ptr, error_code)
  !! Frees previously allocated memory specific for this plan
    class(dtfft_plan_t),        intent(inout) :: self
      !! Abstract plan
    type(c_ptr),                intent(in)    :: ptr
      !! Pointer allocated with [[dtfft_plan_t(type):mem_alloc]]
    integer(int32), optional,   intent(out)   :: error_code
      !! Optional error code returned to user
    integer(int32)  :: ierr     !! Error code

    ierr = DTFFT_SUCCESS
    if ( .not. self%is_created ) ierr = DTFFT_ERROR_PLAN_NOT_CREATED
    CHECK_ERROR_AND_RETURN

    if ( self%platform == DTFFT_PLATFORM_HOST ) then
      if( self%is_transpose_plan ) then
        call mem_free_host(ptr)
      else
        call self%fft(1)%fft%mem_free(ptr)
      endif
#ifdef DTFFT_WITH_CUDA
    else
      call self%plan%mem_free(ptr, ierr)
#endif
    endif
    CHECK_ERROR_AND_RETURN
    if ( present( error_code ) ) error_code = DTFFT_SUCCESS
  end subroutine mem_free_ptr

#define STORAGE_BYTES FLOAT_STORAGE_SIZE
#define BUFFER_TYPE real(real32)
#define ALLOC_1D mem_alloc_r32_1d
#define ALLOC_2D mem_alloc_r32_2d
#define ALLOC_3D mem_alloc_r32_3d
#define FREE_1D mem_free_r32_1d
#define FREE_2D mem_free_r32_2d
#define FREE_3D mem_free_r32_3d
#include "_dtfft_mem_alloc_free.inc"

#define STORAGE_BYTES DOUBLE_STORAGE_SIZE
#define BUFFER_TYPE real(real64)
#define ALLOC_1D mem_alloc_r64_1d
#define ALLOC_2D mem_alloc_r64_2d
#define ALLOC_3D mem_alloc_r64_3d
#define FREE_1D mem_free_r64_1d
#define FREE_2D mem_free_r64_2d
#define FREE_3D mem_free_r64_3d
#include "_dtfft_mem_alloc_free.inc"

#define STORAGE_BYTES COMPLEX_STORAGE_SIZE
#define BUFFER_TYPE complex(real32)
#define ALLOC_1D mem_alloc_c32_1d
#define ALLOC_2D mem_alloc_c32_2d
#define ALLOC_3D mem_alloc_c32_3d
#define FREE_1D mem_free_c32_1d
#define FREE_2D mem_free_c32_2d
#define FREE_3D mem_free_c32_3d
#include "_dtfft_mem_alloc_free.inc"

#define STORAGE_BYTES DOUBLE_COMPLEX_STORAGE_SIZE
#define BUFFER_TYPE complex(real64)
#define ALLOC_1D mem_alloc_c64_1d
#define ALLOC_2D mem_alloc_c64_2d
#define ALLOC_3D mem_alloc_c64_3d
#define FREE_1D mem_free_c64_1d
#define FREE_2D mem_free_c64_2d
#define FREE_3D mem_free_c64_3d
#include "_dtfft_mem_alloc_free.inc"
end module dtfft_plan