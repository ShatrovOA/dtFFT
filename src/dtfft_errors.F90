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
module dtfft_errors
!! Module that defines error codes and provides utility to get error string
use iso_fortran_env, only: int32
implicit none
private
public :: dtfft_get_error_string

  integer(int32),  parameter,  public  :: DTFFT_SUCCESS = CONF_DTFFT_SUCCESS
    !! Successful execution
  integer(int32),  parameter,  public  :: DTFFT_ERROR_MPI_FINALIZED = CONF_DTFFT_ERROR_MPI_FINALIZED
    !! MPI_Init is not called or MPI_Finalize has already been called
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PLAN_NOT_CREATED = CONF_DTFFT_ERROR_PLAN_NOT_CREATED
    !! Plan not created
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_TRANSPOSE_TYPE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
    !! Invalid `transpose_type` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_N_DIMENSIONS = CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS
    !! Invalid Number of dimensions provided. Valid options are 2 and 3
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_DIMENSION_SIZE = CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE
    !! One or more provided dimension sizes <= 0 
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_TYPE = CONF_DTFFT_ERROR_INVALID_COMM_TYPE
    !! Invalid communicator type provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PRECISION = CONF_DTFFT_ERROR_INVALID_PRECISION
    !! Invalid `precision` parameter provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_EFFORT = CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG
    !! Invalid `effort` parameter provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_EXECUTOR = CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE
    !! Invalid `executor` parameter provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_DIMS = CONF_DTFFT_ERROR_INVALID_COMM_DIMS
    !! Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_FAST_DIM = CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM
    !! Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1
  integer(int32),  parameter,  public  :: DTFFT_ERROR_MISSING_R2R_KINDS = CONF_DTFFT_ERROR_MISSING_R2R_KINDS
    !! For R2R plan, `kinds` parameter must be passed if `executor` != `DTFFT_EXECUTOR_NONE`
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_R2R_KINDS = CONF_DTFFT_ERROR_INVALID_R2R_KINDS
    !! Invalid values detected in `kinds` parameter 
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2C_TRANSPOSE_PLAN = CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN
    !! Transpose plan is not supported in R2C, use R2R or C2C plan instead
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INPLACE_TRANSPOSE = CONF_DTFFT_ERROR_INPLACE_TRANSPOSE
    !! Inplace transpose is not supported
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_AUX = CONF_DTFFT_ERROR_INVALID_AUX
    !! Invalid `aux` buffer provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_LAYOUT = CONF_DTFFT_ERROR_INVALID_LAYOUT
    !! Invalid `dim` passed to `plan.get_pencil`
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_USAGE = CONF_DTFFT_ERROR_INVALID_USAGE
    !! Invalid API Usage.
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PLAN_IS_CREATED = CONF_DTFFT_ERROR_PLAN_IS_CREATED
    !! Trying to create already created plan
  integer(int32),  parameter,  public  :: DTFFT_ERROR_ALLOC_FAILED = CONF_DTFFT_ERROR_ALLOC_FAILED
    !! Internal allocation failed
  integer(int32),  parameter,  public  :: DTFFT_ERROR_FREE_FAILED = CONF_DTFFT_ERROR_FREE_FAILED
    !! Internal memory free failed
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_ALLOC_BYTES = CONF_DTFFT_ERROR_INVALID_ALLOC_BYTES
    !! Invalid `alloc_bytes` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_DLOPEN_FAILED = CONF_DTFFT_ERROR_DLOPEN_FAILED
    !! dlopen failed
  integer(int32),  parameter,  public  :: DTFFT_ERROR_DLSYM_FAILED = CONF_DTFFT_ERROR_DLSYM_FAILED
    !! dlsym failed
  ! integer(int32),  parameter,  public  :: DTFFT_ERROR_R2C_TRANSPOSE_CALLED = CONF_DTFFT_ERROR_R2C_TRANSPOSE_CALLED
    !! Calling to `transpose` method for R2C plan is not allowed
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH = CONF_DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH
    !! Sizes of `starts` and `counts` arrays passed to `dtfft_pencil_t` constructor do not match
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES = CONF_DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES
    !! Sizes of `starts` and `counts` < 2 or > 3 provided to `dtfft_pencil_t` constructor
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_INVALID_COUNTS = CONF_DTFFT_ERROR_PENCIL_INVALID_COUNTS
    !! Invalid `counts` provided to `dtfft_pencil_t` constructor
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_INVALID_STARTS = CONF_DTFFT_ERROR_PENCIL_INVALID_STARTS
    !! Invalid `starts` provided to `dtfft_pencil_t` constructor
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_SHAPE_MISMATCH = CONF_DTFFT_ERROR_PENCIL_SHAPE_MISMATCH
    !! Processes have same lower bounds but different sizes in some dimensions
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_OVERLAP = CONF_DTFFT_ERROR_PENCIL_OVERLAP
    !! Pencil overlap detected, i.e. two processes share same part of global space
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_NOT_CONTINUOUS = CONF_DTFFT_ERROR_PENCIL_NOT_CONTINUOUS
    !! Local pencils do not cover the global space without gaps
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PENCIL_NOT_INITIALIZED = CONF_DTFFT_ERROR_PENCIL_NOT_INITIALIZED
    !! Pencil is not initialized, i.e. `constructor` subroutine was not called
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS = CONF_DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS
    !! Invalid `n_measure_warmup_iters` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_MEASURE_ITERS = CONF_DTFFT_ERROR_INVALID_MEASURE_ITERS
    !! Invalid `n_measure_iters` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_REQUEST = CONF_DTFFT_ERROR_INVALID_REQUEST
    !! Invalid `dtfft_request_t` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_TRANSPOSE_ACTIVE = CONF_DTFFT_ERROR_TRANSPOSE_ACTIVE
    !! Attempting to execute already active transposition
  integer(int32),  parameter,  public  :: DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE = CONF_DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE
    !! Attempting to finalize non-active transposition
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_RESHAPE_TYPE = CONF_DTFFT_ERROR_INVALID_RESHAPE_TYPE
    !! Invalid `reshape_type` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_RESHAPE_ACTIVE = CONF_DTFFT_ERROR_RESHAPE_ACTIVE
    !! Attempting to execute already active reshape
  integer(int32),  parameter,  public  :: DTFFT_ERROR_RESHAPE_NOT_ACTIVE = CONF_DTFFT_ERROR_RESHAPE_NOT_ACTIVE
    !! Attempting to finalize non-active reshape
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INPLACE_RESHAPE = CONF_DTFFT_ERROR_INPLACE_RESHAPE
    !! Inplace reshape is not supported
  ! integer(int32),  parameter,  public  :: DTFFT_ERROR_R2C_RESHAPE_CALLED = CONF_DTFFT_ERROR_R2C_RESHAPE_CALLED
    !! R2C reshape was called
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_EXECUTE_TYPE = CONF_DTFFT_ERROR_INVALID_EXECUTE_TYPE
    !! Invalid `execute_type` provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_RESHAPE_NOT_SUPPORTED = CONF_DTFFT_ERROR_RESHAPE_NOT_SUPPORTED
    !! Reshape is not supported for this plan
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2C_EXECUTE_CALLED = CONF_DTFFT_ERROR_R2C_EXECUTE_CALLED
    !! Execute called for transpose-only R2C Plan
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_CART_COMM = CONF_DTFFT_ERROR_INVALID_CART_COMM
    !! Invalid cartesian communicator provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
    !! Selected `executor` do not support R2R FFTs
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM
    !! Invalid stream provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_BACKEND = CONF_DTFFT_ERROR_INVALID_BACKEND
    !! Invalid backend provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET
    !! Multiple MPI Processes located on same host share same GPU which is not supported
  integer(int32),  parameter,  public  :: DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN
    !! When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions
  integer(int32),  parameter,  public  :: DTFFT_ERROR_BACKENDS_DISABLED = CONF_DTFFT_ERROR_BACKENDS_DISABLED
    !! Passed `effort` ==  `DTFFT_PATIENT` but all GPU Backends has been disabled by `dtfft_config_t` */
  integer(int32),  parameter,  public  :: DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR
    !! One of pointers passed to `plan.execute` or `plan.transpose` cannot be accessed from device
  integer(int32),  parameter,  public  :: DTFFT_ERROR_NOT_NVSHMEM_PTR = CONF_DTFFT_ERROR_NOT_NVSHMEM_PTR
    !! One of pointers passed to `plan.execute` or `plan.transpose` is not an `NVSHMEM` pointer
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PLATFORM = CONF_DTFFT_ERROR_INVALID_PLATFORM
    !! Invalid platform provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR = CONF_DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR
    !! Invalid executor provided for selected platform
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PLATFORM_BACKEND = CONF_DTFFT_ERROR_INVALID_PLATFORM_BACKEND
    !! Invalid backend provided for selected platform

contains

  pure function dtfft_get_error_string(error_code) result(error_string)
  !! Gets the string description of an error code
    integer(int32),   intent(in)   :: error_code    !! Error code
    character(len=:), allocatable  :: error_string  !! Error string

    select case (error_code)
    case ( DTFFT_SUCCESS )
      allocate(error_string, source="DTFFT_SUCCESS")
    case ( DTFFT_ERROR_MPI_FINALIZED )
      allocate(error_string, source="MPI_Init is not called or MPI_Finalize has already been called")
    case ( DTFFT_ERROR_PLAN_NOT_CREATED )
      allocate(error_string, source="Plan not created")
    case ( DTFFT_ERROR_INVALID_TRANSPOSE_TYPE )
      allocate(error_string, source="Invalid `transpose_type` provided")
    case ( DTFFT_ERROR_INVALID_N_DIMENSIONS )
      allocate(error_string, source="Invalid Number of dimensions provided. Valid options are 2 and 3")
    case ( DTFFT_ERROR_INVALID_DIMENSION_SIZE )
      allocate(error_string, source="One or more provided dimension sizes <= 0")
    case ( DTFFT_ERROR_INVALID_COMM_TYPE )
      allocate(error_string, source="Invalid communicator type provided")
    case ( DTFFT_ERROR_INVALID_PRECISION )
      allocate(error_string, source="Invalid `precision` parameter provided")
    case ( DTFFT_ERROR_INVALID_EFFORT )
      allocate(error_string, source="Invalid `effort` parameter provided")
    case ( DTFFT_ERROR_INVALID_EXECUTOR )
      allocate(error_string, source="Invalid `executor` parameter provided")
    case ( DTFFT_ERROR_INVALID_COMM_DIMS )
      allocate(error_string, source="Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine")
    case ( DTFFT_ERROR_INVALID_COMM_FAST_DIM )
      allocate(error_string, source="Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1 OR provided dtfft_pencil_t with distribution over 1st dimension")
    case ( DTFFT_ERROR_MISSING_R2R_KINDS )
      allocate(error_string, source="For R2R plan, `kinds` parameter must be passed if `executor` != `DTFFT_EXECUTOR_NONE`")
    case ( DTFFT_ERROR_INVALID_R2R_KINDS )
      allocate(error_string, source="Invalid values detected in `kinds` parameter")
    case ( DTFFT_ERROR_R2C_TRANSPOSE_PLAN )
      allocate(error_string, source="Transpose plan is not supported in R2C, use R2R or C2C plan instead")
    case ( DTFFT_ERROR_INVALID_AUX )
      allocate(error_string, source="Invalid `aux` buffer provided")
    case ( DTFFT_ERROR_INPLACE_TRANSPOSE )
      allocate(error_string, source="Inplace transpose is not supported")
    case ( DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED )
      allocate(error_string, source="Selected `executor` do not support R2R FFTs")
    case ( DTFFT_ERROR_INVALID_LAYOUT )
      allocate(error_string, source="Invalid `layout` passed to `dtfft_get_pencil`")
    case ( DTFFT_ERROR_INVALID_USAGE )
      allocate(error_string, source="Invalid API Usage.")
    case ( DTFFT_ERROR_PLAN_IS_CREATED )
      allocate(error_string, source="Trying to create already created plan")
    case ( DTFFT_ERROR_ALLOC_FAILED )
      allocate(error_string, source="Allocation failed")
    case ( DTFFT_ERROR_FREE_FAILED )
      allocate(error_string, source="Memory free failed")
    case ( DTFFT_ERROR_INVALID_ALLOC_BYTES )
      allocate(error_string, source="Invalid `alloc_bytes` provided")
    case ( DTFFT_ERROR_DLOPEN_FAILED )
      allocate(error_string, source="Failed to open shared library. Set DTFFT_ENABLE_LOG=1 to see the error")
    case ( DTFFT_ERROR_DLSYM_FAILED )
      allocate(error_string, source="Failed to find symbol in shared library. Set DTFFT_ENABLE_LOG=1 to see the error")
    ! case ( DTFFT_ERROR_R2C_TRANSPOSE_CALLED )
    !   allocate(error_string, source="Calling to `transpose` method for R2C plan is not allowed")
    case ( DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH )
      allocate(error_string, source="Sizes of `lbound` and `sizes` arrays passed to `dtfft_pencil_t` constructor do not match")
    case ( DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES )
      allocate(error_string, source="Sizes of `lbound` and `sizes` < 2 or > 3 provided to `dtfft_pencil_t` constructor")
    case ( DTFFT_ERROR_PENCIL_INVALID_COUNTS )
      allocate(error_string, source="Invalid `counts` provided to `dtfft_pencil_t` constructor")
    case ( DTFFT_ERROR_PENCIL_INVALID_STARTS )
      allocate(error_string, source="Invalid `starts` provided to `dtfft_pencil_t` constructor")
    case ( DTFFT_ERROR_PENCIL_SHAPE_MISMATCH )
      allocate(error_string, source="Processes have same `starts` but different `counts` in some dimensions")
    case ( DTFFT_ERROR_PENCIL_OVERLAP )
      allocate(error_string, source="Pencil overlap detected, i.e. two processes share same part of global space")
    case ( DTFFT_ERROR_PENCIL_NOT_CONTINUOUS )
      allocate(error_string, source="Local pencils do not cover the global space without gaps")
    case ( DTFFT_ERROR_PENCIL_NOT_INITIALIZED )
      allocate(error_string, source="Pencil is not initialized, i.e. `constructor` subroutine was not called")
    case ( DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS )
      allocate(error_string, source="Invalid `n_measure_warmup_iters` provided")
    case ( DTFFT_ERROR_INVALID_MEASURE_ITERS )
      allocate(error_string, source="Invalid `n_measure_iters` provided")
    case ( DTFFT_ERROR_INVALID_REQUEST )
      allocate(error_string, source="Invalid `dtfft_request_t` provided")
    case ( DTFFT_ERROR_TRANSPOSE_ACTIVE )
      allocate(error_string, source="Attempting to execute already active transposition")
    case ( DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE )
      allocate(error_string, source="Attempting to finalize non-active transposition")
    case ( DTFFT_ERROR_INVALID_RESHAPE_TYPE )
      allocate(error_string, source="Invalid `reshape_type` provided")
    case ( DTFFT_ERROR_RESHAPE_ACTIVE )
      allocate(error_string, source="Attempting to execute already active reshape")
    case ( DTFFT_ERROR_RESHAPE_NOT_ACTIVE )
      allocate(error_string, source="Attempting to finalize non-active reshape")
    case ( DTFFT_ERROR_INPLACE_RESHAPE )
      allocate(error_string, source="Inplace reshape is not supported")
    ! case ( DTFFT_ERROR_R2C_RESHAPE_CALLED )
    !   allocate(error_string, source="R2C reshape was called")
    case ( DTFFT_ERROR_INVALID_EXECUTE_TYPE )
      allocate(error_string, source="Invalid `execute_type` provided")
    case ( DTFFT_ERROR_RESHAPE_NOT_SUPPORTED )
      allocate(error_string, source="Reshape is not supported for this plan")
    case ( DTFFT_ERROR_R2C_EXECUTE_CALLED )
      allocate(error_string, source="Execute called for transpose-only R2C Plan")
    case ( DTFFT_ERROR_INVALID_CART_COMM )
      allocate(error_string, source="Invalid cartesian communicator provided")
    case ( DTFFT_ERROR_GPU_INVALID_STREAM )
      allocate(error_string, source="Invalid stream provided")
    case ( DTFFT_ERROR_INVALID_BACKEND )
      allocate(error_string, source="Invalid backend provided")
    case ( DTFFT_ERROR_GPU_NOT_SET )
      allocate(error_string, source="Multiple MPI Processes located on same host share same GPU which is not supported")
    case ( DTFFT_ERROR_VKFFT_R2R_2D_PLAN )
      allocate(error_string, source="When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions")
    case ( DTFFT_ERROR_BACKENDS_DISABLED )
      allocate(error_string, source="Passed `effort` ==  `DTFFT_PATIENT` but all backends has been disabled by `dtfft_config_t`")
    case ( DTFFT_ERROR_NOT_DEVICE_PTR )
      allocate(error_string, source="One of pointers passed to `dtfft_execute` or `dtfft_transpose` cannot be accessed from device" )
    case ( DTFFT_ERROR_NOT_NVSHMEM_PTR )
      allocate(error_string, source="One of pointers passed to `dtfft_execute` or `dtfft_transpose` is not an `NVSHMEM` pointer" )
    case ( DTFFT_ERROR_INVALID_PLATFORM )
      allocate(error_string, source="Invalid platform provided")
    case ( DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR )
      allocate(error_string, source="Invalid executor provided for selected platform")
    case ( DTFFT_ERROR_INVALID_PLATFORM_BACKEND )
      allocate(error_string, source="Invalid backend provided for selected platform")
    case default
      allocate(error_string, source="Unknown error")
    endselect
  end function dtfft_get_error_string
end module dtfft_errors