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
module dtfft_parameters
!! This module defines common ``dtFFT`` parameters
use iso_c_binding,    only: c_int32_t, c_null_ptr, c_ptr
use iso_fortran_env,  only: int8, int32, int64, real32, real64
#include "dtfft_mpi.h"
#include "dtfft_private.h"
implicit none
private
public :: dtfft_execute_type_t, dtfft_transpose_type_t
public :: dtfft_executor_t, dtfft_effort_t
public :: dtfft_precision_t, dtfft_r2r_kind_t
public :: is_valid_execute_type, is_valid_transpose_type
public :: is_valid_executor, is_valid_effort
public :: is_valid_precision, is_valid_r2r_kind
public :: is_valid_dimension, is_valid_comm_type
public :: dtfft_get_error_string
public :: dtfft_get_version
public :: is_host_executor, is_cuda_executor
public :: is_valid_platform
#ifdef DTFFT_WITH_CUDA
public :: dtfft_gpu_backend_t
public :: dtfft_get_gpu_backend_string
public :: is_valid_gpu_backend, is_backend_pipelined, is_backend_mpi, is_backend_nccl, is_backend_nvshmem
public :: dtfft_stream_t
#endif

  integer(int32), parameter, public :: DTFFT_VERSION_MAJOR = CONF_DTFFT_VERSION_MAJOR
  !! dtFFT Major Version
  integer(int32), parameter, public :: DTFFT_VERSION_MINOR = CONF_DTFFT_VERSION_MINOR
  !! dtFFT Minor Version
  integer(int32), parameter, public :: DTFFT_VERSION_PATCH = CONF_DTFFT_VERSION_PATCH
  !! dtFFT Patch Version
  integer(int32), parameter, public :: DTFFT_VERSION_CODE  = CONF_DTFFT_VERSION_CODE
  !! dtFFT Version Code. Can be used in Version comparison

  interface dtfft_get_version
    module procedure dtfft_get_version_current
    module procedure dtfft_get_version_required
  end interface dtfft_get_version

!------------------------------------------------------------------------------------------------
! Execute types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_execute_type_t
  !! Type that is used during call to `execute` method
    integer(c_int32_t) :: val
  end type dtfft_execute_type_t

  type(dtfft_execute_type_t), parameter, public :: DTFFT_EXECUTE_FORWARD = dtfft_execute_type_t(CONF_DTFFT_TRANSPOSE_OUT)
  !! Transpose out of "real" space to Fourier space
  type(dtfft_execute_type_t), parameter, public :: DTFFT_EXECUTE_BACKWARD  = dtfft_execute_type_t(CONF_DTFFT_TRANSPOSE_IN)
  !! Transpose into "real" space from Fourier space
  type(dtfft_execute_type_t), parameter :: VALID_EXECUTE_TYPES(*) = [DTFFT_EXECUTE_FORWARD, DTFFT_EXECUTE_BACKWARD]

!------------------------------------------------------------------------------------------------
! Transpose types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_transpose_type_t
  !! Type that is used during call to [[dtfft_plan_t(type):transpose]] method
    integer(c_int32_t) :: val
  end type dtfft_transpose_type_t

  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_X_TO_Y = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_X_TO_Y)
  !! Perform single transposition, from X aligned to Y aligned
  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_Y_TO_X = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_Y_TO_X)
  !! Perform single transposition, from Y aligned to X aligned
  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_X_TO_Z = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_X_TO_Z)
  !! Perform single transposition, from X aligned to Z aligned
  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_Y_TO_Z = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_Y_TO_Z)
  !! Perform single transposition, from Y aligned to Z aligned
  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_Z_TO_Y = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_Z_TO_Y)
  !! Perform single transposition, from Z aligned to Y aligned
  type(dtfft_transpose_type_t), parameter,  public :: DTFFT_TRANSPOSE_Z_TO_X = dtfft_transpose_type_t(CONF_DTFFT_TRANSPOSE_Z_TO_X)
  !! Perform single transposition, from Z aligned to X aligned
  type(dtfft_transpose_type_t), parameter :: VALID_TRANSPOSE_TYPES(*) = [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_Y, DTFFT_TRANSPOSE_X_TO_Z, DTFFT_TRANSPOSE_Z_TO_X]
  !! Types of transpose that are valid to pass to `transpose` method
  character(len=*), parameter,  public :: TRANSPOSE_NAMES(-3:3) = ["Z2X", "Z2Y", "Y2X", "NUL", "X2Y", "Y2Z", "X2Z"]
  !! String representation of `dtfft_transpose_type_t`

!------------------------------------------------------------------------------------------------
! External FFT executor types
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_executor_t
  !! Type that specifies external FFT executor
    integer(c_int32_t) :: val
  end type dtfft_executor_t

  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_NONE   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_NONE)
  !! Do not setup any executor. If this type is provided, then `execute` method should not be called.
  !! Use `transpose` method instead
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_FFTW3 = dtfft_executor_t(CONF_DTFFT_EXECUTOR_FFTW3)
  !! FFTW3 executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_MKL   = dtfft_executor_t(CONF_DTFFT_EXECUTOR_MKL)
  !! MKL executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_CUFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_CUFFT)
  !! cuFFT GPU executor
  type(dtfft_executor_t),  parameter,  public :: DTFFT_EXECUTOR_VKFFT = dtfft_executor_t(CONF_DTFFT_EXECUTOR_VKFFT)
  !! VkFFT GPU executor
  type(dtfft_executor_t),  parameter  :: VALID_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE    &
#ifdef DTFFT_WITH_FFTW
    ,DTFFT_EXECUTOR_FFTW3                                                             &
#endif
#ifdef DTFFT_WITH_MKL
    ,DTFFT_EXECUTOR_MKL                                                               &
#endif
#ifdef DTFFT_WITH_CUFFT
    ,DTFFT_EXECUTOR_CUFFT                                                             &
#endif
#ifdef DTFFT_WITH_VKFFT
    ,DTFFT_EXECUTOR_VKFFT                                                             &
#endif
  ]

  type(dtfft_executor_t),  parameter        :: HOST_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_FFTW3, DTFFT_EXECUTOR_MKL]
  type(dtfft_executor_t),  parameter        :: CUDA_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE, DTFFT_EXECUTOR_CUFFT, DTFFT_EXECUTOR_VKFFT]

!------------------------------------------------------------------------------------------------
! FFT Execution directions
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: FFT_FORWARD   = CONF_FFT_FORWARD
  !! Forward c2c transform
  integer(int8),  parameter,  public :: FFT_BACKWARD  = CONF_FFT_BACKWARD
  !! Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags.
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_effort_t
  !! Type that specifies effort that dtFFT should use when creating plan
    integer(c_int32_t) :: val
  end type dtfft_effort_t

  type(dtfft_effort_t), parameter,  public :: DTFFT_ESTIMATE = dtfft_effort_t(CONF_DTFFT_ESTIMATE)
  !! Estimate flag. ``dtFFT`` will use default decomposition provided by MPI_Dims_create
  type(dtfft_effort_t), parameter,  public :: DTFFT_MEASURE  = dtfft_effort_t(CONF_DTFFT_MEASURE)
  !! Measure flag. ``dtFFT`` will run transpose routines to find the best grid decomposition.
  !! Passing this flag and MPI Communicator with Cartesian topology to `plan%create` makes dtFFT do nothing.
  type(dtfft_effort_t), parameter,  public :: DTFFT_PATIENT  = dtfft_effort_t(CONF_DTFFT_PATIENT)
  !! Patient flag. Same as `DTFFT_MEASURE`, but different MPI datatypes will also be tested
  type(dtfft_effort_t), parameter :: VALID_EFFORTS(*) = [DTFFT_ESTIMATE, DTFFT_MEASURE, DTFFT_PATIENT]

!------------------------------------------------------------------------------------------------
! Precision flags
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_precision_t
  !! Type that specifies precision of dtFFT plan
    integer(c_int32_t) :: val
  end type dtfft_precision_t

  type(dtfft_precision_t),  parameter,  public :: DTFFT_SINGLE = dtfft_precision_t(CONF_DTFFT_SINGLE)
  !! Use single precision
  type(dtfft_precision_t),  parameter,  public :: DTFFT_DOUBLE = dtfft_precision_t(CONF_DTFFT_DOUBLE)
  !! Use double precision
  type(dtfft_precision_t),  parameter :: VALID_PRECISIONS(*) = [DTFFT_SINGLE, DTFFT_DOUBLE]

!------------------------------------------------------------------------------------------------
! R2R Transform kinds
! This parameters matches FFTW definitions. Hope they will never change there.
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_r2r_kind_t
  !! Type that specifies various kinds of R2R FFTs
    integer(c_int32_t) :: val
  end type dtfft_r2r_kind_t

  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_1 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_1)
  !! DCT-I (Logical N=2*(n-1), inverse is `DTFFT_DCT_1`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_2 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_2)
  !! DCT-II (Logical N=2*n, inverse is `DTFFT_DCT_3`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_3 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_3)
  !! DCT-III (Logical N=2*n, inverse is `DTFFT_DCT_2`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DCT_4 = dtfft_r2r_kind_t(CONF_DTFFT_DCT_4)
  !! DCT-IV (Logical N=2*n, inverse is `DTFFT_DCT_4`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_1 = dtfft_r2r_kind_t(CONF_DTFFT_DST_1)
  !! DST-I (Logical N=2*(n+1), inverse is `DTFFT_DST_1`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_2 = dtfft_r2r_kind_t(CONF_DTFFT_DST_2)
  !! DST-II (Logical N=2*n, inverse is `DTFFT_DST_3`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_3 = dtfft_r2r_kind_t(CONF_DTFFT_DST_3)
  !! DST-III (Logical N=2*n, inverse is `DTFFT_DST_2`)
  type(dtfft_r2r_kind_t),   parameter,  public :: DTFFT_DST_4 = dtfft_r2r_kind_t(CONF_DTFFT_DST_4)
  !! DST-IV (Logical N=2*n, inverse is `DTFFT_DST_4`)
  type(dtfft_r2r_kind_t),   parameter :: VALID_R2R_KINDS(*) = [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_4, DTFFT_DST_1, DTFFT_DST_2, DTFFT_DST_3, DTFFT_DST_4]


public :: operator(==)
  interface operator(==)
    module procedure execute_type_eq
    module procedure transpose_type_eq
    module procedure executor_eq
    module procedure effort_eq
    module procedure precision_eq
    module procedure r2r_kind_eq
    module procedure platform_eq
#ifdef DTFFT_WITH_CUDA
    module procedure gpu_backend_eq
#endif
  end interface

public :: operator(/=)
  interface operator(/=)
    module procedure execute_type_ne
    module procedure transpose_type_ne
    module procedure executor_ne
    module procedure effort_ne
    module procedure precision_ne
    module procedure r2r_kind_ne
    module procedure platform_ne
#ifdef DTFFT_WITH_CUDA
    module procedure gpu_backend_ne
#endif
  end interface

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(int8), parameter,  public :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._real64, 1._real64)) / 8_int8
  !! Number of bytes to store single double precision complex element
  integer(int8), parameter,  public :: COMPLEX_STORAGE_SIZE          = storage_size((1._real32, 1._real32)) / 8_int8
  !! Number of bytes to store single float precision complex element
  integer(int8), parameter,  public :: DOUBLE_STORAGE_SIZE           = storage_size(1._real64) / 8_int8
  !! Number of bytes to store single double precision real element
  integer(int8), parameter,  public :: FLOAT_STORAGE_SIZE            = storage_size(1._real32) / 8_int8
  !! Number of bytes to store single single precision real element


  integer(int8),    parameter :: VALID_DIMENSIONS(*) = [2_int8, 3_int8]
  integer(int32),   parameter :: VALID_COMM_TYPES(*) = [MPI_UNDEFINED, MPI_CART]


  integer(int32), parameter,  public :: COLOR_CREATE        = int(Z'00FAB53C')
  integer(int32), parameter,  public :: COLOR_EXECUTE       = int(Z'00E25DFC')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE     = int(Z'00B175BD')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_XY  = int(Z'005DFCCA')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_YX  = int(Z'0076A797')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_YZ  = int(Z'00E3CF9F')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_ZY  = int(Z'008C826A')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_XZ  = int(Z'00546F66')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_ZX  = int(Z'007A6D7D')
  integer(int32), parameter,  public :: COLOR_FFT           = int(Z'00FCD05D')
  integer(int32), parameter,  public :: COLOR_AUTOTUNE      = int(Z'006075FF')
  integer(int32), parameter,  public :: COLOR_AUTOTUNE2     = int(Z'0056E874')
  integer(int32), parameter,  public :: COLOR_DESTROY       = int(Z'00000000')
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_PALLETTE(-3:3) = [COLOR_TRANSPOSE_ZX, COLOR_TRANSPOSE_ZY, COLOR_TRANSPOSE_YX, 0, COLOR_TRANSPOSE_XY, COLOR_TRANSPOSE_YZ, COLOR_TRANSPOSE_XZ]

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
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_DIM = CONF_DTFFT_ERROR_INVALID_DIM
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
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
  !! Selected `executor` do not support R2R FFTs
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM
  !! Invalid stream provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_INVALID_BACKEND = CONF_DTFFT_ERROR_GPU_INVALID_BACKEND
  !! Invalid GPU backend provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET
  !! Multiple MPI Processes located on same host share same GPU which is not supported
  integer(int32),  parameter,  public  :: DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN
  !! When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_BACKENDS_DISABLED = CONF_DTFFT_ERROR_GPU_BACKENDS_DISABLED
  !! Passed `effort` ==  `DTFFT_PATIENT` but all GPU Backends has been disabled by `dtfft_config_t` */
  integer(int32),  parameter,  public  :: DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR
  !! One of pointers passed to `plan.execute` or `plan.transpose` cannot be accessed from device
  integer(int32),  parameter,  public  :: DTFFT_ERROR_NOT_NVSHMEM_PTR = CONF_DTFFT_ERROR_NOT_NVSHMEM_PTR
  !! One of pointers passed to `plan.execute` or `plan.transpose` is not and `NVSHMEM` pointer
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PLATFORM = CONF_DTFFT_ERROR_INVALID_PLATFORM
  !! Invalid platform provided
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE = CONF_DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE
  !! Invalid executor provided for selected platform

#ifdef DTFFT_WITH_CUDA

!------------------------------------------------------------------------------------------------
! GPU Backends that are responsible for transfering data across GPUs
!------------------------------------------------------------------------------------------------
  type, bind(C) :: dtfft_gpu_backend_t
  !! Type that specifies various GPU Backend present in dtFFT
    integer(c_int32_t) :: val
  end type dtfft_gpu_backend_t


  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_DATATYPE = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_MPI_DATATYPE)
  !! Backend that uses MPI datatypes
  !! Not really recommended to use, since it is a million times slower than other backends
  !! Left here just to show how slow MPI Datatypes are for GPU usage
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_P2P = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_MPI_P2P)
  !! MPI peer-to-peer algorithm
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_A2A = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_MPI_A2A)
  !! MPI backend using MPI_Alltoallv
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_NCCL = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_NCCL)
  !! NCCL backend
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED)
  !! MPI peer-to-peer algorithm with overlapping data copying and unpacking
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_NCCL_PIPELINED = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_NCCL_PIPELINED)
  !! NCCL backend with overlapping data copying and unpacking
  type(dtfft_gpu_backend_t),  parameter,  public  :: DTFFT_GPU_BACKEND_CUFFTMP = dtfft_gpu_backend_t(CONF_DTFFT_GPU_BACKEND_CUFFTMP)
  !! cuFFTMp backend
  type(dtfft_gpu_backend_t),  parameter,  public  :: BACKEND_NOT_SET = dtfft_gpu_backend_t(-1_int8)
  type(dtfft_gpu_backend_t),  parameter :: PIPELINED_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED, DTFFT_GPU_BACKEND_NCCL_PIPELINED]
  type(dtfft_gpu_backend_t),  parameter :: MPI_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_P2P, DTFFT_GPU_BACKEND_MPI_A2A, DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED]
  type(dtfft_gpu_backend_t),  parameter :: NCCL_BACKENDS(*) = [DTFFT_GPU_BACKEND_NCCL, DTFFT_GPU_BACKEND_NCCL_PIPELINED]
  type(dtfft_gpu_backend_t),  parameter :: NVSHMEM_BACKENDS(*) = [DTFFT_GPU_BACKEND_CUFFTMP]

  type(dtfft_gpu_backend_t),  parameter,  public :: VALID_GPU_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_DATATYPE         &
                                                                             ,DTFFT_GPU_BACKEND_MPI_P2P             &
                                                                             ,DTFFT_GPU_BACKEND_MPI_A2A             &
                                                                             ,DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED   &
#ifdef DTFFT_WITH_NCCL
                                                                             ,DTFFT_GPU_BACKEND_NCCL_PIPELINED      &
                                                                             ,DTFFT_GPU_BACKEND_NCCL                &
#endif
#ifdef DTFFT_WITH_NVSHMEM
                                                                             ,DTFFT_GPU_BACKEND_CUFFTMP             &
#endif
                                                                             ]

  type, bind(C) :: dtfft_stream_t
  !! `dtFFT` stream representation.
    type(c_ptr) :: stream
      !! Actual stream
  end type dtfft_stream_t

  type(dtfft_stream_t), parameter,  public :: NULL_STREAM = dtfft_stream_t(c_null_ptr)

  interface dtfft_stream_t
    module procedure stream_from_int64
    module procedure stream_from_ptr
  end interface dtfft_stream_t
#endif

public :: dtfft_platform_t
  type, bind(C) :: dtfft_platform_t
  !! Type that specifies runtime platform, e.g. Host, CUDA, HIP
    integer(c_int32_t) :: val
  end type dtfft_platform_t

  type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_HOST = dtfft_platform_t(CONF_DTFFT_PLATFORM_HOST)
  !! Host
  type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_CUDA = dtfft_platform_t(CONF_DTFFT_PLATFORM_CUDA)
  !! CUDA
  type(dtfft_platform_t),         parameter :: VALID_PLATFORMS(*) = [DTFFT_PLATFORM_HOST, DTFFT_PLATFORM_CUDA]

  ! type(dtfft_platform_t), public, parameter :: DTFFT_PLATFORM_HIP = dtfft_platform_t(3)

  type(dtfft_platform_t), public, parameter :: PLATFORM_UNDEFINED = dtfft_platform_t(-1)

#define MAKE_EQ_FUN(datatype, name)                         \
  pure elemental function name(left, right) result(res);    \
    type(datatype), intent(in) :: left;                     \
    type(datatype), intent(in) :: right;                    \
    logical :: res;                                         \
    res = left%val == right%val;                            \
  end function name

#define MAKE_NE_FUN(datatype, name)                         \
  pure elemental function name(left, right) result(res);    \
    type(datatype), intent(in) :: left;                     \
    type(datatype), intent(in) :: right;                    \
    logical :: res;                                         \
    res = left%val /= right%val;                            \
  end function name

#define MAKE_VALID_FUN(type, name, valid_values)            \
  pure elemental function name(param) result(res);          \
    type, intent(in)  :: param;                             \
    logical :: res;                                         \
    res = any(param == valid_values);                       \
  end function name

#define MAKE_VALID_FUN_DTYPE(datatype, name, valid_values)  \
  MAKE_VALID_FUN(type(datatype), name, valid_values)

contains

MAKE_EQ_FUN(dtfft_execute_type_t, execute_type_eq)
MAKE_EQ_FUN(dtfft_transpose_type_t, transpose_type_eq)
MAKE_EQ_FUN(dtfft_executor_t, executor_eq)
MAKE_EQ_FUN(dtfft_effort_t, effort_eq)
MAKE_EQ_FUN(dtfft_precision_t, precision_eq)
MAKE_EQ_FUN(dtfft_r2r_kind_t, r2r_kind_eq)
MAKE_EQ_FUN(dtfft_platform_t, platform_eq)


MAKE_NE_FUN(dtfft_execute_type_t, execute_type_ne)
MAKE_NE_FUN(dtfft_transpose_type_t, transpose_type_ne)
MAKE_NE_FUN(dtfft_executor_t, executor_ne)
MAKE_NE_FUN(dtfft_effort_t, effort_ne)
MAKE_NE_FUN(dtfft_precision_t, precision_ne)
MAKE_NE_FUN(dtfft_r2r_kind_t, r2r_kind_ne)
MAKE_NE_FUN(dtfft_platform_t, platform_ne)

MAKE_VALID_FUN_DTYPE(dtfft_execute_type_t, is_valid_execute_type, VALID_EXECUTE_TYPES)
MAKE_VALID_FUN_DTYPE(dtfft_transpose_type_t, is_valid_transpose_type, VALID_TRANSPOSE_TYPES)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_valid_executor, VALID_EXECUTORS)
MAKE_VALID_FUN_DTYPE(dtfft_effort_t, is_valid_effort, VALID_EFFORTS)
MAKE_VALID_FUN_DTYPE(dtfft_precision_t, is_valid_precision, VALID_PRECISIONS)
MAKE_VALID_FUN_DTYPE(dtfft_r2r_kind_t, is_valid_r2r_kind, VALID_R2R_KINDS)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_host_executor, HOST_EXECUTORS)
MAKE_VALID_FUN_DTYPE(dtfft_executor_t, is_cuda_executor, CUDA_EXECUTORS)
MAKE_VALID_FUN_DTYPE(dtfft_platform_t, is_valid_platform, VALID_PLATFORMS)

MAKE_VALID_FUN(integer(int8), is_valid_dimension, VALID_DIMENSIONS)
MAKE_VALID_FUN(integer(int32), is_valid_comm_type, VALID_COMM_TYPES)

  integer(c_int32_t) function dtfft_get_version_current() bind(C)
    dtfft_get_version_current = DTFFT_VERSION_CODE
  end function dtfft_get_version_current

  integer(int32) function dtfft_get_version_required(major, minor, patch)
    integer(int32), intent(in) :: major
    integer(int32), intent(in) :: minor
    integer(int32), intent(in) :: patch

    dtfft_get_version_required = CONF_DTFFT_VERSION(major, minor, patch)
  end function dtfft_get_version_required

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
      allocate(error_string, source="Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1")
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
    case ( DTFFT_ERROR_INVALID_DIM )
      allocate(error_string, source="Invalid `dim` passed to `dtfft_get_pencil`")
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
    case ( DTFFT_ERROR_GPU_INVALID_STREAM )
      allocate(error_string, source="Invalid stream provided")
    case ( DTFFT_ERROR_GPU_INVALID_BACKEND )
      allocate(error_string, source="Invalid GPU backend provided")
    case ( DTFFT_ERROR_GPU_NOT_SET )
      allocate(error_string, source="Multiple MPI Processes located on same host share same GPU which is not supported")
    case ( DTFFT_ERROR_VKFFT_R2R_2D_PLAN )
      allocate(error_string, source="When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions")
    case ( DTFFT_ERROR_GPU_BACKENDS_DISABLED )
      allocate(error_string, source="Passed `effort` ==  `::DTFFT_PATIENT` but all GPU Backends has been disabled by `dtfft_config_t`")
    case ( DTFFT_ERROR_NOT_DEVICE_PTR )
      allocate(error_string, source="One of pointers passed to `dtfft_execute` or `dtfft_transpose` cannot be accessed from device" )
    case default
      allocate(error_string, source="Unknown error")
    endselect
  end function dtfft_get_error_string

#ifdef DTFFT_WITH_CUDA
  MAKE_EQ_FUN(dtfft_gpu_backend_t, gpu_backend_eq)
  MAKE_NE_FUN(dtfft_gpu_backend_t, gpu_backend_ne)
  MAKE_VALID_FUN_DTYPE(dtfft_gpu_backend_t, is_valid_gpu_backend, VALID_GPU_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_gpu_backend_t, is_backend_pipelined, PIPELINED_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_gpu_backend_t, is_backend_mpi, MPI_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_gpu_backend_t, is_backend_nccl, NCCL_BACKENDS)
  MAKE_VALID_FUN_DTYPE(dtfft_gpu_backend_t, is_backend_nvshmem, NVSHMEM_BACKENDS)

  function dtfft_get_gpu_backend_string(gpu_backend) result(string)
  !! Gets the string description of a GPU backend
    type(dtfft_gpu_backend_t),  intent(in)  :: gpu_backend !! GPU backend
    character(len=:),           allocatable :: string     !! Backend string

    select case ( gpu_backend%val )
    case ( DTFFT_GPU_BACKEND_MPI_DATATYPE%val )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_DATATYPE")
    case ( DTFFT_GPU_BACKEND_MPI_P2P%val )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_P2P")
    case ( DTFFT_GPU_BACKEND_MPI_A2A%val )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_A2A")
    case ( DTFFT_GPU_BACKEND_NCCL%val )
      allocate(string, source="DTFFT_GPU_BACKEND_NCCL")
    case ( DTFFT_GPU_BACKEND_CUFFTMP%val )
      allocate(string, source="DTFFT_GPU_BACKEND_CUFFTMP")
    case ( DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED%val )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED")
    case ( DTFFT_GPU_BACKEND_NCCL_PIPELINED%val )
      allocate(string, source="DTFFT_GPU_BACKEND_NCCL_PIPELINED")
    case ( BACKEND_NOT_SET%val )
      allocate(string, source="Backend is not set, executing on 1 process" )
    case default
      allocate(string, source="Unknown backend")
    endselect
  end function dtfft_get_gpu_backend_string

  function stream_from_int64(val) result(stream)
    integer(int64), intent(in)  :: val
    type(dtfft_stream_t)        :: stream

    stream = transfer(val, stream)
  end function stream_from_int64

  function stream_from_ptr(val) result(stream)
    type(c_ptr),    intent(in)  :: val
    type(dtfft_stream_t)        :: stream

    stream%stream = val
  end function stream_from_ptr
#endif
end module dtfft_parameters