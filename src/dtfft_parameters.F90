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
!! This module defines common DTFFT parameters
use iso_fortran_env, only: int8, int32, real32, real64
#include "dtfft_mpi.h"
#include "dtfft_private.h"
implicit none
private
public :: dtfft_get_error_string
#ifdef DTFFT_WITH_CUDA
public :: dtfft_get_gpu_backend_string
public :: is_backend_pipelined, is_backend_mpi, is_backend_nccl
#endif


!------------------------------------------------------------------------------------------------
! Traspose types
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_OUT          = CONF_DTFFT_TRANSPOSE_OUT
  !< Transpose out of "real" space to Fourier space
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_IN           = CONF_DTFFT_TRANSPOSE_IN
  !< Transpose into "real" space from Fourier space
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_X_TO_Y       = CONF_DTFFT_TRANSPOSE_X_TO_Y
  !< Perform single transposition, from X aligned to Y aligned
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_Y_TO_X       = CONF_DTFFT_TRANSPOSE_Y_TO_X
  !< Perform single transposition, from Y aligned to X aligned
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_X_TO_Z       = CONF_DTFFT_TRANSPOSE_X_TO_Z
  !< Perform single transposition, from X aligned to Z aligned
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_Y_TO_Z       = CONF_DTFFT_TRANSPOSE_Y_TO_Z
  !< Perform single transposition, from Y aligned to Z aligned
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_Z_TO_Y       = CONF_DTFFT_TRANSPOSE_Z_TO_Y
  !< Perform single transposition, from Z aligned to Y aligned
  integer(int8),  parameter,  public :: DTFFT_TRANSPOSE_Z_TO_X       = CONF_DTFFT_TRANSPOSE_Z_TO_X
  !< Perform single transposition, from Z aligned to X aligned

!------------------------------------------------------------------------------------------------
! External executor types
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_EXECUTOR_NONE          = CONF_DTFFT_EXECUTOR_NONE
  !< Do not setup any executor. If this type is provided, then execute method cannot be called.
  !< Use transpose method instead
#ifdef DTFFT_WITH_FFTW
  integer(int8),  parameter,  public :: DTFFT_EXECUTOR_FFTW3         = CONF_DTFFT_EXECUTOR_FFTW3
  !< FFTW3 executor
#endif
#ifdef DTFFT_WITH_MKL
  integer(int8),  parameter,  public :: DTFFT_EXECUTOR_MKL           = CONF_DTFFT_EXECUTOR_MKL
  !< MKL executor
#endif
#ifdef DTFFT_WITH_CUFFT
  integer(int8),  parameter,  public :: DTFFT_EXECUTOR_CUFFT         = CONF_DTFFT_EXECUTOR_CUFFT
  !< cuFFT executor
#endif
#ifdef DTFFT_WITH_VKFFT
  integer(int8),  parameter,  public :: DTFFT_EXECUTOR_VKFFT         = CONF_DTFFT_EXECUTOR_VKFFT
  !< KFR executor
#endif

!------------------------------------------------------------------------------------------------
! C2C Execution directions
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_FORWARD                = FFT_FORWARD
  !< Forward c2c transform
  integer(int8),  parameter,  public :: DTFFT_BACKWARD               = FFT_BACKWARD
  !< Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags.
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_ESTIMATE               = CONF_DTFFT_ESTIMATE
  !< Estimate flag. DTFFT will use default decomposition provided by MPI_Dims_create
  integer(int8),  parameter,  public :: DTFFT_MEASURE                = CONF_DTFFT_MEASURE
  !< Measure flag. DTFFT will run transpose routines to find the best grid decomposition.
  !< Passing this flag and MPI Communicator with cartesian topology to `plan%create` makes dtFFT do nothing.
  integer(int8),  parameter,  public :: DTFFT_PATIENT                = CONF_DTFFT_PATIENT
  !< Patient flag. Same as `DTFFT_MEASURE`, but different MPI datatypes will also be tested

!------------------------------------------------------------------------------------------------
! Precision flags
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_SINGLE                 = CONF_DTFFT_SINGLE
  !< Use single precision
  integer(int8),  parameter,  public :: DTFFT_DOUBLE                 = CONF_DTFFT_DOUBLE
  !< Use double precision

!------------------------------------------------------------------------------------------------
! R2R Transform kinds
! This parameters matches FFTW definitions. Hope they will never change there.
!------------------------------------------------------------------------------------------------
  integer(int8),  parameter,  public :: DTFFT_DCT_1                  = CONF_DTFFT_DCT_1
  !< DCT-I (Logical N=2*(n-1), inverse is `DTFFT_DCT_1` )
  integer(int8),  parameter,  public :: DTFFT_DCT_2                  = CONF_DTFFT_DCT_2
  !< DCT-II (Logical N=2*n, inverse is `DTFFT_DCT_3` )
  integer(int8),  parameter,  public :: DTFFT_DCT_3                  = CONF_DTFFT_DCT_3
  !< DCT-III (Logical N=2*n, inverse is `DTFFT_DCT_2` )
  integer(int8),  parameter,  public :: DTFFT_DCT_4                  = CONF_DTFFT_DCT_4
  !< DCT-IV (Logical N=2*n, inverse is `DTFFT_DCT_4` )
  integer(int8),  parameter,  public :: DTFFT_DST_1                  = CONF_DTFFT_DST_1
  !< DST-I (Logical N=2*(n+1), inverse is `DTFFT_DST_1` )
  integer(int8),  parameter,  public :: DTFFT_DST_2                  = CONF_DTFFT_DST_2
  !< DST-II (Logical N=2*n, inverse is `DTFFT_DST_3` )
  integer(int8),  parameter,  public :: DTFFT_DST_3                  = CONF_DTFFT_DST_3
  !< DST-III (Logical N=2*n, inverse is `DTFFT_DST_2` )
  integer(int8),  parameter,  public :: DTFFT_DST_4                  = CONF_DTFFT_DST_4
  !< DST-IV (Logical N=2*n, inverse is `DTFFT_DST_4` )

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(int8), parameter,  public :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._real64, 1._real64)) / 8_int8
  !< Number of bytes to store single double precision complex element
  integer(int8), parameter,  public :: COMPLEX_STORAGE_SIZE          = storage_size((1._real32, 1._real32)) / 8_int8
  !< Number of bytes to store single float precision complex element
  integer(int8), parameter,  public :: DOUBLE_STORAGE_SIZE           = storage_size(1._real64) / 8_int8
  !< Number of bytes to store single double precision real element
  integer(int8), parameter,  public :: FLOAT_STORAGE_SIZE            = storage_size(1._real32) / 8_int8
  !< Number of bytes to store single single precision real element

!------------------------------------------------------------------------------------------------
! Correct values for input parameters
!------------------------------------------------------------------------------------------------
  integer(int8),    parameter,  public :: VALID_FULL_TRANSPOSES(*) = [DTFFT_TRANSPOSE_OUT, DTFFT_TRANSPOSE_IN]
  integer(int8),    parameter,  public :: VALID_TRANSPOSES(*) = [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_Y, DTFFT_TRANSPOSE_X_TO_Z, DTFFT_TRANSPOSE_Z_TO_X]
  character(len=*), parameter,  public :: TRANSPOSE_NAMES(-3:3) = ["Z2X", "Z2Y", "Y2X", "NUL", "X2Y", "Y2Z", "X2Z"]
  integer(int8),    parameter,  public :: VALID_EFFORTS(*) = [DTFFT_ESTIMATE, DTFFT_MEASURE, DTFFT_PATIENT]
  integer(int8),    parameter,  public :: VALID_PRECISIONS(*) = [DTFFT_SINGLE, DTFFT_DOUBLE]
  integer(int8),    parameter,  public :: VALID_DIMENSIONS(*) = [2_int8, 3_int8]
  integer(int32),   parameter,  public :: VALID_COMMUNICATORS(*) = [MPI_UNDEFINED, MPI_CART]
  integer(int8),    parameter,  public :: VALID_R2R_FFTS(*) = [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_4, DTFFT_DST_1, DTFFT_DST_2, DTFFT_DST_3, DTFFT_DST_4]
  integer(int8),    parameter,  public :: VALID_EXECUTORS(*) = [DTFFT_EXECUTOR_NONE   &
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
  integer(int32), parameter,  public :: COLOR_TRANSPOSE_PALLETTE(-3:3) = [COLOR_TRANSPOSE_ZX, COLOR_TRANSPOSE_ZY, COLOR_TRANSPOSE_YX, 0, COLOR_TRANSPOSE_XY, COLOR_TRANSPOSE_YZ, COLOR_TRANSPOSE_XZ]

  integer(int32),  parameter,  public  :: DTFFT_SUCCESS = CONF_DTFFT_SUCCESS
  integer(int32),  parameter,  public  :: DTFFT_ERROR_MPI_FINALIZED = CONF_DTFFT_ERROR_MPI_FINALIZED
  integer(int32),  parameter,  public  :: DTFFT_ERROR_PLAN_NOT_CREATED = CONF_DTFFT_ERROR_PLAN_NOT_CREATED
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_TRANSPOSE_TYPE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_N_DIMENSIONS = CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_DIMENSION_SIZE = CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_TYPE = CONF_DTFFT_ERROR_INVALID_COMM_TYPE
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_PRECISION = CONF_DTFFT_ERROR_INVALID_PRECISION
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_EFFORT_FLAG = CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_EXECUTOR_TYPE = CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_DIMS = CONF_DTFFT_ERROR_INVALID_COMM_DIMS
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_FAST_DIM = CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM
  integer(int32),  parameter,  public  :: DTFFT_ERROR_MISSING_R2R_KINDS = CONF_DTFFT_ERROR_MISSING_R2R_KINDS
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_R2R_KINDS = CONF_DTFFT_ERROR_INVALID_R2R_KINDS
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2C_TRANSPOSE_PLAN = CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INPLACE_TRANSPOSE = CONF_DTFFT_ERROR_INPLACE_TRANSPOSE
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_AUX = CONF_DTFFT_ERROR_INVALID_AUX
  integer(int32),  parameter,  public  :: DTFFT_ERROR_INVALID_DIM = CONF_DTFFT_ERROR_INVALID_DIM
  integer(int32),  parameter,  public  :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
  ! integer(int32),  parameter,  public  :: DTFFT_ERROR_CUFFTMP_2D_PLAN = CONF_DTFFT_ERROR_CUFFTMP_2D_PLAN
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_INVALID_BACKEND = CONF_DTFFT_ERROR_GPU_INVALID_BACKEND
  integer(int32),  parameter,  public  :: DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET
  integer(int32),  parameter,  public  :: DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN
  integer(int32),  parameter,  public  :: DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR
 

#if (DTFFT_FORWARD_X_Y > 2) || (DTFFT_FORWARD_X_Y <= 0)
#error "Invalid DTFFT_FORWARD_X_Y parameter"
#endif
#if (DTFFT_BACKWARD_X_Y > 2) || (DTFFT_BACKWARD_X_Y <= 0)
#error "Invalid DTFFT_BACKWARD_X_Y parameter"
#endif
#if (DTFFT_FORWARD_Y_Z > 2) || (DTFFT_FORWARD_Y_Z <= 0)
#error "Invalid DTFFT_FORWARD_Y_Z parameter"
#endif
#if (DTFFT_BACKWARD_Y_Z > 2) || (DTFFT_BACKWARD_Y_Z <= 0)
#error "Invalid DTFFT_BACKWARD_Y_Z parameter"
#endif
#if (DTFFT_FORWARD_X_Z > 2) || (DTFFT_FORWARD_X_Z <= 0)
#error "Invalid DTFFT_FORWARD_X_Z parameter"
#endif
#if (DTFFT_BACKWARD_X_Z > 2) || (DTFFT_BACKWARD_X_Z <= 0)
#error "Invalid DTFFT_BACKWARD_X_Z parameter"
#endif

#ifdef DTFFT_WITH_CUDA
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_DATATYPE = CONF_DTFFT_GPU_BACKEND_MPI_DATATYPE
  !< Backend that uses MPI datatypes
  !< Not really recommended to use, since it is a million times slower than other backends
  !< Left here just to show how slow MPI Datatypes are for GPU usage
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_P2P = CONF_DTFFT_GPU_BACKEND_MPI_P2P
  !< MPI peer-to-peer algorithm
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_A2A = CONF_DTFFT_GPU_BACKEND_MPI_A2A
  !< MPI backend using MPI_Alltoallv
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_NCCL = CONF_DTFFT_GPU_BACKEND_NCCL
  !< NCCL backend
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED = CONF_DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED
  !< MPI peer-to-peer algorithm with overlapping data copying and unpacking
  integer(int8),  parameter,  public  :: DTFFT_GPU_BACKEND_NCCL_PIPELINED = CONF_DTFFT_GPU_BACKEND_NCCL_PIPELINED
  !< NCCL backend with overlapping data copying and unpacking
  integer(int8),  parameter,  public  :: BACKEND_NOT_SET = -1_int8
  integer(int8),  parameter           :: PIPELINED_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED, DTFFT_GPU_BACKEND_NCCL_PIPELINED]
  integer(int8),  parameter           :: MPI_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_P2P, DTFFT_GPU_BACKEND_MPI_A2A, DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED]
  integer(int8),  parameter           :: NCCL_BACKENDS(*) = [DTFFT_GPU_BACKEND_NCCL, DTFFT_GPU_BACKEND_NCCL_PIPELINED]
  integer(int8),  parameter,  public  :: VALID_GPU_BACKENDS(*) = [DTFFT_GPU_BACKEND_MPI_DATATYPE    &
  , DTFFT_GPU_BACKEND_MPI_P2P                                                                       &
  , DTFFT_GPU_BACKEND_MPI_A2A                                                                       &
  , DTFFT_GPU_BACKEND_NCCL                                                                          &
  ! , DTFFT_GPU_BACKEND_CUFFTMP                                                                       &
  , DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED                                                             &
  , DTFFT_GPU_BACKEND_NCCL_PIPELINED                                                                &
  ]
#endif

contains

  pure function dtfft_get_error_string(error_code) result(error_string)
  !! Gets the string description of an error code
    integer(int32),   intent(in)   :: error_code    !< Error code
    character(len=:), allocatable  :: error_string  !< Error string

    select case (error_code)
    case ( DTFFT_SUCCESS )
      allocate(error_string, source="DTFFT_SUCCESS")
    case ( DTFFT_ERROR_MPI_FINALIZED )
      allocate(error_string, source="MPI_Init is not called or MPI_Finalize has already been called")
    case ( DTFFT_ERROR_PLAN_NOT_CREATED)
      allocate(error_string, source="Plan not created")
    case ( DTFFT_ERROR_INVALID_TRANSPOSE_TYPE)
      allocate(error_string, source="Invalid `transpose_type` provided")
    case ( DTFFT_ERROR_INVALID_N_DIMENSIONS)
      allocate(error_string, source="Invalid Number of dimensions provided. Valid options are 2 and 3")
    case ( DTFFT_ERROR_INVALID_DIMENSION_SIZE)
      allocate(error_string, source="One or more provided dimension sizes <= 0")
    case ( DTFFT_ERROR_INVALID_COMM_TYPE)
      allocate(error_string, source="Invalid communicator type provided")
    case ( DTFFT_ERROR_INVALID_PRECISION )
      allocate(error_string, source="Invalid `precision` parameter provided")
    case ( DTFFT_ERROR_INVALID_EFFORT_FLAG )
      allocate(error_string, source="Invalid `effort_flag` parameter provided")
    case ( DTFFT_ERROR_INVALID_EXECUTOR_TYPE )
      allocate(error_string, source="Invalid `executor_type` parameter provided")
    case ( DTFFT_ERROR_INVALID_COMM_DIMS )
      allocate(error_string, source="Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine")
    case ( DTFFT_ERROR_INVALID_COMM_FAST_DIM )
      allocate(error_string, source="Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1")
    case ( DTFFT_ERROR_MISSING_R2R_KINDS )
      allocate(error_string, source="For R2R plan, `kinds` parameter must be passed if `executor_type` != `DTFFT_EXECUTOR_NONE`")
    case ( DTFFT_ERROR_INVALID_R2R_KINDS )
      allocate(error_string, source="Invalid values detected in `kinds` parameter")
    case ( DTFFT_ERROR_R2C_TRANSPOSE_PLAN )
      allocate(error_string, source="Transpose plan is not supported in R2C, use R2R or C2C plan instead")
    case ( DTFFT_ERROR_INVALID_AUX )
      allocate(error_string, source="Invalid `aux` buffer provided")
    case ( DTFFT_ERROR_INPLACE_TRANSPOSE )
      allocate(error_string, source="Inplace transpose is not supported")
    case ( DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED )
      allocate(error_string, source="Selected `executor_type` does not support R2R FFTs")
    case ( DTFFT_ERROR_INVALID_DIM )
      allocate(error_string, source="Invalid `dim` passed to `dtfft_get_pencil`")
    ! case ( DTFFT_ERROR_CUFFTMP_2D_PLAN )
    !   allocate(error_string, source="cufftMp backends support only 3d plan")
    case ( DTFFT_ERROR_GPU_INVALID_STREAM )
      allocate(error_string, source="Invalid stream provided")
    case ( DTFFT_ERROR_GPU_INVALID_BACKEND )
      allocate(error_string, source="Invalid GPU backend provided")
    case ( DTFFT_ERROR_GPU_NOT_SET )
      allocate(error_string, source="Multiple MPI Processes located on same host share same GPU which is not supported")
    case ( DTFFT_ERROR_VKFFT_R2R_2D_PLAN )
      allocate(error_string, source="When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions")
    case ( DTFFT_ERROR_NOT_DEVICE_PTR )
      allocate(error_string, source="Pointer passed to `dtfft_execute` or `dtfft_transpose` is not device nor managed" )
    case default
      allocate(error_string, source="Unknown error")
    endselect
  end function dtfft_get_error_string

#ifdef DTFFT_WITH_CUDA
  function dtfft_get_gpu_backend_string(backend_id) result(string)
  !! Gets the string description of a GPU backend
    integer(int8),    intent(in)  :: backend    !< GPU backend
    character(len=:), allocatable :: string     !< Backend string

    select case ( backend_id )
    case ( DTFFT_GPU_BACKEND_MPI_DATATYPE )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_DATATYPE")
    case ( DTFFT_GPU_BACKEND_MPI_P2P )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_P2P")
    case ( DTFFT_GPU_BACKEND_MPI_A2A )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_A2A")
    case ( DTFFT_GPU_BACKEND_NCCL )
      allocate(string, source="DTFFT_GPU_BACKEND_NCCL")
    ! case ( DTFFT_GPU_BACKEND_CUFFTMP )
    !   allocate(string, source="DTFFT_GPU_BACKEND_CUFFTMP")
    case ( DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED )
      allocate(string, source="DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED")
    case ( DTFFT_GPU_BACKEND_NCCL_PIPELINED)
      allocate(string, source="DTFFT_GPU_BACKEND_NCCL_PIPELINED")
    case ( BACKEND_NOT_SET )
      allocate(string, source="Backend is not set, executing on 1 process" )
    case default
      allocate(string, source="Unknown backend")
    endselect
  end function dtfft_get_gpu_backend_string

  elemental logical function is_backend_pipelined(backend_id)
    integer(int8),  intent(in)  :: backend_id
    is_backend_pipelined = any(backend_id == PIPELINED_BACKENDS)
  end function is_backend_pipelined

  logical function is_backend_mpi(backend_id)
    integer(int8),  intent(in)  :: backend_id
    is_backend_mpi = any(backend_id == MPI_BACKENDS)
  end function is_backend_mpi

  elemental logical function is_backend_nccl(backend_id)
    integer(int8),  intent(in)  :: backend_id
    is_backend_nccl = any(backend_id == NCCL_BACKENDS)
  end function is_backend_nccl
#endif
end module dtfft_parameters