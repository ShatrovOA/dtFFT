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
#include "dtfft_private.h"
module dtfft_parameters
!------------------------------------------------------------------------------------------------
!< This module defines common DTFFT parameters
!------------------------------------------------------------------------------------------------
use dtfft_precisions
#include "dtfft_mpi.h"
implicit none
private
public :: dtfft_get_error_string

!------------------------------------------------------------------------------------------------
! Traspose types
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_OUT          = CONF_DTFFT_TRANSPOSE_OUT
  !< Transpose out of "real" space to Fourier space
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_IN           = CONF_DTFFT_TRANSPOSE_IN
  !< Transpose into "real" space from Fourier space
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_X_TO_Y       = CONF_DTFFT_TRANSPOSE_X_TO_Y
  !< Perform single transposition, from X aligned to Y aligned
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_Y_TO_X       = CONF_DTFFT_TRANSPOSE_Y_TO_X
  !< Perform single transposition, from Y aligned to X aligned
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_X_TO_Z       = CONF_DTFFT_TRANSPOSE_X_TO_Z
  !< Perform single transposition, from X aligned to Z aligned
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_Y_TO_Z       = CONF_DTFFT_TRANSPOSE_Y_TO_Z
  !< Perform single transposition, from Y aligned to Z aligned
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_Z_TO_Y       = CONF_DTFFT_TRANSPOSE_Z_TO_Y
  !< Perform single transposition, from Z aligned to Y aligned
  integer(IP),  parameter,  public :: DTFFT_TRANSPOSE_Z_TO_X       = CONF_DTFFT_TRANSPOSE_Z_TO_X
  !< Perform single transposition, from Z aligned to X aligned
!------------------------------------------------------------------------------------------------
! External executor types
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_EXECUTOR_NONE          = CONF_DTFFT_EXECUTOR_NONE
  !< Do not setup any executor. If this type is provided, then execute method cannot be called.
  !< Use transpose method instead
#ifndef DTFFT_WITHOUT_FFTW
  integer(IP),  parameter,  public :: DTFFT_EXECUTOR_FFTW3         = CONF_DTFFT_EXECUTOR_FFTW3
  !< FFTW3 executor
#endif

#ifdef DTFFT_WITH_MKL
  integer(IP),  parameter,  public :: DTFFT_EXECUTOR_MKL           = CONF_DTFFT_EXECUTOR_MKL
  !< MKL executor
#endif
#ifdef DTFFT_WITH_CUFFT
  integer(IP),  parameter,  public :: DTFFT_EXECUTOR_CUFFT         = CONF_DTFFT_EXECUTOR_CUFFT
  !< cuFFT executor
#endif
! #ifdef DTFFT_WITH_KFR
!   integer(IP),  parameter,  public :: DTFFT_EXECUTOR_KFR           = CONF_DTFFT_EXECUTOR_KFR
!   !< KFR executor
! #endif
#ifdef DTFFT_WITH_VKFFT
  integer(IP),  parameter,  public :: DTFFT_EXECUTOR_VKFFT         = CONF_DTFFT_EXECUTOR_VKFFT
  !< KFR executor
#endif

!------------------------------------------------------------------------------------------------
! C2C Execution directions
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_FORWARD                = FFT_FORWARD
  !< Forward c2c transform
  integer(IP),  parameter,  public :: DTFFT_BACKWARD               = FFT_BACKWARD
  !< Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags. Reserved for future. Unused
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_ESTIMATE               = CONF_DTFFT_ESTIMATE
  !< Estimate flag. DTFFT will use default decomposition provided by MPI_Dims_create
  integer(IP),  parameter,  public :: DTFFT_MEASURE                = CONF_DTFFT_MEASURE
  !< Measure flag. DTFFT will run transpose routines to find the best grid decomposition.
  !< Passing this flag and MPI Communicator with cartesian topology to `plan%create` makes dtFFT do nothing.
  integer(IP),  parameter,  public :: DTFFT_PATIENT                = CONF_DTFFT_PATIENT
  !< Patient flag. Same as `DTFFT_MEASURE`, but different MPI datatypes will also be tested

!------------------------------------------------------------------------------------------------
! Precision flags
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_SINGLE                 = CONF_DTFFT_SINGLE
  !< Use single precision
  integer(IP),  parameter,  public :: DTFFT_DOUBLE                 = CONF_DTFFT_DOUBLE
  !< Use double precision

!------------------------------------------------------------------------------------------------
! R2R Transform kinds
! This parameters matches FFTW definitions. I hope they will never change there.
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: DTFFT_DCT_1                  = CONF_DTFFT_DCT_1
  integer(IP),  parameter,  public :: DTFFT_DCT_2                  = CONF_DTFFT_DCT_2
  integer(IP),  parameter,  public :: DTFFT_DCT_3                  = CONF_DTFFT_DCT_3
  integer(IP),  parameter,  public :: DTFFT_DCT_4                  = CONF_DTFFT_DCT_4
  integer(IP),  parameter,  public :: DTFFT_DST_1                  = CONF_DTFFT_DST_1
  integer(IP),  parameter,  public :: DTFFT_DST_2                  = CONF_DTFFT_DST_2
  integer(IP),  parameter,  public :: DTFFT_DST_3                  = CONF_DTFFT_DST_3
  integer(IP),  parameter,  public :: DTFFT_DST_4                  = CONF_DTFFT_DST_4

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(IP), parameter,  public :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._C8P, 1._C8P)) / 8_IP
  !< Number of bytes to store single double precision complex element
  integer(IP), parameter,  public :: COMPLEX_STORAGE_SIZE          = storage_size((1._C4P, 1._C4P)) / 8_IP
  !< Number of bytes to store single float precision complex element
  integer(IP), parameter,  public :: DOUBLE_STORAGE_SIZE           = storage_size(1._R8P) / 8_IP
  !< Number of bytes to store single double precision real element
  integer(IP), parameter,  public :: FLOAT_STORAGE_SIZE            = storage_size(1._R4P) / 8_IP
  !< Number of bytes to store single single precision real element

!------------------------------------------------------------------------------------------------
! Correct values for input parameters
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter,  public :: VALID_FULL_TRANSPOSES(*) = [DTFFT_TRANSPOSE_OUT, DTFFT_TRANSPOSE_IN]
  integer(IP),  parameter,  public :: VALID_TRANSPOSES(*) = [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_Y, DTFFT_TRANSPOSE_X_TO_Z, DTFFT_TRANSPOSE_Z_TO_X]
  character(len=*), parameter,  public :: TRANSPOSE_NAMES(-3:3) = ["Z -> X", "Z -> Y", "Y -> X", "EMPTY ", "X -> Y", "Y -> Z", "X -> Z"]
  integer(IP),  parameter,  public :: VALID_EFFORTS(*) = [DTFFT_ESTIMATE, DTFFT_MEASURE, DTFFT_PATIENT]
  integer(IP),  parameter,  public :: VALID_PRECISIONS(*) = [DTFFT_SINGLE, DTFFT_DOUBLE]
  integer(IP),  parameter,  public :: VALID_DIMENSIONS(*) = [2, 3]
  integer(IP),  parameter,  public :: VALID_COMMUNICATORS(*) = [MPI_UNDEFINED, MPI_CART]
  integer(IP),  parameter,  public :: VALID_R2R_FFTS(*) = [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_4, DTFFT_DST_1, DTFFT_DST_2, DTFFT_DST_3, DTFFT_DST_4]
  integer(IP),  parameter,  public :: VALID_EXECUTORS(*) = [   &
    DTFFT_EXECUTOR_NONE                               &
#ifndef DTFFT_WITHOUT_FFTW
    ,DTFFT_EXECUTOR_FFTW3                             &
#endif
#ifdef DTFFT_WITH_MKL
    ,DTFFT_EXECUTOR_MKL                               &
#endif
#ifdef DTFFT_WITH_CUFFT
    ,DTFFT_EXECUTOR_CUFFT                             &
#endif
! #ifdef DTFFT_WITH_KFR
!     ,DTFFT_EXECUTOR_KFR                               &
! #endif
#ifdef DTFFT_WITH_VKFFT
    ,DTFFT_EXECUTOR_VKFFT                             &
#endif
  ]

  integer(IP),  parameter,  public  :: DTFFT_SUCCESS = CONF_DTFFT_SUCCESS
  integer(IP),  parameter,  public  :: DTFFT_ERROR_MPI_FINALIZED = CONF_DTFFT_ERROR_MPI_FINALIZED
  integer(IP),  parameter,  public  :: DTFFT_ERROR_PLAN_NOT_CREATED = CONF_DTFFT_ERROR_PLAN_NOT_CREATED
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_TRANSPOSE_TYPE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_N_DIMENSIONS = CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_DIMENSION_SIZE = CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_TYPE = CONF_DTFFT_ERROR_INVALID_COMM_TYPE
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_PRECISION = CONF_DTFFT_ERROR_INVALID_PRECISION
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_EFFORT_FLAG = CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_EXECUTOR_TYPE = CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_DIMS = CONF_DTFFT_ERROR_INVALID_COMM_DIMS
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_COMM_FAST_DIM = CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM
  integer(IP),  parameter,  public  :: DTFFT_ERROR_MISSING_R2R_KINDS = CONF_DTFFT_ERROR_MISSING_R2R_KINDS
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INVALID_R2R_KINDS = CONF_DTFFT_ERROR_INVALID_R2R_KINDS
  integer(IP),  parameter,  public  :: DTFFT_ERROR_R2C_TRANSPOSE_PLAN = CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN
  integer(IP),  parameter,  public  :: DTFFT_ERROR_INPLACE_TRANSPOSE = CONF_DTFFT_ERROR_INPLACE_TRANSPOSE
  integer(IP),  parameter,  public  :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
  integer(IP),  parameter,  public  :: DTFFT_ERROR_CUFFTMP_2D_PLAN = CONF_DTFFT_ERROR_CUFFTMP_2D_PLAN

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


contains

  pure subroutine dtfft_get_error_string(error_code, error_string)
    integer(IP),                   intent(in)  :: error_code
    character(len=:), allocatable, intent(out) :: error_string

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
    case ( DTFFT_ERROR_INPLACE_TRANSPOSE )
      allocate(error_string, source="Inplace transpose is not supported")
    case ( DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED )
      allocate(error_string, source="Selected `executor_type` do not support R2R FFTs")
    ! case ( DTFFT_ERROR_KFR_R2R_TYPE )
    !   allocate(error_string, source="KFR R2R executor only supports DTFFT_DCT_2 and DTFFT_DCT_3")
    ! case ( DTFFT_ERROR_KFR_R2C_SIZE )
    !   allocate(error_string, source="KFR R2C executor only supports even size")
    case ( DTFFT_ERROR_CUFFTMP_2D_PLAN )
      allocate(error_string, source="cufftMp support only 3d plan")
    case default
      allocate(error_string, source="Unknown error")
    endselect
  end subroutine dtfft_get_error_string
end module dtfft_parameters