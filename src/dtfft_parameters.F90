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
module dtfft_parameters
!------------------------------------------------------------------------------------------------
!< This module defines common DTFFT parameters
!------------------------------------------------------------------------------------------------
use dtfft_precisions
implicit none
public

!------------------------------------------------------------------------------------------------
! Traspose types
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_TRANSPOSE_OUT          = -1
  !< Transpose out of "real" space to Fourier space
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_OUT")     :: C_DTFFT_TRANSPOSE_OUT = DTFFT_TRANSPOSE_OUT
  !< Export [[DTFFT_TRANSPOSE_OUT]] parameter to object file
  integer(IP),  parameter :: DTFFT_TRANSPOSE_IN           = +1
  !< Transpose into "real" space from Fourier space
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_IN")      :: C_DTFFT_TRANSPOSE_IN = DTFFT_TRANSPOSE_IN
  !< Export [[DTFFT_TRANSPOSE_IN]] parameter to object file
  integer(IP),  parameter :: DTFFT_TRANSPOSE_X_TO_Y       = 1
  !< Perform single transposition, from X aligned to Y aligned
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_X_TO_Y")  :: C_DTFFT_TRANSPOSE_X_TO_Y = DTFFT_TRANSPOSE_X_TO_Y
  !< Export [[DTFFT_TRANSPOSE_X_TO_Y]] parameter to object file
  integer(IP),  parameter :: DTFFT_TRANSPOSE_Y_TO_X       = -1
  !< Perform single transposition, from Y aligned to X aligned
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_Y_TO_X")  :: C_DTFFT_TRANSPOSE_Y_TO_X = DTFFT_TRANSPOSE_Y_TO_X
  !< Export [[DTFFT_TRANSPOSE_Y_TO_X]] parameter to object file
  integer(IP),  parameter :: DTFFT_TRANSPOSE_Y_TO_Z       = 2
  !< Perform single transposition, from Y aligned to Z aligned
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_Y_TO_Z")  :: C_DTFFT_TRANSPOSE_Y_TO_Z = DTFFT_TRANSPOSE_Y_TO_Z
  !< Export [[DTFFT_TRANSPOSE_Y_TO_Z]] parameter to object file
  integer(IP),  parameter :: DTFFT_TRANSPOSE_Z_TO_Y       = -2
  !< Perform single transposition, from Z aligned to Y aligned
  integer(IP),  bind(C, name="C_DTFFT_TRANSPOSE_Z_TO_Y")  :: C_DTFFT_TRANSPOSE_Z_TO_Y = DTFFT_TRANSPOSE_Z_TO_Y
  !< Export [[DTFFT_TRANSPOSE_Z_TO_Y]] parameter to object file

!------------------------------------------------------------------------------------------------
! External executor types
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_EXECUTOR_NONE          = 0
  !< Do not setup any executor. If this type is provided, then execute method cannot be called.
  !< Use transpose method instead
  integer(IP),  bind(C, name="C_DTFFT_EXECUTOR_NONE")     :: C_DTFFT_EXECUTOR_NONE = DTFFT_EXECUTOR_NONE
  !< Export [[DTFFT_EXECUTOR_NONE]] parameter to object file
  integer(IP),  parameter :: DTFFT_EXECUTOR_FFTW3         = 1
  !< FFTW3 executor
  integer(IP),  bind(C, name="C_DTFFT_EXECUTOR_FFTW3")    :: C_DTFFT_EXECUTOR_FFTW3 = DTFFT_EXECUTOR_FFTW3
  !< Export [[DTFFT_EXECUTOR_FFTW3]] parameter to object file
  integer(IP),  parameter :: DTFFT_EXECUTOR_MKL           = 2
  !< MKL executor
  integer(IP),  bind(C, name="C_DTFFT_EXECUTOR_MKL")      :: C_DTFFT_EXECUTOR_MKL = DTFFT_EXECUTOR_MKL
  !< Export [[DTFFT_EXECUTOR_MKL]] parameter to object file
  integer(IP),  parameter :: DTFFT_EXECUTOR_CUFFT         = 3
  !< cuFFT executor
  integer(IP),  bind(C, name="C_DTFFT_EXECUTOR_CUFFT")    :: C_DTFFT_EXECUTOR_CUFFT = DTFFT_EXECUTOR_CUFFT
  !< Export [[DTFFT_EXECUTOR_CUFFT]] parameter to object file
  integer(IP),  parameter :: DTFFT_EXECUTOR_KFR           = 4
  !< KFR executor
  integer(IP),  bind(C, name="C_DTFFT_EXECUTOR_KFR")      :: C_DTFFT_EXECUTOR_KFR = DTFFT_EXECUTOR_KFR
  !< Export [[DTFFT_EXECUTOR_KFR]] parameter to object file

!------------------------------------------------------------------------------------------------
! C2C Execution directions
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_FORWARD                = -1
  !< Forward c2c transform
  integer(IP),  bind(C, name="C_DTFFT_FORWARD")           :: C_DTFFT_FORWARD = DTFFT_FORWARD
  !< Export [[DTFFT_FORWARD]] parameter to object file
  integer(IP),  parameter :: DTFFT_BACKWARD               = +1
  !< Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags. Reserved for future. Unused
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_ESTIMATE               = 0
  !< Estimate flag. DTFFT will use default decomposition provided by MPI_Dims_create
  integer(IP),  bind(C, name="C_DTFFT_ESTIMATE")          :: C_DTFFT_ESTIMATE = DTFFT_ESTIMATE
  !< Export [[DTFFT_ESTIMATE]] parameter to object file
  integer(IP),  parameter :: DTFFT_MEASURE                = 1
  !< Measure flag. DTFFT will run transpose routines to find the best decomposition
  integer(IP),  bind(C, name="C_DTFFT_MEASURE")           :: C_DTFFT_MEASURE = DTFFT_MEASURE
  !< Export [[DTFFT_MEASURE]] parameter to object file
  integer(IP),  parameter :: DTFFT_PATIENT                = 2
  !< Patient flag. Same as DTFFT_MEASURE, but different datatypes will also be tested
  integer(IP),  bind(C, name="C_DTFFT_PATIENT")           :: C_DTFFT_PATIENT = DTFFT_PATIENT
  !< Export [[DTFFT_PATIENT]] parameter to object file

!------------------------------------------------------------------------------------------------
! Precision flags
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_SINGLE                 = 0
  !< Use single precision
  integer(IP),  bind(C, name="C_DTFFT_SINGLE")            :: C_DTFFT_SINGLE = DTFFT_SINGLE
  !< Export [[DTFFT_SINGLE]] parameter to object file
  integer(IP),  parameter :: DTFFT_DOUBLE                 = 1
  !< Use double precision
  integer(IP),  bind(C, name="C_DTFFT_DOUBLE")            :: C_DTFFT_DOUBLE = DTFFT_DOUBLE
  !< Export [[DTFFT_DOUBLE]] parameter to object file

!------------------------------------------------------------------------------------------------
! R2R Transform kinds
! This parameters matches FFTW definitions. I hope they will never change there.
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_DCT_1                  = 3
  integer(IP),  bind(C, name="C_DTFFT_DCT_1")             :: C_DTFFT_DCT_1 = DTFFT_DCT_1
  integer(IP),  parameter :: DTFFT_DCT_2                  = 5
  integer(IP),  bind(C, name="C_DTFFT_DCT_2")             :: C_DTFFT_DCT_2 = DTFFT_DCT_2
  integer(IP),  parameter :: DTFFT_DCT_3                  = 4
  integer(IP),  bind(C, name="C_DTFFT_DCT_3")             :: C_DTFFT_DCT_3 = DTFFT_DCT_3
  integer(IP),  parameter :: DTFFT_DCT_4                  = 6
  integer(IP),  bind(C, name="C_DTFFT_DCT_4")             :: C_DTFFT_DCT_4 = DTFFT_DCT_4
  integer(IP),  parameter :: DTFFT_DST_1                  = 7
  integer(IP),  bind(C, name="C_DTFFT_DST_1")             :: C_DTFFT_DST_1 = DTFFT_DST_1
  integer(IP),  parameter :: DTFFT_DST_2                  = 9
  integer(IP),  bind(C, name="C_DTFFT_DST_2")             :: C_DTFFT_DST_2 = DTFFT_DST_2
  integer(IP),  parameter :: DTFFT_DST_3                  = 8
  integer(IP),  bind(C, name="C_DTFFT_DST_3")             :: C_DTFFT_DST_3 = DTFFT_DST_3
  integer(IP),  parameter :: DTFFT_DST_4                  = 10
  integer(IP),  bind(C, name="C_DTFFT_DST_4")             :: C_DTFFT_DST_4 = DTFFT_DST_4

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(IP), parameter :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._C8P, 1._C8P)) / 8_IP
  !< Number of bytes to store single double precision complex element
  integer(IP), parameter :: COMPLEX_STORAGE_SIZE          = storage_size((1._C4P, 1._C4P)) / 8_IP
  !< Number of bytes to store single float precision complex element
  integer(IP), parameter :: DOUBLE_STORAGE_SIZE           = storage_size(1._R8P) / 8_IP
  !< Number of bytes to store single double precision real element
  integer(IP), parameter :: FLOAT_STORAGE_SIZE            = storage_size(1._R4P) / 8_IP
  !< Number of bytes to store single single precision real element

!------------------------------------------------------------------------------------------------
! Pencil types. Used to identify which work buffer is needed
!------------------------------------------------------------------------------------------------
  integer(IP), parameter :: X_PENCIL = 1
  !< X aligned work buffer is required
  integer(IP), parameter :: Y_PENCIL = 2
  !< Y aligned work buffer is required

!------------------------------------------------------------------------------------------------
! Correct values for input parameters
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: VALID_FULL_TRANSPOSES(*) = [DTFFT_TRANSPOSE_OUT, DTFFT_TRANSPOSE_IN]
  integer(IP),  parameter :: VALID_TRANSPOSES(*) = [DTFFT_TRANSPOSE_X_TO_Y, DTFFT_TRANSPOSE_Y_TO_X, DTFFT_TRANSPOSE_Y_TO_Z, DTFFT_TRANSPOSE_Z_TO_Y]
  character(len=*), parameter :: TRANSPOSE_NAMES(-2:2) = ["Z -> Y", "Y -> X", "EMPTY ", "X -> Y", "Y -> Z"]
  integer(IP),  parameter :: VALID_EFFORTS(*) = [DTFFT_ESTIMATE, DTFFT_MEASURE, DTFFT_PATIENT]
  integer(IP),  parameter :: VALID_PRECISIONS(*) = [DTFFT_SINGLE, DTFFT_DOUBLE]
  integer(IP),  parameter :: VALID_DIMENSIONS(*) = [2, 3]
  integer(IP),  parameter :: VALID_R2R_FFTS(*) = [DTFFT_DCT_1, DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_4, DTFFT_DST_1, DTFFT_DST_2, DTFFT_DST_3, DTFFT_DST_4]
  integer(IP),  parameter :: VALID_EXECUTORS(*) = [   &
    DTFFT_EXECUTOR_NONE                               &
#if !defined(NO_FFTW3) 
    ,DTFFT_EXECUTOR_FFTW3                             &
#endif
#if defined(MKL_ENABLED)
    ,DTFFT_EXECUTOR_MKL                               &
#endif
#if defined(CUFFT_ENABLED)
    ,DTFFT_EXECUTOR_CUFFT                             &
#endif
#if defined(KFR_ENABLED)
    ,DTFFT_EXECUTOR_KFR                               &
#endif
  ]
end module dtfft_parameters