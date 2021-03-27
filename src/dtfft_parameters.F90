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
  integer(IP),  parameter :: DTFFT_TRANSPOSE_OUT    = 101 
  !< Transpose out of "real" space to Fourier space
  integer(IP),  parameter :: DTFFT_TRANSPOSE_IN     = 102   
  !< Transpose into "real" space from Fourier space

!------------------------------------------------------------------------------------------------
! External executor types
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_EXECUTOR_FFTW3   = 200   
  !< FFTW3 executor
  integer(IP),  parameter :: DTFFT_EXECUTOR_MKL     = 201   
  !< MKL executor
  integer(IP),  parameter :: DTFFT_EXECUTOR_CUFFT   = 202   
  !< cuFFT executor

!------------------------------------------------------------------------------------------------
! C2C Execution directions
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_FORWARD          = -1    
  !< Forward c2c transform
  integer(IP),  parameter :: DTFFT_BACKWARD         = +1    
  !< Backward c2c transform

!------------------------------------------------------------------------------------------------
! Effort flags. Reserved for future. Unused
!------------------------------------------------------------------------------------------------
  integer(IP),  parameter :: DTFFT_ESTIMATE         = 0     
  !< Estimate flag. DTFFT will use default decomposition provided by MPI_Dims_create
  integer(IP),  parameter :: DTFFT_MEASURE          = 1     
  !< Measure flag. DTFFT will run transpose routines to find the best decomposition
  integer(IP),  parameter :: DTFFT_PATIENT          = 2     
  !< Patient flag. Same as DTFFT_MEASURE, but different datatypes will also be tested

!------------------------------------------------------------------------------------------------
! Storage sizes
!------------------------------------------------------------------------------------------------
  integer(IP), parameter, public :: DOUBLE_COMPLEX_STORAGE_SIZE   = storage_size((1._C8P, 1._C8P)) / 8_IP   
  !< Number of bytes to store single double precision complex element
  integer(IP), parameter, public :: COMPLEX_STORAGE_SIZE          = storage_size((1._C4P, 1._C4P)) / 8_IP   
  !< Number of bytes to store single float precision complex element
  integer(IP), parameter, public :: DOUBLE_STORAGE_SIZE           = storage_size(1._R8P) / 8_IP             
  !< Number of bytes to store single double precision real element
  integer(IP), parameter, public :: FLOAT_STORAGE_SIZE            = storage_size(1._R4P) / 8_IP             
  !< Number of bytes to store single single precision real element

!------------------------------------------------------------------------------------------------
! Pensil types. Used to identify which work buffer is needed
!------------------------------------------------------------------------------------------------
  integer(IP), parameter, public :: X_PENCIL = 1
  !< X aligned work buffer is required
  integer(IP), parameter, public :: Y_PENCIL = 2
  !< Y aligned work buffer is required
end module dtfft_parameters