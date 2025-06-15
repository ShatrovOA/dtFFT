!------------------------------------------------------------------------------------------------
! Copyright (c) 2021, Oleg Shatrov
! All rights reserved.
! This file is part of dtFFT library.

! dtFFT is free software: you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.

! dtFFT is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.

! You should have received a copy of the GNU General Public License
! along with this program.  If not, see <https://www.gnu.org/licenses/>.
!------------------------------------------------------------------------------------------------
#include "mkl_dfti.f90"
module dtfft_interface_mkl_native_m
!! This module creates native interface with MKL library
use MKL_DFTI
implicit none
private
public :: DFTI_NO_ERROR,                &
          DFTI_DOUBLE, DFTI_SINGLE,     &
          DFTI_NUMBER_OF_TRANSFORMS,    &
          DFTI_PLACEMENT,               &
          DFTI_INPUT_DISTANCE,          &
          DFTI_OUTPUT_DISTANCE,         &
          DFTI_CONJUGATE_EVEN_STORAGE,  &
          DFTI_COMPLEX_COMPLEX,         &
          DFTI_COMPLEX,                 &
          DFTI_REAL,                    &
          DFTI_INPLACE,                 &
          DFTI_NOT_INPLACE,             &
          DFTI_INPUT_STRIDES,           &
          DFTI_OUTPUT_STRIDES
end module dtfft_interface_mkl_native_m