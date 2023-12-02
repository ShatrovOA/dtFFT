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
module dtfft
!------------------------------------------------------------------------------------------------
!< Main DTFFT module. Should be included in a Fortran program.
!------------------------------------------------------------------------------------------------
use dtfft_parameters
use dtfft_core_m
private

! Plans
public :: dtfft_core,                                               &
          dtfft_plan_c2c,                                           &
          dtfft_plan_r2c,                                           &
          dtfft_plan_r2r

! Transpose types
public :: DTFFT_TRANSPOSE_OUT,                                      &
          DTFFT_TRANSPOSE_IN,                                       &
          DTFFT_TRANSPOSE_X_TO_Y,                                   &
          DTFFT_TRANSPOSE_Y_TO_X,                                   &
          DTFFT_TRANSPOSE_Y_TO_Z,                                   &
          DTFFT_TRANSPOSE_Z_TO_Y

! 1d FFT External Executor types
public :: DTFFT_EXECUTOR_NONE,                                      &
          DTFFT_EXECUTOR_FFTW3,                                     &
          DTFFT_EXECUTOR_MKL                                        !&
          ! DTFFT_EXECUTOR_CUFFT,                                     &
          ! DTFFT_EXECUTOR_KFR

! Effort flags, currently its value is ignored
public :: DTFFT_ESTIMATE,                                           &
          DTFFT_MEASURE,                                            &
          DTFFT_PATIENT

! Precision flags
public :: DTFFT_SINGLE,                                             &
          DTFFT_DOUBLE

public :: DTFFT_DCT_1,                                              &
          DTFFT_DCT_2,                                              &
          DTFFT_DCT_3,                                              &
          DTFFT_DCT_4,                                              &
          DTFFT_DST_1,                                              &
          DTFFT_DST_2,                                              &
          DTFFT_DST_3,                                              &
          DTFFT_DST_4
end module dtfft