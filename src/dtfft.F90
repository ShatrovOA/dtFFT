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
!< Main DTFFT module. Should be included in Fortran program.
!------------------------------------------------------------------------------------------------
use dtfft_parameters
use dtfft_plan_c2c_m
use dtfft_plan_r2r_m
use dtfft_plan_r2c_m
private

! Plans
public :: dtfft_plan_c2c_2d, dtfft_plan_c2c_3d,                   &
          dtfft_plan_r2r_2d, dtfft_plan_r2r_3d,                   &
          dtfft_plan_r2c_2d, dtfft_plan_r2c_3d
          
! Transpose types
public :: DTFFT_TRANSPOSE_OUT,                                    &
          DTFFT_TRANSPOSE_IN

! 1d FFT External Executor types
public :: DTFFT_EXECUTOR_FFTW3,                                   &
          DTFFT_EXECUTOR_MKL,                                     &
          DTFFT_EXECUTOR_CUFFT

! Effort flags, currently its value is ignored
public :: DTFFT_ESTIMATE,                                         &
          DTFFT_MEASURE,                                          &
          DTFFT_PATIENT
end module dtfft