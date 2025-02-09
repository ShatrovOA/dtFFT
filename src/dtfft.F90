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
module dtfft
!! Main DTFFT module. Should be used in a Fortran program.
use dtfft_parameters
use dtfft_pencil
use dtfft_plan
use dtfft_utils
implicit none
private

public :: dtfft_get_version
public :: DTFFT_VERSION_MAJOR
public :: DTFFT_VERSION_MINOR
public :: DTFFT_VERSION_PATCH
! Plans
public :: dtfft_plan_t
public :: dtfft_plan_c2c_t
#ifndef DTFFT_TRANSPOSE_ONLY
public :: dtfft_plan_r2c_t
#endif
public :: dtfft_plan_r2r_t

public :: dtfft_pencil_t
public :: dtfft_get_error_string

public :: dtfft_execute_type_t, dtfft_transpose_type_t
public :: dtfft_executor_t, dtfft_effort_t
public :: dtfft_precision_t, dtfft_r2r_kind_t

public :: operator(==)
public :: operator(/=)

! Transpose types
public :: DTFFT_TRANSPOSE_OUT,                                      &
          DTFFT_TRANSPOSE_IN,                                       &
          DTFFT_TRANSPOSE_X_TO_Y,                                   &
          DTFFT_TRANSPOSE_Y_TO_X,                                   &
          DTFFT_TRANSPOSE_Y_TO_Z,                                   &
          DTFFT_TRANSPOSE_Z_TO_Y,                                   &
          DTFFT_TRANSPOSE_X_TO_Z,                                   &
          DTFFT_TRANSPOSE_Z_TO_X

! 1d FFT External Executor types
public :: DTFFT_EXECUTOR_NONE
public :: DTFFT_EXECUTOR_FFTW3
public :: DTFFT_EXECUTOR_MKL
public :: DTFFT_EXECUTOR_CUFFT
public :: DTFFT_EXECUTOR_VKFFT

! Effort flags
public :: DTFFT_ESTIMATE,                                           &
          DTFFT_MEASURE,                                            &
          DTFFT_PATIENT

! Precision flags
public :: DTFFT_SINGLE,                                             &
          DTFFT_DOUBLE

! Types of R2R Transform
public :: DTFFT_DCT_1,                                              &
          DTFFT_DCT_2,                                              &
          DTFFT_DCT_3,                                              &
          DTFFT_DCT_4,                                              &
          DTFFT_DST_1,                                              &
          DTFFT_DST_2,                                              &
          DTFFT_DST_3,                                              &
          DTFFT_DST_4

public :: DTFFT_SUCCESS

public :: dtfft_config_t
public :: dtfft_create_config, dtfft_set_config

#ifdef DTFFT_WITH_CUDA

public :: DTFFT_GPU_BACKEND_MPI_DATATYPE
public :: DTFFT_GPU_BACKEND_MPI_P2P
public :: DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED
public :: DTFFT_GPU_BACKEND_MPI_A2A
public :: DTFFT_GPU_BACKEND_NCCL
public :: DTFFT_GPU_BACKEND_NCCL_PIPELINED

public :: dtfft_gpu_backend_t
public :: dtfft_get_gpu_backend_string

#endif
end module dtfft