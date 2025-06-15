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
!! Main ``dtFFT`` module. Should be used in a Fortran program.
use dtfft_config
use dtfft_parameters
use dtfft_pencil
use dtfft_plan
implicit none (type, external)
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

public :: dtfft_execute_t, dtfft_transpose_t
public :: dtfft_executor_t, dtfft_effort_t
public :: dtfft_precision_t, dtfft_r2r_kind_t

public :: operator(==)
public :: operator(/=)

! Execute types
public :: DTFFT_EXECUTE_FORWARD,                                    &
          DTFFT_EXECUTE_BACKWARD
! Transpose types
public :: DTFFT_TRANSPOSE_X_TO_Y,                                   &
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
public :: DTFFT_ERROR_MPI_FINALIZED
public :: DTFFT_ERROR_PLAN_NOT_CREATED
public :: DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
public :: DTFFT_ERROR_INVALID_N_DIMENSIONS
public :: DTFFT_ERROR_INVALID_DIMENSION_SIZE
public :: DTFFT_ERROR_INVALID_COMM_TYPE
public :: DTFFT_ERROR_INVALID_PRECISION
public :: DTFFT_ERROR_INVALID_EFFORT
public :: DTFFT_ERROR_INVALID_EXECUTOR
public :: DTFFT_ERROR_INVALID_COMM_DIMS
public :: DTFFT_ERROR_INVALID_COMM_FAST_DIM
public :: DTFFT_ERROR_MISSING_R2R_KINDS
public :: DTFFT_ERROR_INVALID_R2R_KINDS
public :: DTFFT_ERROR_R2C_TRANSPOSE_PLAN
public :: DTFFT_ERROR_INPLACE_TRANSPOSE
public :: DTFFT_ERROR_INVALID_AUX
public :: DTFFT_ERROR_INVALID_DIM
public :: DTFFT_ERROR_INVALID_USAGE
public :: DTFFT_ERROR_PLAN_IS_CREATED
public :: DTFFT_ERROR_ALLOC_FAILED
public :: DTFFT_ERROR_FREE_FAILED
public :: DTFFT_ERROR_INVALID_ALLOC_BYTES
public :: DTFFT_ERROR_DLOPEN_FAILED
public :: DTFFT_ERROR_DLSYM_FAILED
public :: DTFFT_ERROR_R2C_TRANSPOSE_CALLED
public :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
public :: DTFFT_ERROR_GPU_INVALID_STREAM
public :: DTFFT_ERROR_GPU_INVALID_BACKEND
public :: DTFFT_ERROR_GPU_NOT_SET
public :: DTFFT_ERROR_VKFFT_R2R_2D_PLAN
public :: DTFFT_ERROR_GPU_BACKENDS_DISABLED
public :: DTFFT_ERROR_NOT_DEVICE_PTR
public :: DTFFT_ERROR_NOT_NVSHMEM_PTR
public :: DTFFT_ERROR_INVALID_PLATFORM
public :: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE


public :: dtfft_config_t
public :: dtfft_create_config, dtfft_set_config

#ifdef DTFFT_WITH_CUDA

public :: dtfft_stream_t, dtfft_get_cuda_stream
public :: dtfft_platform_t

public :: DTFFT_PLATFORM_HOST, DTFFT_PLATFORM_CUDA

public :: DTFFT_BACKEND_MPI_DATATYPE
public :: DTFFT_BACKEND_MPI_P2P
public :: DTFFT_BACKEND_MPI_P2P_PIPELINED
public :: DTFFT_BACKEND_MPI_A2A
public :: DTFFT_BACKEND_NCCL
public :: DTFFT_BACKEND_NCCL_PIPELINED
public :: DTFFT_BACKEND_CUFFTMP

public :: dtfft_backend_t
public :: dtfft_get_backend_string

#endif
end module dtfft