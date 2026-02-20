!------------------------------------------------------------------------------------------------
! Copyright (c) 2021 - 2025, Oleg Shatrov
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
#ifdef DTFFT_WITH_COMPRESSION
use dtfft_abstract_compressor
#endif
use dtfft_config
use dtfft_errors
use dtfft_parameters
use dtfft_pencil
use dtfft_plan
implicit none
private

public :: dtfft_get_version
public :: DTFFT_VERSION_MAJOR
public :: DTFFT_VERSION_MINOR
public :: DTFFT_VERSION_PATCH
! Plans
public :: dtfft_plan_t
public :: dtfft_plan_c2c_t
public :: dtfft_plan_r2c_t
public :: dtfft_plan_r2r_t
! Helper types
public :: dtfft_request_t
public :: dtfft_pencil_t
! String getters
public :: dtfft_get_error_string
public :: dtfft_get_precision_string
public :: dtfft_get_executor_string
! Enums
public :: dtfft_execute_t, dtfft_transpose_t, dtfft_reshape_t, dtfft_layout_t
public :: dtfft_executor_t, dtfft_effort_t
public :: dtfft_precision_t, dtfft_r2r_kind_t
public :: dtfft_transpose_mode_t, dtfft_access_mode_t

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

! Reshape types
public :: DTFFT_RESHAPE_X_BRICKS_TO_PENCILS,                        &
          DTFFT_RESHAPE_X_PENCILS_TO_BRICKS,                        &
          DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS,                        &
          DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS,                        &
          DTFFT_RESHAPE_Y_BRICKS_TO_PENCILS,                        &
          DTFFT_RESHAPE_Y_PENCILS_TO_BRICKS

! Layout types
public :: DTFFT_LAYOUT_X_BRICKS,                                    &
          DTFFT_LAYOUT_Z_BRICKS,                                    &
          DTFFT_LAYOUT_X_PENCILS,                                   &
          DTFFT_LAYOUT_Y_PENCILS,                                   &
          DTFFT_LAYOUT_Z_PENCILS,                                   &
          DTFFT_LAYOUT_X_PENCILS_FOURIER

! 1d FFT External Executor types
public :: DTFFT_EXECUTOR_NONE
public :: DTFFT_EXECUTOR_FFTW3
public :: DTFFT_EXECUTOR_MKL
public :: DTFFT_EXECUTOR_CUFFT
public :: DTFFT_EXECUTOR_VKFFT

! Effort flags
public :: DTFFT_ESTIMATE,                                           &
          DTFFT_MEASURE,                                            &
          DTFFT_PATIENT,                                            &
          DTFFT_EXHAUSTIVE

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

! Transpose and access modes
public :: DTFFT_TRANSPOSE_MODE_PACK, DTFFT_TRANSPOSE_MODE_UNPACK
public :: DTFFT_ACCESS_MODE_WRITE, DTFFT_ACCESS_MODE_READ

! Error codes
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
public :: DTFFT_ERROR_INVALID_LAYOUT
public :: DTFFT_ERROR_INVALID_USAGE
public :: DTFFT_ERROR_PLAN_IS_CREATED
public :: DTFFT_ERROR_ALLOC_FAILED
public :: DTFFT_ERROR_FREE_FAILED
public :: DTFFT_ERROR_INVALID_ALLOC_BYTES
public :: DTFFT_ERROR_DLOPEN_FAILED
public :: DTFFT_ERROR_DLSYM_FAILED
! public :: DTFFT_ERROR_R2C_TRANSPOSE_CALLED
public :: DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH
public :: DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES
public :: DTFFT_ERROR_PENCIL_INVALID_COUNTS
public :: DTFFT_ERROR_PENCIL_INVALID_STARTS
public :: DTFFT_ERROR_PENCIL_SHAPE_MISMATCH
public :: DTFFT_ERROR_PENCIL_OVERLAP
public :: DTFFT_ERROR_PENCIL_NOT_CONTINUOUS
public :: DTFFT_ERROR_PENCIL_NOT_INITIALIZED
public :: DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS
public :: DTFFT_ERROR_INVALID_MEASURE_ITERS
public :: DTFFT_ERROR_INVALID_REQUEST
public :: DTFFT_ERROR_TRANSPOSE_ACTIVE
public :: DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE
public :: DTFFT_ERROR_INVALID_RESHAPE_TYPE
public :: DTFFT_ERROR_RESHAPE_ACTIVE
public :: DTFFT_ERROR_RESHAPE_NOT_ACTIVE
public :: DTFFT_ERROR_INPLACE_RESHAPE
public :: DTFFT_ERROR_INVALID_EXECUTE_TYPE
public :: DTFFT_ERROR_RESHAPE_NOT_SUPPORTED
public :: DTFFT_ERROR_R2C_EXECUTE_CALLED
public :: DTFFT_ERROR_INVALID_CART_COMM
public :: DTFFT_ERROR_INVALID_TRANSPOSE_MODE
public :: DTFFT_ERROR_INVALID_ACCESS_MODE
public :: DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
public :: DTFFT_ERROR_GPU_INVALID_STREAM
public :: DTFFT_ERROR_INVALID_BACKEND
public :: DTFFT_ERROR_GPU_NOT_SET
public :: DTFFT_ERROR_VKFFT_R2R_2D_PLAN
public :: DTFFT_ERROR_BACKENDS_DISABLED
public :: DTFFT_ERROR_NOT_DEVICE_PTR
public :: DTFFT_ERROR_NOT_NVSHMEM_PTR
public :: DTFFT_ERROR_INVALID_PLATFORM
public :: DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR
public :: DTFFT_ERROR_INVALID_PLATFORM_BACKEND

! Configuration
public :: dtfft_config_t
public :: dtfft_create_config, dtfft_set_config
! Backends
public :: DTFFT_BACKEND_MPI_DATATYPE
public :: DTFFT_BACKEND_MPI_P2P
public :: DTFFT_BACKEND_MPI_P2P_PIPELINED
public :: DTFFT_BACKEND_MPI_A2A
public :: DTFFT_BACKEND_MPI_RMA
public :: DTFFT_BACKEND_MPI_RMA_PIPELINED
public :: DTFFT_BACKEND_MPI_P2P_SCHEDULED
public :: DTFFT_BACKEND_MPI_P2P_FUSED
public :: DTFFT_BACKEND_MPI_RMA_FUSED
public :: DTFFT_BACKEND_ADAPTIVE

public :: dtfft_backend_t
public :: dtfft_get_backend_string
public :: dtfft_get_backend_pipelined

#ifdef DTFFT_WITH_CUDA

public :: dtfft_stream_t, dtfft_get_cuda_stream
public :: dtfft_platform_t
! Platforms
public :: DTFFT_PLATFORM_HOST, DTFFT_PLATFORM_CUDA
! GPU backends
public :: DTFFT_BACKEND_NCCL
public :: DTFFT_BACKEND_NCCL_PIPELINED
public :: DTFFT_BACKEND_CUFFTMP
public :: DTFFT_BACKEND_CUFFTMP_PIPELINED
#endif

#ifdef DTFFT_WITH_COMPRESSION
public :: dtfft_compression_lib_t
public :: dtfft_compression_mode_t
public :: dtfft_compression_config_t
public :: DTFFT_COMPRESSION_LIB_ZFP
public :: DTFFT_COMPRESSION_MODE_LOSSLESS
public :: DTFFT_COMPRESSION_MODE_FIXED_RATE
public :: DTFFT_COMPRESSION_MODE_FIXED_PRECISION
public :: DTFFT_COMPRESSION_MODE_FIXED_ACCURACY

public :: DTFFT_BACKEND_MPI_P2P_COMPRESSED
public :: DTFFT_BACKEND_MPI_RMA_COMPRESSED
# ifdef DTFFT_WITH_CUDA
public :: DTFFT_BACKEND_NCCL_COMPRESSED
# endif
#endif
end module dtfft