/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
  All rights reserved.
  This file is part of dtFFT library.

  dtFFT is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  dtFFT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/**
 * @file dtfft.h
 * @author Oleg Shatrov
 * @date 2024 - 2025
 * @brief File containing C API of dtFFT Library
 */
#ifndef DTFFT_H
#define DTFFT_H

#include "dtfft_config.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h> // bool
#include <stdint.h> // int8_t, int32_t, uint8_t
#include <stdio.h> // fprintf, stderr, size_t

/** dtFFT Major Version */
#define DTFFT_VERSION_MAJOR CONF_DTFFT_VERSION_MAJOR
/** dtFFT Minor Version */
#define DTFFT_VERSION_MINOR CONF_DTFFT_VERSION_MINOR
/** dtFFT Patch Version */
#define DTFFT_VERSION_PATCH CONF_DTFFT_VERSION_PATCH
/** dtFFT Version Code. Can be used for version comparison */
#define DTFFT_VERSION_CODE CONF_DTFFT_VERSION_CODE
/** Generates Version Code based on Major, Minor, Patch */
#define DTFFT_VERSION(X, Y, Z) CONF_DTFFT_VERSION(X, Y, Z)

/** @return `::DTFFT_VERSION_CODE` defined during library compilation */
int32_t
dtfft_get_version();

/** Safe call macro.
 *
 * Should be used to check error codes returned by ``dtFFT``.
 *
 * @details Writes an error message to ``stderr`` and calls ``MPI_Abort`` if an error occurs.
 *
 * **Example**
 * @code
 * DTFFT_CALL( dtfft_transpose(plan, a, b) )
 * @endcode
 */
#define DTFFT_CALL(call)                                           \
    do {                                                           \
        dtfft_error_t ierr = call;                                 \
        if (ierr != DTFFT_SUCCESS) {                               \
            fprintf(stderr, "dtFFT error in file '%s:%i': %s.\n",  \
                __FILE__, __LINE__, dtfft_get_error_string(ierr)); \
            MPI_Abort(MPI_COMM_WORLD, ierr);                       \
        }                                                          \
    } while (0);

/** This enum lists the different error codes that ``dtFFT`` can return.
 * @see dtfft_get_error_string, DTFFT_CALL
 */
typedef enum {
    /** Successful execution */
    DTFFT_SUCCESS = CONF_DTFFT_SUCCESS,
    /** MPI_Init is not called or MPI_Finalize has already been called */
    DTFFT_ERROR_MPI_FINALIZED = CONF_DTFFT_ERROR_MPI_FINALIZED,
    /** Plan not created */
    DTFFT_ERROR_PLAN_NOT_CREATED = CONF_DTFFT_ERROR_PLAN_NOT_CREATED,
    /** Invalid `transpose_type` provided */
    DTFFT_ERROR_INVALID_TRANSPOSE_TYPE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE,
    /** Invalid Number of dimensions provided. Valid options are 2 and 3 */
    DTFFT_ERROR_INVALID_N_DIMENSIONS = CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS,
    /** One or more provided dimension sizes <= 0 */
    DTFFT_ERROR_INVALID_DIMENSION_SIZE = CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE,
    /** Invalid communicator type provided */
    DTFFT_ERROR_INVALID_COMM_TYPE = CONF_DTFFT_ERROR_INVALID_COMM_TYPE,
    /** Invalid `precision` parameter provided */
    DTFFT_ERROR_INVALID_PRECISION = CONF_DTFFT_ERROR_INVALID_PRECISION,
    /** Invalid `effort` parameter provided */
    DTFFT_ERROR_INVALID_EFFORT = CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG,
    /** Invalid `executor` parameter provided */
    DTFFT_ERROR_INVALID_EXECUTOR = CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE,
    /** Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine */
    DTFFT_ERROR_INVALID_COMM_DIMS = CONF_DTFFT_ERROR_INVALID_COMM_DIMS,
    /** Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1 */
    DTFFT_ERROR_INVALID_COMM_FAST_DIM = CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM,
    /** For R2R plan, `kinds` parameter must be passed if `executor` != `::DTFFT_EXECUTOR_NONE` */
    DTFFT_ERROR_MISSING_R2R_KINDS = CONF_DTFFT_ERROR_MISSING_R2R_KINDS,
    /** Invalid values detected in `kinds` parameter */
    DTFFT_ERROR_INVALID_R2R_KINDS = CONF_DTFFT_ERROR_INVALID_R2R_KINDS,
    /** Transpose plan is not supported in R2C, use R2R or C2C plan instead */
    DTFFT_ERROR_R2C_TRANSPOSE_PLAN = CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN,
    /** Inplace transpose is not supported */
    DTFFT_ERROR_INPLACE_TRANSPOSE = CONF_DTFFT_ERROR_INPLACE_TRANSPOSE,
    /** Invalid `aux` buffer provided */
    DTFFT_ERROR_INVALID_AUX = CONF_DTFFT_ERROR_INVALID_AUX,
    /** Invalid `layout` passed to `::dtfft_get_pencil` */
    DTFFT_ERROR_INVALID_LAYOUT = CONF_DTFFT_ERROR_INVALID_LAYOUT,
    /** Invalid API Usage. */
    DTFFT_ERROR_INVALID_USAGE = CONF_DTFFT_ERROR_INVALID_USAGE,
    /** Trying to create already created plan */
    DTFFT_ERROR_PLAN_IS_CREATED = CONF_DTFFT_ERROR_PLAN_IS_CREATED,
    /** Selected `executor` do not support R2R FFTs */
    DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED,
    /** Internal call of `::dtfft_mem_alloc` failed */
    DTFFT_ERROR_ALLOC_FAILED = CONF_DTFFT_ERROR_ALLOC_FAILED,
    /** Internal call of `::dtfft_mem_free` failed */
    DTFFT_ERROR_FREE_FAILED = CONF_DTFFT_ERROR_FREE_FAILED,
    /** Invalid `alloc_bytes` provided */
    DTFFT_ERROR_INVALID_ALLOC_BYTES = CONF_DTFFT_ERROR_INVALID_ALLOC_BYTES,
    /** Failed to dynamically load library */
    DTFFT_ERROR_DLOPEN_FAILED = CONF_DTFFT_ERROR_DLOPEN_FAILED,
    /** Failed to dynamically load symbol */
    DTFFT_ERROR_DLSYM_FAILED = CONF_DTFFT_ERROR_DLSYM_FAILED,
    /** Deprecated/unused: R2C transpose call restriction (kept for backward compatibility of error code numbering) */
    // DTFFT_ERROR_R2C_TRANSPOSE_CALLED = CONF_DTFFT_ERROR_R2C_TRANSPOSE_CALLED,
    /** Sizes of `starts` and `counts` arrays passed to `dtfft_pencil_t` constructor do not match */
    DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH = CONF_DTFFT_ERROR_PENCIL_ARRAYS_SIZE_MISMATCH,
    /** Sizes of `starts` and `counts` < 2 or > 3 provided to `dtfft_pencil_t` constructor */
    DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES = CONF_DTFFT_ERROR_PENCIL_ARRAYS_INVALID_SIZES,
    /** Invalid `counts` provided to `dtfft_pencil_t` constructor */
    DTFFT_ERROR_PENCIL_INVALID_COUNTS = CONF_DTFFT_ERROR_PENCIL_INVALID_COUNTS,
    /** Invalid `starts` provided to `dtfft_pencil_t` constructor */
    DTFFT_ERROR_PENCIL_INVALID_STARTS = CONF_DTFFT_ERROR_PENCIL_INVALID_STARTS,
    /** Processes have same lower bounds but different sizes in some dimensions */
    DTFFT_ERROR_PENCIL_SHAPE_MISMATCH = CONF_DTFFT_ERROR_PENCIL_SHAPE_MISMATCH,
    /** Pencil overlap detected, i.e. two processes share same part of global space */
    DTFFT_ERROR_PENCIL_OVERLAP = CONF_DTFFT_ERROR_PENCIL_OVERLAP,
    /** Local pencils do not cover the global space without gaps */
    DTFFT_ERROR_PENCIL_NOT_CONTINUOUS = CONF_DTFFT_ERROR_PENCIL_NOT_CONTINUOUS,
    /** Pencil is not initialized, i.e. `constructor` subroutine was not called */
    DTFFT_ERROR_PENCIL_NOT_INITIALIZED = CONF_DTFFT_ERROR_PENCIL_NOT_INITIALIZED,
    /** Invalid `n_measure_warmup_iters` provided */
    DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS = CONF_DTFFT_ERROR_INVALID_MEASURE_WARMUP_ITERS,
    /** Invalid `n_measure_iters` provided */
    DTFFT_ERROR_INVALID_MEASURE_ITERS = CONF_DTFFT_ERROR_INVALID_MEASURE_ITERS,
    /** Invalid `dtfft_request_t` provided */
    DTFFT_ERROR_INVALID_REQUEST = CONF_DTFFT_ERROR_INVALID_REQUEST,
    /** Attempting to execute already active transposition */
    DTFFT_ERROR_TRANSPOSE_ACTIVE = CONF_DTFFT_ERROR_TRANSPOSE_ACTIVE,
    /** Attempting to finalize non-active transposition */
    DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE = CONF_DTFFT_ERROR_TRANSPOSE_NOT_ACTIVE,
    /** Invalid `reshape_type` provided */
    DTFFT_ERROR_INVALID_RESHAPE_TYPE = CONF_DTFFT_ERROR_INVALID_RESHAPE_TYPE,
    /** Attempting to execute already active reshape */
    DTFFT_ERROR_RESHAPE_ACTIVE = CONF_DTFFT_ERROR_RESHAPE_ACTIVE,
    /** Attempting to finalize non-active reshape */
    DTFFT_ERROR_RESHAPE_NOT_ACTIVE = CONF_DTFFT_ERROR_RESHAPE_NOT_ACTIVE,
    /** Inplace reshape is not supported */
    DTFFT_ERROR_INPLACE_RESHAPE = CONF_DTFFT_ERROR_INPLACE_RESHAPE,
    /** R2C reshape was called */
    // DTFFT_ERROR_R2C_RESHAPE_CALLED = CONF_DTFFT_ERROR_R2C_RESHAPE_CALLED,
    /** Invalid `execute_type` provided */
    DTFFT_ERROR_INVALID_EXECUTE_TYPE = CONF_DTFFT_ERROR_INVALID_EXECUTE_TYPE,
    /** Reshape is not supported for this plan */
    DTFFT_ERROR_RESHAPE_NOT_SUPPORTED = CONF_DTFFT_ERROR_RESHAPE_NOT_SUPPORTED,
    /** Execute called for transpose-only R2C Plan */
    DTFFT_ERROR_R2C_EXECUTE_CALLED = CONF_DTFFT_ERROR_R2C_EXECUTE_CALLED,
    /** Invalid cartesian communicator provided */
    DTFFT_ERROR_INVALID_CART_COMM = CONF_DTFFT_ERROR_INVALID_CART_COMM,
    /** Invalid transpose mode provided */
    DTFFT_ERROR_INVALID_TRANSPOSE_MODE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_MODE,
    /** Invalid stream provided */
    DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM,
    /** Invalid backend provided */
    DTFFT_ERROR_INVALID_BACKEND = CONF_DTFFT_ERROR_INVALID_BACKEND,
    /** Multiple MPI Processes located on same host share same GPU which is not supported */
    DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET,
    /** When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions */
    DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN,
    /** Passed `effort` ==  `::DTFFT_PATIENT` but all Backends has been disabled by `::dtfft_config_t`*/
    DTFFT_ERROR_BACKENDS_DISABLED = CONF_DTFFT_ERROR_BACKENDS_DISABLED,
    /** One of pointers passed to `::dtfft_execute` or `::dtfft_transpose` cannot be accessed from device */
    DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR,
    /** One of pointers passed to `::dtfft_execute` or `::dtfft_transpose` is not an `NVSHMEM` pointer */
    DTFFT_ERROR_NOT_NVSHMEM_PTR = CONF_DTFFT_ERROR_NOT_NVSHMEM_PTR,
    /** Invalid platform provided */
    DTFFT_ERROR_INVALID_PLATFORM = CONF_DTFFT_ERROR_INVALID_PLATFORM,
    /** Invalid executor provided for selected platform */
    DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR = CONF_DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR,
    /** Invalid backend provided for selected platform */
    DTFFT_ERROR_INVALID_PLATFORM_BACKEND = CONF_DTFFT_ERROR_INVALID_PLATFORM_BACKEND,
    /** Invalid access mode provided */
    DTFFT_ERROR_INVALID_ACCESS_MODE = CONF_DTFFT_ERROR_INVALID_ACCESS_MODE,
    /** CUDA support is not available for compression */
    DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED = CONF_DTFFT_ERROR_COMPRESSION_CUDA_NOT_SUPPORTED,
    /** Invalid compression rate */
    DTFFT_ERROR_COMPRESSION_INVALID_RATE = CONF_DTFFT_ERROR_COMPRESSION_INVALID_RATE,
    /** Invalid compression precision */
    DTFFT_ERROR_COMPRESSION_INVALID_PRECISION = CONF_DTFFT_ERROR_COMPRESSION_INVALID_PRECISION,
    /** Invalid compression tolerance */
    DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE = CONF_DTFFT_ERROR_COMPRESSION_INVALID_TOLERANCE,
    /** Invalid compression mode */
    DTFFT_ERROR_COMPRESSION_INVALID_MODE = CONF_DTFFT_ERROR_COMPRESSION_INVALID_MODE,
    /** Invalid compression library */
    DTFFT_ERROR_COMPRESSION_INVALID_LIBRARY = CONF_DTFFT_ERROR_COMPRESSION_INVALID_LIBRARY
} dtfft_error_t;

/** This enum lists valid `execute_type` parameters that can be passed to `::dtfft_execute`. */
typedef enum {
    /** Perform XYZ --> YZX --> ZXY plan execution (Forward) */
    DTFFT_EXECUTE_FORWARD = CONF_DTFFT_EXECUTE_FORWARD,

    /** Perform ZXY --> YZX --> XYZ plan execution (Backward) */
    DTFFT_EXECUTE_BACKWARD = CONF_DTFFT_EXECUTE_BACKWARD
} dtfft_execute_t;

/** This enum lists valid transpose_type parameters that can be passed to `::dtfft_transpose`. */
typedef enum {
    /** Transpose from Fortran X aligned to Fortran Y aligned */
    DTFFT_TRANSPOSE_X_TO_Y = CONF_DTFFT_TRANSPOSE_X_TO_Y,

    /** Transpose from Fortran Y aligned to Fortran X aligned */
    DTFFT_TRANSPOSE_Y_TO_X = CONF_DTFFT_TRANSPOSE_Y_TO_X,

    /** Transpose from Fortran Y aligned to Fortran Z aligned */
    DTFFT_TRANSPOSE_Y_TO_Z = CONF_DTFFT_TRANSPOSE_Y_TO_Z,

    /** Transpose from Fortran Z aligned to Fortran Y aligned */
    DTFFT_TRANSPOSE_Z_TO_Y = CONF_DTFFT_TRANSPOSE_Z_TO_Y,

    /** Transpose from Fortran X aligned to Fortran Z aligned
     * @note This value is valid to pass only in 3D Plan and value returned by `::dtfft_get_z_slab_enabled` must be `true`
     */
    DTFFT_TRANSPOSE_X_TO_Z = CONF_DTFFT_TRANSPOSE_X_TO_Z,

    /** Transpose from Fortran Z aligned to Fortran X aligned
     * @note This value is valid to pass only in 3D Plan and value returned by `::dtfft_get_z_slab_enabled` must be `true`
     */
    DTFFT_TRANSPOSE_Z_TO_X = CONF_DTFFT_TRANSPOSE_Z_TO_X
} dtfft_transpose_t;

/** This enum lists valid `reshape_type` parameters that can be passed to `::dtfft_reshape`. */
typedef enum {
    /** Reshape from X-bricks to X-pencils */
    DTFFT_RESHAPE_X_BRICKS_TO_PENCILS = CONF_DTFFT_RESHAPE_X_BRICKS_TO_PENCILS,

    /** Reshape from X-pencils to X-bricks */
    DTFFT_RESHAPE_X_PENCILS_TO_BRICKS = CONF_DTFFT_RESHAPE_X_PENCILS_TO_BRICKS,

    /** Reshape from Z-bricks to Z-pencils */
    DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS = CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS,

    /** Reshape from Z-pencils to Z-bricks */
    DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS = CONF_DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS,

    /** Reshape from Y-bricks to Y-pencils
     * This is to be used in 2D Plans.
     */
    DTFFT_RESHAPE_Y_BRICKS_TO_PENCILS = CONF_DTFFT_RESHAPE_Z_BRICKS_TO_PENCILS,

    /** Reshape from Y-pencils to Y-bricks
     * This is to be used in 2D Plans.
     */
    DTFFT_RESHAPE_Y_PENCILS_TO_BRICKS = CONF_DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS
} dtfft_reshape_t;

/** This enum lists valid `precision` values that can be passed while creating plan. */
typedef enum {
    /** Use Single precision */
    DTFFT_SINGLE = CONF_DTFFT_SINGLE,
    /** Use Double precision */
    DTFFT_DOUBLE = CONF_DTFFT_DOUBLE
} dtfft_precision_t;

/** This enum lists valid `effort` values that can be passed while creating plan. */
typedef enum {
    /** Create plan as fast as possible */
    DTFFT_ESTIMATE = CONF_DTFFT_ESTIMATE,

    /** Will attempt to find best MPI Grid decomposition.
     * Passing this flag and MPI Communicator with cartesian topology to `dtfft_create_plan_*` is same as `::DTFFT_ESTIMATE`.
     */
    DTFFT_MEASURE = CONF_DTFFT_MEASURE,

    /** Same as `::DTFFT_MEASURE` plus autotune will try to find best backend
     */
    DTFFT_PATIENT = CONF_DTFFT_PATIENT,

    /** Same as `::DTFFT_PATIENT` plus will autotune all possible kernels
     * and reshape backends to find best configuration.
     */
    DTFFT_EXHAUSTIVE = CONF_DTFFT_EXHAUSTIVE
} dtfft_effort_t;

/** This enum lists available FFT executors. */
typedef enum {
    /** Do not create any FFT plans. Creates transpose only plan. */
    DTFFT_EXECUTOR_NONE = CONF_DTFFT_EXECUTOR_NONE,

    /** FFTW3 Executor (Host only) */
    DTFFT_EXECUTOR_FFTW3 = CONF_DTFFT_EXECUTOR_FFTW3,

    /** MKL DFTI Executor (Host only) */
    DTFFT_EXECUTOR_MKL = CONF_DTFFT_EXECUTOR_MKL,

    /** CUFFT Executor (GPU Only) */
    DTFFT_EXECUTOR_CUFFT = CONF_DTFFT_EXECUTOR_CUFFT,

    /** VkFFT Executor (GPU Only) */
    DTFFT_EXECUTOR_VKFFT = CONF_DTFFT_EXECUTOR_VKFFT
} dtfft_executor_t;

/** This enum lists the different R2R FFT kinds. */
typedef enum {
    /** DCT-I (Logical N=2*(n-1), inverse is `::DTFFT_DCT_1`) */
    DTFFT_DCT_1 = CONF_DTFFT_DCT_1,

    /** DCT-II (Logical N=2*n, inverse is `::DTFFT_DCT_3`) */
    DTFFT_DCT_2 = CONF_DTFFT_DCT_2,

    /** DCT-III (Logical N=2*n, inverse is `::DTFFT_DCT_2`) */
    DTFFT_DCT_3 = CONF_DTFFT_DCT_3,

    /** DCT-IV (Logical N=2*n, inverse is `::DTFFT_DCT_4`) */
    DTFFT_DCT_4 = CONF_DTFFT_DCT_4,

    /** DST-I (Logical N=2*(n+1), inverse is `::DTFFT_DST_1`) */
    DTFFT_DST_1 = CONF_DTFFT_DST_1,

    /** DST-II (Logical N=2*n, inverse is `::DTFFT_DST_3`) */
    DTFFT_DST_2 = CONF_DTFFT_DST_2,

    /** DST-III (Logical N=2*n, inverse is `::DTFFT_DST_2`) */
    DTFFT_DST_3 = CONF_DTFFT_DST_3,

    /** DST-IV (Logical N=2*n, inverse is `::DTFFT_DST_4`) */
    DTFFT_DST_4 = CONF_DTFFT_DST_4
} dtfft_r2r_kind_t;

/** Structure to hold plan data */
typedef void* dtfft_plan_t;

/** Structure to hold pencil decomposition info
 *
 * There are two ways users might find pencils useful inside dtFFT:
 * 1. To create a Plan using users's own grid decomposition, you can pass Pencil to Plan constructors.
 * 2. To obtain Pencil from Plan in all possible layouts, in order to run FFT not available in dtFFT.
 *
 * In order to create plan using dtfft_pencil_t, user need to provide `ndims`, `starts` and `counts` arrays, other values will be ignored.
 *
 * When pencil is returned from `::dtfft_get_pencil`, all pencil properties are defined.
 *
 * @see dtfft_get_pencil dtfft_create_plan_r2r_pencil dtfft_create_plan_c2c_pencil dtfft_create_plan_r2c_pencil
 */
typedef struct
{
    /** Aligned dimension ID starting from 1 */
    uint8_t dim;

    /** Number of dimensions in a pencil */
    uint8_t ndims;

    /** Local starts in natural Fortran order. If `ndims` == 2, then only first two elements are defined */
    int32_t starts[3];

    /** Local counts in natural Fortran order. If `ndims` == 2, then only first two elements are defined */
    int32_t counts[3];

    /** Total number of elements in a pencil */
    size_t size;
} dtfft_pencil_t;

/** Real-to-Real Plan constructor.
 *
 * @param[in]      ndims                  Number of dimensions: 2 or 3
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order.
 *                                          dims[0] must be the fastest varying
 * @param[in]      kinds                  Array of size `ndims` containing Real FFT kinds in reverse order.
 *                                          Can be NULL if `executor` == `::DTFFT_EXECUTOR_NONE`
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 Effort level for the plan creation
 * @param[in]      executor               Type of external FFT executor.
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_r2r(
    int8_t ndims,
    const int32_t* dims,
    const dtfft_r2r_kind_t* kinds,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/**
 * @brief Creates a Real-to-Real Plan using a pencil handle.
 *
 * @param[in] pencil       Pencil structure containing local dimensions and starts
 * @param[in] kinds        Array of size `ndims` containing Real FFT kinds in reverse order.
 *                           Can be NULL if `executor` == `::DTFFT_EXECUTOR_NONE`
 * @param[in] comm         MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in] precision    Precision of the transform
 * @param[in] effort       Effort level for the plan creation
 * @param[in] executor     Executor to be used for the plan
 * @param[out] plan        Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_r2r_pencil(
    const dtfft_pencil_t* pencil,
    const dtfft_r2r_kind_t* kinds,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/** Complex-to-Complex Plan constructor.
 *
 * @param[in]      ndims                  Number of dimensions: 2 or 3
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 Effort level for the plan creation
 * @param[in]      executor               Type of external FFT executor.
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_c2c(
    int8_t ndims,
    const int32_t* dims,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/** Complex-to-Complex Plan constructor using a pencil structure.
 *
 * @param[in]      pencil                 Pencil handle
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of the transform
 * @param[in]      effort                 Effort level for the plan creation
 * @param[in]      executor               Executor to be used for the plan
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_c2c_pencil(
    const dtfft_pencil_t* pencil,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/** Real-to-Complex Plan constructor.
 *
 * @param[in]      ndims                  Number of dimensions: 2 or 3
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 Effort level for the plan creation
 * @param[in]      executor               Type of external FFT executor
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_r2c(
    int8_t ndims,
    const int32_t* dims,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/**
 * @brief Creates a Real-to-Complex Plan using a pencil structure.
 *
 * @param[in]      pencil                 Pencil structure containing local dimensions and starts
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of the transform
 * @param[in]      effort                 Effort level for the plan creation
 * @param[in]      executor               Executor to be used for the plan
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_r2c_pencil(
    const dtfft_pencil_t* pencil,
    MPI_Comm comm,
    dtfft_precision_t precision,
    dtfft_effort_t effort,
    dtfft_executor_t executor,
    dtfft_plan_t* plan);

/** @brief Checks if plan is using Z-slab optimization.
 * If `true` then flags `::DTFFT_TRANSPOSE_X_TO_Z` and `::DTFFT_TRANSPOSE_Z_TO_X` will be valid to pass to `::dtfft_transpose`.
 *
 * @param[in]      plan              Plan handle
 * @param[out]     is_z_slab_enabled Boolean value if Z-slab is used.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_z_slab_enabled(dtfft_plan_t plan, bool* is_z_slab_enabled);

/** @brief Checks if plan is using Y-slab optimization.
 * If `true` then `dtFFT` will skip the transpose step between Y and Z aligned layouts during call to `::dtfft_execute`.
 *
 * @param[in]      plan              Plan handle
 * @param[out]     is_y_slab_enabled Boolean value if Y-slab is used.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_y_slab_enabled(dtfft_plan_t plan, bool* is_y_slab_enabled);

/** @brief Plan execution. Neither `in` nor `out` are allowed to be `NULL`. The same pointer can safely be passed to both `in` and `out`.
 *
 * @param[in]      plan            Plan handle
 * @param[inout]   in              Incoming buffer
 * @param[out]     out             Result buffer
 * @param[in]      execute_type    Type of transform.
 * @param[inout]   aux             Optional auxiliary buffer. Can be `NULL`.
 *                                 If `NULL` during first call to this function, then auxiliary will be allocated
 *                                 internally and freed after call to `::dtfft_destroy`.
 *                                 If provided, must be at least `::dtfft_get_aux_bytes` bytes.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 *
 * @note This function is not supported for transpose-only R2C plans.
 */
dtfft_error_t
dtfft_execute(dtfft_plan_t plan, void* in, void* out, dtfft_execute_t execute_type, void* aux);

/** @brief Transpose data in single dimension, e.g. X align -> Y align
 * \attention `in` and `out` cannot be the same pointers
 *
 * @param[in]      plan            Plan handle
 * @param[inout]   in              Incoming buffer
 * @param[out]     out             Transposed buffer
 * @param[in]      transpose_type  Type of transpose.
 * @param[inout]   aux             Optional auxiliary buffer. Can be `NULL`.
 *                                 If provided, must be at least `::dtfft_get_alloc_size` elements.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_transpose(dtfft_plan_t plan, void* in, void* out, dtfft_transpose_t transpose_type, void* aux);

/** @brief Helper type to manage asynchronous operations */
typedef void* dtfft_request_t;

/**
 * @brief Starts an asynchronous transpose operation.
 *
 * @param[in]      plan             Plan handle
 * @param[inout]   in               Incoming buffer
 * @param[out]     out              Transposed buffer
 * @param[in]      transpose_type   Type of transpose.
 * @param[inout]   aux              Optional auxiliary buffer. Can be `NULL`.
 *                                  If provided, must be at least `::dtfft_get_alloc_size` elements.
 * @param[out]     request          Handle to manage the asynchronous operation.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 *
 * @note Both `in` and `out` buffers must not be changed or freed until call to `::dtfft_transpose_end`.
 */
dtfft_error_t
dtfft_transpose_start(dtfft_plan_t plan, void* in, void* out, dtfft_transpose_t transpose_type, void* aux, dtfft_request_t* request);

/**
 * @brief Finalizes an asynchronous transpose operation.
 *
 * @param[in]      plan             Plan handle
 * @param[inout]   request          Handle to manage the asynchronous operation.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_transpose_end(dtfft_plan_t plan, dtfft_request_t request);

/**
 * @brief Executes data reshape between brick and pencil decompositions.
 *
 * @param[in]       plan            Plan handle
 * @param[inout]    in              Input pointer
 * @param[out]      out             Output pointer
 * @param[in]       reshape_type    Type of reshape.
 * @param[inout]    aux             Optional auxiliary buffer. Can be `NULL`.
 *                                  If provided, must be at least `::dtfft_get_alloc_size` elements.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_reshape(dtfft_plan_t plan, void* in, void* out, dtfft_reshape_t reshape_type, void* aux);

/**
 * @brief Starts an asynchronous reshape operation.
 *
 * @param[in]       plan            Plan handle
 * @param[inout]    in              Input pointer
 * @param[out]      out             Output pointer
 * @param[in]       reshape_type    Type of reshape.
 * @param[inout]    aux             Optional auxiliary buffer. Can be `NULL`.
 *                                  If provided, must be at least `::dtfft_get_alloc_size` elements.
 * @param[out]      request         Handle to manage the asynchronous operation.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 *
 * @note Both `in` and `out` buffers must not be changed or freed until call to `::dtfft_reshape_end`.
 */
dtfft_error_t
dtfft_reshape_start(dtfft_plan_t plan, void* in, void* out, dtfft_reshape_t reshape_type, void* aux, dtfft_request_t* request);

/**
 * @brief Finalizes an asynchronous reshape operation.
 *
 * @param[in]       plan            Plan handle
 * @param[inout]    request         Handle to manage the asynchronous operation.
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_reshape_end(dtfft_plan_t plan, dtfft_request_t request);

/** @brief Plan Destructor.
 *
 * @param[inout]    plan            Plan handle
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_destroy(dtfft_plan_t* plan);

/** @brief Get grid decomposition information. Results may differ on different MPI processes
 *
 * @param[in]      plan            Plan handle
 * @param[out]     in_starts       Starts of local portion of data in `real` space in reversed order
 * @param[out]     in_counts       Number of elements of local portion of data in `real` space in reversed order
 * @param[out]     out_starts      Starts of local portion of data in `fourier` space in reversed order
 * @param[out]     out_counts      Number of elements of local portion of data in `fourier` space in reversed order
 * @param[out]     alloc_size      Minimum number of elements to be allocated for `in`, `out` buffers required by `::dtfft_execute`, `::dtfft_transpose`, or `::dtfft_reshape`.
 *                                 Size of each element in bytes can be obtained by calling `::dtfft_get_element_size`.
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_local_sizes(dtfft_plan_t plan, int32_t* in_starts, int32_t* in_counts, int32_t* out_starts, int32_t* out_counts, size_t* alloc_size);

/** @brief Wrapper around `dtfft_get_local_sizes` to obtain number of elements only
 *
 * @param[in]      plan            Plan handle
 * @param[out]     alloc_size      Minimum number of elements to be allocated for `in` and `out` buffers required by `::dtfft_execute`, `::dtfft_transpose`, or `::dtfft_reshape`.
 *                                 Size of each element in bytes can be obtained by calling `::dtfft_get_element_size`.
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_alloc_size(dtfft_plan_t plan, size_t* alloc_size);

/** @brief Gets the number of elements required for auxiliary buffer by `::dtfft_execute`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_size        Size of auxiliary buffer in bytes.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_size(dtfft_plan_t plan, size_t* aux_size);

/** @brief Gets the number of bytes required for auxiliary buffer by `::dtfft_execute`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_bytes       Number of bytes required for auxiliary buffer.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_bytes(dtfft_plan_t plan, size_t* aux_bytes);

/** @brief Gets the number of elements required for auxiliary buffer by `::dtfft_reshape`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_size        Size of auxiliary buffer in elements.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_size_reshape(dtfft_plan_t plan, size_t* aux_size);

/** @brief Gets the number of bytes required for auxiliary buffer by `::dtfft_reshape`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_bytes       Number of bytes required for auxiliary buffer.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_bytes_reshape(dtfft_plan_t plan, size_t* aux_bytes);

/** @brief Gets the number of elements required for auxiliary buffer by `::dtfft_transpose`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_size        Size of auxiliary buffer in elements.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_size_transpose(dtfft_plan_t plan, size_t* aux_size);

/** @brief Gets the number of bytes required for auxiliary buffer by `::dtfft_transpose`.
 *
 * @param[in]      plan            Plan handle
 * @param[out]     aux_bytes       Number of bytes required for auxiliary buffer.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_aux_bytes_transpose(dtfft_plan_t plan, size_t* aux_bytes);

#ifdef DTFFT_WITH_COMPRESSION
/**
 * @brief Prints compression-related information to stdout
 *
 * @param[in]     plan            Plan handle
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 **/
dtfft_error_t
dtfft_report_compression(dtfft_plan_t plan);
#endif

/**
 * @brief Gets the string description of an error code
 *
 * @param[in]       error_code      Error code to convert to string
 * @return Error string explaining error.
 */
const char*
dtfft_get_error_string(dtfft_error_t error_code);

/**
 * @brief Gets the string description of a precision level
 *
 * @param[in]       precision      Precision level to convert to string
 * @return String representation of `::dtfft_precision_t`.
 */
const char*
dtfft_get_precision_string(dtfft_precision_t precision);

/**
 * @brief Gets the string description of an executor type
 *
 * @param[in]       executor       Executor type to convert to string
 * @return String representation of `::dtfft_executor_t`.
 */
const char*
dtfft_get_executor_string(dtfft_executor_t executor);

/** This enum represents different data layouts used in dtFFT and it should be used to retrieve layout information from plans. */
typedef enum {
    /** X-brick layout: data is distributed along all dimensions */
    DTFFT_LAYOUT_X_BRICKS = CONF_DTFFT_LAYOUT_X_BRICKS,
    /** X-pencil layout: data is distributed along Y and Z dimensions */
    DTFFT_LAYOUT_X_PENCILS = CONF_DTFFT_LAYOUT_X_PENCILS,
    /** X-pencil layout obtained after executing FFT for R2C plan: data is distributed along Y and Z dimensions */
    DTFFT_LAYOUT_X_PENCILS_FOURIER = CONF_DTFFT_LAYOUT_X_PENCILS_FOURIER,
    /** Y-pencil layout: data is distributed along X and Z dimensions */
    DTFFT_LAYOUT_Y_PENCILS = CONF_DTFFT_LAYOUT_Y_PENCILS,
    /** Z-pencil layout: data is distributed along X and Y dimensions */
    DTFFT_LAYOUT_Z_PENCILS = CONF_DTFFT_LAYOUT_Z_PENCILS,
    /** Z-brick layout: data is distributed along all dimensions */
    DTFFT_LAYOUT_Z_BRICKS = CONF_DTFFT_LAYOUT_Z_BRICKS
} dtfft_layout_t;

/**
 * @brief Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation,
 * that is unavailable in dtFFT.
 *
 * @param[in]     plan            Plan handle
 * @param[in]     layout          Required layout of the pencil
 * @param[out]    pencil          Pencil data
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_pencil(dtfft_plan_t plan, dtfft_layout_t layout, dtfft_pencil_t* pencil);

/**
 * @brief Obtains number of bytes required to store single element by this plan.
 *
 * @param[in]     plan            Plan handle
 * @param[out]    element_size    Size of element in bytes
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_element_size(dtfft_plan_t plan, size_t* element_size);

/**
 * @brief Returns minimum number of bytes required for in and out buffers.
 *
 * This function is a combination of two calls: `::dtfft_get_alloc_size` and `::dtfft_get_element_size`.
 * Returns minimum number of bytes to be allocated for `in` and `out` buffers required by `::dtfft_execute`, `::dtfft_transpose`, or `::dtfft_reshape`.
 *
 * @param[in]     plan            Plan handle
 * @param[out]    alloc_bytes     Number of bytes required
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_alloc_bytes(dtfft_plan_t plan, size_t* alloc_bytes);

/**
 * @brief Allocates memory specific for this plan
 *
 * @param[in]     plan            Plan handle
 * @param[in]     alloc_bytes     Number of bytes to allocate
 * @param[out]    ptr             Allocated pointer
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_mem_alloc(dtfft_plan_t plan, size_t alloc_bytes, void** ptr);

/**
 * @brief Frees memory specific for this plan
 *
 * @param[in]     plan            Plan handle
 * @param[inout]  ptr             Allocated pointer
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_mem_free(dtfft_plan_t plan, void* ptr);

/**
 * @brief Prints plan-related information to stdout
 *
 * @param[in]     plan            Plan handle
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 **/
dtfft_error_t
dtfft_report(dtfft_plan_t plan);

/**
 * @brief Returns FFT executor used in plan.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     executor       FFT Executor
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_executor(dtfft_plan_t plan, dtfft_executor_t* executor);

/**
 * @brief Returns precision of the plan.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     precision      Precision of the plan
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_precision(dtfft_plan_t plan, dtfft_precision_t* precision);

/**
 * @brief Returns global dimensions of the plan.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     ndims          Number of dimensions in plan. User can pass NULL if this value is not needed.
 * @param[out]     dims           Pointer of size `ndims` containing global dimensions in reverse order
 *                                dims[0] is the fastest varying. User can pass NULL if this value is not needed.
 *
 * @note Do not free `dims` array, it is freed when the `dtfft_plan_t` is destroyed.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_dims(dtfft_plan_t plan, int8_t* ndims, const int32_t* dims[]);

/**
 * @brief Returns grid decomposition dimensions of the plan.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     ndims          Number of dimensions in plan. User can pass NULL if this value is not needed.
 * @param[out]     grid_dims      Pointer of size `ndims` containing grid decomposition dimensions in reverse order
 *                                grid_dims[0] is the fastest varying and is always equal to 1.
 *                                User can pass NULL if this value is not needed.
 *
 * @note Do not free `grid_dims` array, it is freed when the `dtfft_plan_t` is destroyed.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_grid_dims(dtfft_plan_t plan, int8_t* ndims, const int32_t* grid_dims[]);

/** This enum lists the different available backend options.
 * @see dtfft_get_backend_string, dtfft_get_backend
 */
typedef enum {
    /** @brief Backend that uses MPI datatypes.
     * @details This is default backend for Host platform.
     *
     * Not really recommended to use for GPU usage, since it is a 'million' times slower than other backends.
     * Not available for autotune when `effort` is `::DTFFT_PATIENT` on CUDA platform.
     */
    DTFFT_BACKEND_MPI_DATATYPE = CONF_DTFFT_BACKEND_MPI_DATATYPE,

    /** MPI peer-to-peer algorithm */
    DTFFT_BACKEND_MPI_P2P = CONF_DTFFT_BACKEND_MPI_P2P,

    /** MPI peer-to-peer algorithm with overlapping data copying and unpacking */
    DTFFT_BACKEND_MPI_P2P_PIPELINED = CONF_DTFFT_BACKEND_MPI_P2P_PIPELINED,

    /** MPI backend using MPI_Alltoallv */
    DTFFT_BACKEND_MPI_A2A = CONF_DTFFT_BACKEND_MPI_A2A,

    /** MPI backend using one-sided communications */
    DTFFT_BACKEND_MPI_RMA = CONF_DTFFT_BACKEND_MPI_RMA,

    /** MPI backend using pipelined one-sided communications */
    DTFFT_BACKEND_MPI_RMA_PIPELINED = CONF_DTFFT_BACKEND_MPI_RMA_PIPELINED,

    /** MPI peer-to-peer algorithm with scheduled communication */
    DTFFT_BACKEND_MPI_P2P_SCHEDULED = CONF_DTFFT_BACKEND_MPI_P2P_SCHEDULED,

    /** MPI peer-to-peer pipelined algorithm with overlapping packing,
     * exchange and unpacking with scheduled communication */
    DTFFT_BACKEND_MPI_P2P_FUSED = CONF_DTFFT_BACKEND_MPI_P2P_FUSED,

    /** MPI RMA pipelined algorithm with overlapping packing,
     * exchange and unpacking with scheduled communication
    */
    DTFFT_BACKEND_MPI_RMA_FUSED = CONF_DTFFT_BACKEND_MPI_RMA_FUSED,

    /** Extension of Backend.MPI_P2P_FUSED
     * Data is getting compressed before sending and decompressed after receiving
     */
    DTFFT_BACKEND_MPI_P2P_COMPRESSED = CONF_DTFFT_BACKEND_MPI_P2P_COMPRESSED,

    /** Extension of Backend.MPI_RMA_FUSED
     * Data is getting compressed before sending and decompressed after receiving
     */
    DTFFT_BACKEND_MPI_RMA_COMPRESSED = CONF_DTFFT_BACKEND_MPI_RMA_COMPRESSED,

    /** NCCL backend */
    DTFFT_BACKEND_NCCL = CONF_DTFFT_BACKEND_NCCL,

    /** NCCL backend with overlapping data copying and unpacking */
    DTFFT_BACKEND_NCCL_PIPELINED = CONF_DTFFT_BACKEND_NCCL_PIPELINED,

    /** NCCL backend that performs compression before data exchange and decompression after. */
    DTFFT_BACKEND_NCCL_COMPRESSED = CONF_DTFFT_BACKEND_NCCL_COMPRESSED,

    /** cuFFTMp backend */
    DTFFT_BACKEND_CUFFTMP = CONF_DTFFT_BACKEND_CUFFTMP,

    /** cuFFTMp backend that uses additional buffer to avoid extra copy and gain performance */
    DTFFT_BACKEND_CUFFTMP_PIPELINED = CONF_DTFFT_BACKEND_CUFFTMP_PIPELINED,

    /** Adaptive backend selection: during plan creation dtFFT benchmarks multiple
     * backends and selects the fastest backend independently for each transpose/reshape operation.
     * The selection is fixed for the lifetime of the plan.
     *
     * @note Can only be used when effort >= `::DTFFT_PATIENT`.
     * @note Currently only available for HOST execution platform
    */
    DTFFT_BACKEND_ADAPTIVE = CONF_DTFFT_BACKEND_ADAPTIVE
} dtfft_backend_t;

/** This enum specifies at which stage the local transposition is performed during global exchange.
 * It affects only Generic backends that perform explicit packing/unpacking.
*/
typedef enum {
    /** Perform transposition during the packing stage (Sender side). */
    DTFFT_TRANSPOSE_MODE_PACK = CONF_DTFFT_TRANSPOSE_MODE_PACK,

    /** Perform transposition during the unpacking stage (Receiver side). */
    DTFFT_TRANSPOSE_MODE_UNPACK = CONF_DTFFT_TRANSPOSE_MODE_UNPACK
} dtfft_transpose_mode_t;

/** This enum lists valid `access_mode` parameters that can be passed to `::dtfft_config_t`. */
typedef enum {
    /** Optimize for write access (Aligned writing).
     * This is the default mode.
     */
    DTFFT_ACCESS_MODE_WRITE = CONF_DTFFT_ACCESS_MODE_WRITE,

    /** Optimize for read access (Aligned reading) */
    DTFFT_ACCESS_MODE_READ = CONF_DTFFT_ACCESS_MODE_READ
} dtfft_access_mode_t;

#ifdef DTFFT_WITH_COMPRESSION
/** This enum lists valid compression mode parameters. */
typedef enum {
    /** Lossless compression mode */
    DTFFT_COMPRESSION_MODE_LOSSLESS = CONF_DTFFT_COMPRESSION_MODE_LOSSLESS,

    /** Fixed rate compression mode */
    DTFFT_COMPRESSION_MODE_FIXED_RATE = CONF_DTFFT_COMPRESSION_MODE_FIXED_RATE,

    /** Fixed precision compression mode */
    DTFFT_COMPRESSION_MODE_FIXED_PRECISION = CONF_DTFFT_COMPRESSION_MODE_FIXED_PRECISION,

    /** Fixed accuracy compression mode */
    DTFFT_COMPRESSION_MODE_FIXED_ACCURACY = CONF_DTFFT_COMPRESSION_MODE_FIXED_ACCURACY
} dtfft_compression_mode_t;

/** This enum lists valid compression library parameters. */
typedef enum {
    /** ZFP compression library */
    DTFFT_COMPRESSION_LIB_ZFP = CONF_DTFFT_COMPRESSION_LIB_ZFP
} dtfft_compression_lib_t;

/** Struct that specifies compression configuration */
typedef struct {
    /** Compression library to use */
    dtfft_compression_lib_t compression_lib;

    /** Compression mode to use */
    dtfft_compression_mode_t compression_mode;

    /** Rate for `::DTFFT_COMPRESSION_MODE_FIXED_RATE` */
    double rate;

    /** Precision for `::DTFFT_COMPRESSION_MODE_FIXED_PRECISION` */
    int32_t precision;

    /** Tolerance for `::DTFFT_COMPRESSION_MODE_FIXED_ACCURACY` */
    double tolerance;
} dtfft_compression_config_t;
#endif

#ifdef DTFFT_WITH_CUDA
/**
 * @brief `dtFFT` stream representation.
 *
 * @details For CUDA platform this should be casted from `cudaStream_t`.
 *
 * **Example**
 * @code
 *  cudaStream_t stream;
 *  cudaStreamCreate(&stream);
 *  dtfft_stream_t dtfftStream = (dtfft_stream_t)stream;
 * @endcode
 */
typedef void* dtfft_stream_t;

/** Enum that specifies the execution platform, such as Host, CUDA, or HIP */
typedef enum {
    /** Host */
    DTFFT_PLATFORM_HOST = CONF_DTFFT_PLATFORM_HOST,
    /** CUDA */
    DTFFT_PLATFORM_CUDA = CONF_DTFFT_PLATFORM_CUDA
} dtfft_platform_t;

/**
 * @brief Returns stream associated with ``dtFFT`` plan.
 * This can either be stream passed by user to `::dtfft_set_config` or stream created internally.
 * Returns NULL pointer if plan's platform is `::DTFFT_PLATFORM_HOST`.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     stream         CUDA stream associated with plan
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_stream(dtfft_plan_t plan, dtfft_stream_t* stream);

/**
 * @brief Returns plan execution platform .
 *
 * @param[in]        plan           Plan handle
 * @param[out]       platform       Plan platform
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_platform(dtfft_plan_t plan, dtfft_platform_t* platform);
#endif

/**
 * @brief Returns selected backend during autotune if `effort` is `::DTFFT_PATIENT`.
 *
 * If `effort` passed to any create function is `::DTFFT_ESTIMATE` or `::DTFFT_MEASURE`
 * returns value set by `::dtfft_set_config` or default value, which is `::DTFFT_BACKEND_NCCL` for CUDA build and `::DTFFT_BACKEND_MPI_DATATYPE` for host build.
 *
 * @param[in]        plan           Plan handle
 * @param[out]       backend     Selected backend
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_backend(dtfft_plan_t plan, dtfft_backend_t* backend);

/**
 * @brief Returns selected backend for reshape operations.
 *
 * @param[in]        plan           Plan handle
 * @param[out]       backend        Selected backend for reshape operations
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_reshape_backend(dtfft_plan_t plan, dtfft_backend_t* backend);

/**
 * @brief Returns null terminated string with name of backend provided as argument.
 *
 * @param[in]         backend    Backend to represent
 *
 * @return Character representation of backend.
 */
const char*
dtfft_get_backend_string(dtfft_backend_t backend);

/**
 * @brief Returns true if passed backend is pipelined and false otherwise.
 *
 * @param[in]         backend    Backend to check
 * @param[out]        is_pipe    Flag
 *
 * @return `::DTFFT_SUCCESS`
 */
dtfft_error_t
dtfft_get_backend_pipelined(const dtfft_backend_t backend, bool *is_pipe);

/** Struct that can be used to set additional configuration parameters to dtFFT
 * @see dtfft_create_config, dtfft_set_config
 */
typedef struct
{
    /**
     * @brief Should dtFFT print additional information or not.
     *
     * @details Default is `false`.
     */
    bool enable_log;
    /**
     * @brief Enables Z-slab optimization.
     *
     * @details Default is `true`
     *
     * One should consider disabling Z-slab optimization in order to resolve `::DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error
     * or when underlying FFT implementation of 2D plan is too slow.
     *
     * In all other cases, Z-slab is considered to be always faster.
     */
    bool enable_z_slab;

    /**
     * @brief Enables Y-slab optimization.
     *
     * @details Default is `false`.
     *
     * If true, then dtFFT will skip the transpose step between Y and Z aligned layouts during call to `::dtfft_execute`.
     *
     * One should consider disabling Y-slab optimization when the underlying FFT implementation of the 2D plan is too slow.
     *
     * In all other cases, Y-slab is considered to be always faster.
     */
    bool enable_y_slab;

    /**
     * @brief Number of warmup iterations to execute during backend and kernel autotuning when effort level is `::DTFFT_MEASURE` or higher.
     *
     * @details Default is `2`.
     */
    int32_t n_measure_warmup_iters;

    /**
     * @brief Number of iterations to execute during backend and kernel autotuning when effort level is `::DTFFT_MEASURE` or higher.
     *
     * @details Default is `5`.
     */
    int32_t n_measure_iters;

#ifdef DTFFT_WITH_CUDA
    /**
     * @brief Selects platform to execute plan.
     *
     * @details Default is `::DTFFT_PLATFORM_HOST`.
     *
     * This option is only available when dtFFT is built with device support.
     * Even when dtFFT is built with device support, it does not necessarily mean that all plans must be device-related.
     * This enables a single library installation to support both host and CUDA plans.
     *
     * @note This option is only defined when dtFFT is built with CUDA support.
     */
    dtfft_platform_t platform;

    /**
     * @brief Main CUDA stream that will be used in dtFFT.
     *
     * @details This parameter is a placeholder for user to set custom stream.
     * Stream that is actually used by dtFFT plan is returned by `::dtfft_get_stream` function.
     * When user sets stream he is responsible of destroying it.
     *
     * Stream must not be destroyed before call to `::dtfft_destroy`.
     *
     * @note This option is only defined when dtFFT is built with CUDA support.
     */
    dtfft_stream_t stream;

#endif
    /**
     * @brief Backend that will be used by dtFFT when `effort` is `::DTFFT_ESTIMATE` or `::DTFFT_MEASURE`.
     *
     * @details Default for HOST platform is `::DTFFT_BACKEND_MPI_DATATYPE`.
     *
     * Default for CUDA platform is `::DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise `::DTFFT_BACKEND_MPI_P2P`.
     */
    dtfft_backend_t backend;

    /**
     * @brief Backend that will be used by dtFFT for data reshaping from bricks to pencils and vice versa when `effort` is `::DTFFT_ESTIMATE` or `::DTFFT_MEASURE`.
     *
     * @details Default for HOST platform is `::DTFFT_BACKEND_MPI_DATATYPE`.
     *
     * Default for CUDA platform is `::DTFFT_BACKEND_NCCL` if NCCL is enabled, otherwise `::DTFFT_BACKEND_MPI_P2P`.
     */
    dtfft_backend_t reshape_backend;

    /**
     * @brief Should `::DTFFT_BACKEND_MPI_DATATYPE` be considered for autotuning when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`
     *
     * This option works only when executing on a host.
     */
    bool enable_datatype_backend;

    /**
     * @brief Should MPI Backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `false`.
     *
     * This option applies to all `DTFFT_BACKEND_MPI_*` backends, except `::DTFFT_BACKEND_MPI_DATATYPE``.
     *
     * The following applies only to CUDA builds.
     * MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
     * It was noticed that during plan autotuning GPU memory not being freed completely.
     *
     * For example:
     * 1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
     * Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.
     *
     * One of the workarounds is to disable MPI Backends by default, which is done here.
     *
     * Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
     * but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms
     */
    bool enable_mpi_backends;

    /**
     * @brief Should pipelined backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`.
     */
    bool enable_pipelined_backends;

    /**
     * @brief Should RMA backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`.
     */
    bool enable_rma_backends;

    /**
     * @brief Should fused backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`.
     */
    bool enable_fused_backends;

#ifdef DTFFT_WITH_CUDA
    /**
     * @brief Should NCCL Backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`.
     *
     * @note This option is only defined when dtFFT is built with CUDA support.
     */
    bool enable_nccl_backends;

    /**
     * @brief Should NVSHMEM Backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `true`.
     *
     * @note This option is only defined when dtFFT is built with CUDA support.
     */
    bool enable_nvshmem_backends;
#endif

    /**
     * @brief Should dtFFT try to optimize kernel launch parameters during plan creation when `effort` is below `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `false`.
     *
     * Kernel optimization is always enabled for `::DTFFT_EXHAUSTIVE` effort level.
     * Setting this option to true enables kernel optimization for lower effort levels (`::DTFFT_ESTIMATE`, `::DTFFT_MEASURE`, `::DTFFT_PATIENT`).
     * This may increase plan creation time but can improve runtime performance.
     * Since kernel optimization is performed without data transfers, the time increase is usually minimal.
     */
    bool enable_kernel_autotune;

    /**
     * @brief Should dtFFT execute reshapes from pencils to bricks and vice versa in Fourier space during calls to execute.
     *
     * @details Default is `false`.
     *
     * When enabled, data will be in brick layout in Fourier space, which may be useful for certain operations
     * between forward and backward transforms. However, this requires additional data transpositions
     * and will reduce overall FFT performance.
     */
    bool enable_fourier_reshape;

    /**
     * @brief Specifies at which stage the local transposition is performed during global exchange when effort level is below `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `::DTFFT_TRANSPOSE_MODE_PACK`.
     *
     * For `::DTFFT_EXHAUSTIVE` effort level, dtFFT will always choose the best transpose mode based on internal autotuning.
     * 
     * @note This option only takes effect when platform is `::DTFFT_PLATFORM_HOST`
    */
    dtfft_transpose_mode_t transpose_mode;

    /**
     * @brief Specifies the memory access pattern (optimization target) for local transposition.
     *
     * @details Default is `::DTFFT_ACCESS_MODE_WRITE`.
     *
     * This option allows user to force specific access mode (`::DTFFT_ACCESS_MODE_WRITE` or `::DTFFT_ACCESS_MODE_READ`) when autotuning is disabled.
     * When autotuning is enabled (e.g. `effort` is `::DTFFT_EXHAUSTIVE`), this option is ignored and best access mode is selected automatically.
     */
    dtfft_access_mode_t access_mode;

#ifdef DTFFT_WITH_COMPRESSION
    /**
     * @brief Should compressed backends be enabled when `effort` is `::DTFFT_PATIENT` or `::DTFFT_EXHAUSTIVE`.
     *
     * @details Default is `false`.
     *
     * Only fixed-rate compression can be used during autotuning, since it provides predictable performance characteristics and does not require data-dependent decisions at runtime.
     * To enable compressed backends during autotuning, set this option to true, set compression type to `::DTFFT_COMPRESSION_MODE_FIXED_RATE` and provide desired compression rate.
     */
    bool enable_compressed_backends;

    /**
     * @brief Options for compression approach during transpositions
     */
    dtfft_compression_config_t compression_config_transpose;

    /**
     * @brief Options for compression approach during reshape operations
     */
    dtfft_compression_config_t compression_config_reshape;
#endif
} dtfft_config_t;

/**
 * @brief Sets default values to config
 *
 * @param[out]  config  Config to set default values into
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_create_config(dtfft_config_t* config);

/**
 * @brief Set configuration values to dtFFT. In order to take effect should be called before plan creation
 *
 * @param[in]   config  Config to set
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_set_config(const dtfft_config_t* config);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT_H
