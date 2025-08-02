/*
  Copyright (c) 2021, Oleg Shatrov
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
 * @date 2024
 * @brief File containing C API functions of dtFFT Library
 */
#ifndef DTFFT_H
#define DTFFT_H

#include "dtfft_config.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>  // int8_t, int32_t, uint8_t
#include <stdbool.h> // bool
#include <stdio.h>   // fprintf, stderr, size_t

/** dtFFT Major Version */
#define DTFFT_VERSION_MAJOR CONF_DTFFT_VERSION_MAJOR
/** dtFFT Minor Version */
#define DTFFT_VERSION_MINOR CONF_DTFFT_VERSION_MINOR
/** dtFFT Patch Version */
#define DTFFT_VERSION_PATCH CONF_DTFFT_VERSION_PATCH
/** dtFFT Version Code. Can be used in Version comparison */
#define DTFFT_VERSION_CODE CONF_DTFFT_VERSION_CODE
/** Generates Version Code based on Major, Minor, Patch */
#define DTFFT_VERSION(X,Y,Z) CONF_DTFFT_VERSION(X,Y,Z)

/** @return Version Code defined during compilation */
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
#define DTFFT_CALL(call)                                                      \
do {                                                                          \
    dtfft_error_t ierr = call;                                                \
    if( ierr != DTFFT_SUCCESS ) {                                             \
        fprintf(stderr, "dtFFT error in file '%s:%i': %s.\n",                 \
                __FILE__, __LINE__, dtfft_get_error_string( ierr ));          \
        MPI_Abort(MPI_COMM_WORLD, ierr);                                      \
    }                                                                         \
} while (0);


/* If <complex.h> is included, use the C99 complex type.  Otherwise
   define a type bit-compatible with C99 complex */
#if defined(_Complex_I) && defined(complex) && defined(I)
  typedef double _Complex dtfft_complex;
  typedef float _Complex dtfftf_complex;
#else
  typedef double dtfft_complex[2];
  typedef float dtfftf_complex[2];
#endif

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
/** Invalid `dim` passed to `::dtfft_get_pencil` */
  DTFFT_ERROR_INVALID_DIM = CONF_DTFFT_ERROR_INVALID_DIM,
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
/** Calling to `::dtfft_transpose` for R2C plan is not allowed */
  DTFFT_ERROR_R2C_TRANSPOSE_CALLED = CONF_DTFFT_ERROR_R2C_TRANSPOSE_CALLED,
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
/** Invalid stream provided */
  DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM,
/** Invalid GPU backend provided */
  DTFFT_ERROR_GPU_INVALID_BACKEND = CONF_DTFFT_ERROR_GPU_INVALID_BACKEND,
/** Multiple MPI Processes located on same host share same GPU which is not supported */
  DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET,
/** When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform are same in X and Y directions */
  DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN,
/** Passed `effort` ==  `::DTFFT_PATIENT` but all GPU Backends has been disabled by `::dtfft_config_t`*/
  DTFFT_ERROR_GPU_BACKENDS_DISABLED = CONF_DTFFT_ERROR_GPU_BACKENDS_DISABLED,
/** One of pointers passed to `::dtfft_execute` or `::dtfft_transpose` cannot be accessed from device */
  DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR,
/** One of pointers passed to `::dtfft_execute` or `::dtfft_transpose` is not an `NVSHMEM` pointer */
  DTFFT_ERROR_NOT_NVSHMEM_PTR = CONF_DTFFT_ERROR_NOT_NVSHMEM_PTR,
/** Invalid platform provided */
  DTFFT_ERROR_INVALID_PLATFORM = CONF_DTFFT_ERROR_INVALID_PLATFORM,
/** Invalid executor provided for selected platform */
  DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE = CONF_DTFFT_ERROR_INVALID_PLATFORM_EXECUTOR_TYPE
} dtfft_error_t;


/** This enum lists valid `execute_type` parameters that can be passed to `::dtfft_execute`. */
typedef enum {
/** Perform XYZ --> YXZ --> ZXY plan execution (Forward) */
  DTFFT_EXECUTE_FORWARD = CONF_DTFFT_EXECUTE_FORWARD,

/** Perform ZXY --> YXZ --> XYZ plan execution (Backward) */
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

/** Same as `::DTFFT_MEASURE` plus cycle through various send and receive MPI_Datatypes.
 * For GPU Build this flag will run autotune procedure to find best backend
 */
  DTFFT_PATIENT = CONF_DTFFT_PATIENT
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
typedef void *dtfft_plan_t;

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
typedef struct {
/** Aligned dimension id starting from 1 */
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
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order
 *                                          dims[0] must be the fastest varying
 * @param[in]      kinds                  Array of size `ndims` containing Real FFT kinds in reverse order.
 *                                          Can be NULL if `executor` == `::DTFFT_EXECUTOR_NONE`
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]      executor               Type of external FFT executor.
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_r2r(
  const int8_t ndims,
  const int32_t *dims,
  const dtfft_r2r_kind_t *kinds,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);

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
  const dtfft_pencil_t *pencil,
  const dtfft_r2r_kind_t *kinds,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);


/** Complex-to-Complex Plan constructor.
 *
 * @param[in]      ndims                  Number of dimensions: 2 or 3
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]      executor               Type of external FFT executor.
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 */
dtfft_error_t
dtfft_create_plan_c2c(
  const int8_t ndims,
  const int32_t *dims,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);

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
  const dtfft_pencil_t *pencil,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);


#if !defined(DTFFT_TRANSPOSE_ONLY)
/** Real-to-Complex Plan constructor.
 *
 * @param[in]      ndims                  Number of dimensions: 2 or 3
 * @param[in]      dims                   Array of size `ndims` containing global dimensions in reverse order
 * @param[in]      comm                   MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
 * @param[in]      precision              Precision of transform.
 * @param[in]      effort                 How thoroughly `dtFFT` searches for the optimal plan
 * @param[in]      executor               Type of external FFT executor
 * @param[out]     plan                   Plan handle ready to be executed
 *
 * @return `::DTFFT_SUCCESS` if plan was created, error code otherwise
 *
 * @note Parameter `executor` cannot be `::DTFFT_EXECUTOR_NONE`. Use C2C plan instead.
 * @note This function is only present in the API when ``dtFFT`` was compiled with any external FFT.
 */
dtfft_error_t
dtfft_create_plan_r2c(
  const int8_t ndims,
  const int32_t *dims,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);


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
 *
 * @note Parameter `executor` cannot be `::DTFFT_EXECUTOR_NONE`. Use C2C plan instead.
 * @note This function is only present in the API when ``dtFFT`` was compiled with any external FFT.
 */
dtfft_error_t
dtfft_create_plan_r2c_pencil(
  const dtfft_pencil_t *pencil,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort,
  const dtfft_executor_t executor,
  dtfft_plan_t *plan);
#endif

/** @brief Checks if plan is using Z-slab optimization.
 * If `true` then flags `::DTFFT_TRANSPOSE_X_TO_Z` and `::DTFFT_TRANSPOSE_Z_TO_X` will be valid to pass to `::dtfft_transpose`.
 *
 * @param[in]      plan              Plan handle
 * @param[out]     is_z_slab_enabled Boolean value if Z-slab is used.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_z_slab_enabled(dtfft_plan_t plan, bool *is_z_slab_enabled);


/** @brief Plan execution. Neither `in` nor `out` are allowed to be `NULL`. The same pointer can safely be passed to both `in` and `out`.
 *
 * @param[in]      plan            Plan handle
 * @param[inout]   in              Incoming buffer
 * @param[out]     out             Result buffer
 * @param[in]      execute_type    Type of transform.
 * @param[inout]   aux             Optional auxiliary buffer. Can be `NULL`.
 *                                 If `NULL` during first call to this function, then auxiliary will be allocated
 *                                 internally and freed after call to `::dtfft_destroy`
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_execute(dtfft_plan_t plan, void *in, void *out, const dtfft_execute_t execute_type, void *aux);


/** @brief Transpose data in single dimension, e.g. X align -> Y align
 * \attention `in` and `out` cannot be the same pointers
 *
 * @param[in]      plan            Plan handle
 * @param[inout]   in              Incoming buffer
 * @param[out]     out             Transposed buffer
 * @param[in]      transpose_type  Type of transpose.
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 *
 * @note This function is not supported for R2C plans. Use R2R or C2C plan instead.
 */
dtfft_error_t
dtfft_transpose(dtfft_plan_t plan, void *in, void *out, const dtfft_transpose_t transpose_type);


/** @brief Plan Destructor.
 *
 * @param[inout]    plan            Plan handle
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_destroy(dtfft_plan_t *plan);


/** @brief Get grid decomposition information. Results may differ on different MPI processes
 *
 * @param[in]      plan            Plan handle
 * @param[out]     in_starts       Starts of local portion of data in `real` space in reversed order
 * @param[out]     in_counts       Number of elements of local portion of data in `real` space in reversed order
 * @param[out]     out_starts      Starts of local portion of data in `fourier` space in reversed order
 * @param[out]     out_counts      Number of elements of local portion of data in `fourier` space in reversed order
 * @param[out]     alloc_size      Minimum number of elements to be allocated for `in`, `out` or `aux` buffers.
 *                                 Size of each element in bytes can be obtained by calling `::dtfft_get_element_size`.
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_local_sizes(dtfft_plan_t plan, int32_t *in_starts, int32_t *in_counts, int32_t *out_starts, int32_t *out_counts, size_t *alloc_size);


/** @brief Wrapper around `dtfft_get_local_sizes` to obtain number of elements only
 *
 * @param[in]      plan            Plan handle
 * @param[out]     alloc_size      Minimum number of elements to be allocated for `in`, `out` or `aux` buffers.
 *                                 Size of each element in bytes can be obtained by calling `::dtfft_get_element_size`.
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_alloc_size(dtfft_plan_t plan, size_t *alloc_size);


/**
 * @brief Gets the string description of an error code
 *
 * @param[in]       error_code      Error code to convert to string
 * @return Error string explaining error.
 */
const char *
dtfft_get_error_string(const dtfft_error_t error_code);


/**
 * @brief Gets the string description of a precision level
 *
 * @param[in]       precision      Precision level to convert to string
 * @return String representation of `::dtfft_precision_t`.
 */
const char *
dtfft_get_precision_string(const dtfft_precision_t precision);


/**
 * @brief Gets the string description of an executor type
 *
 * @param[in]       executor       Executor type to convert to string
 * @return String representation of `::dtfft_executor_t`.
 */
const char *
dtfft_get_executor_string(const dtfft_executor_t executor);

/**
 * @brief Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation,
 * that is unavailable in dtFFT.
 *
 * @param[in]     plan            Plan handle
 * @param[in]     dim             Required dimension:
 *                                  - 0 for XYZ layout (real space, R2C only)
 *                                  - 1 for XYZ layout (real space for C2C and R2R plans and fourier space for R2C plans)
 *                                  - 2 for YXZ layout
 *                                  - 3 for ZXY layout
 * @param[out]    pencil          Pencil data
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_pencil(dtfft_plan_t plan, int32_t dim, dtfft_pencil_t *pencil);


/**
 * @brief Obtains number of bytes required to store single element by this plan.
 *
 * @param[in]     plan            Plan handle
 * @param[out]    element_size    Size of element in bytes
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_element_size(dtfft_plan_t plan, size_t *element_size);


/**
 * @brief Returns minimum number of bytes required to execute plan.
 * 
 * This function is a combination of two calls: `::dtfft_get_alloc_size` and `::dtfft_get_element_size`
 *
 * @param[in]     plan            Plan handle
 * @param[out]    alloc_bytes     Number of bytes required
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_alloc_bytes(dtfft_plan_t plan, size_t *alloc_bytes);


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
dtfft_mem_alloc(dtfft_plan_t plan, size_t alloc_bytes, void **ptr);


/**
 * @brief Frees memory specific for this plan
 *
 * @param[in]     plan            Plan handle
 * @param[inout]  ptr             Allocated pointer
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_mem_free(dtfft_plan_t plan, void *ptr);

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
dtfft_get_executor(dtfft_plan_t plan, dtfft_executor_t *executor);

/**
 * @brief Returns precision of the plan.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     precision      Precision of the plan
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_precision(dtfft_plan_t plan, dtfft_precision_t *precision);

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
dtfft_get_dims(dtfft_plan_t plan, int8_t *ndims, const int32_t *dims[]);


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
typedef void *dtfft_stream_t;

/** Enum that specifies the execution platform, such as Host, CUDA, or HIP */
typedef enum {
/** Host */
  DTFFT_PLATFORM_HOST = CONF_DTFFT_PLATFORM_HOST,
/** CUDA */
  DTFFT_PLATFORM_CUDA = CONF_DTFFT_PLATFORM_CUDA
} dtfft_platform_t;


/** This enum lists the different available GPU backend options.
 * @see dtfft_get_backend_string, dtfft_get_backend
*/
typedef enum {
/** @brief Backend that uses MPI datatypes
 * @details Not really recommended to use, since it is a million times slower than other backends.
 * It is present here just to show how slow MPI Datatypes are for GPU usage.
 */
  DTFFT_BACKEND_MPI_DATATYPE = CONF_DTFFT_BACKEND_MPI_DATATYPE,

/** MPI peer-to-peer algorithm */
  DTFFT_BACKEND_MPI_P2P = CONF_DTFFT_BACKEND_MPI_P2P,

/** MPI peer-to-peer algorithm with overlapping data copying and unpacking */
  DTFFT_BACKEND_MPI_P2P_PIPELINED = CONF_DTFFT_BACKEND_MPI_P2P_PIPELINED,

/** MPI backend using MPI_Alltoallv */
  DTFFT_BACKEND_MPI_A2A = CONF_DTFFT_BACKEND_MPI_A2A,

/** NCCL backend */
  DTFFT_BACKEND_NCCL = CONF_DTFFT_BACKEND_NCCL,

/** NCCL backend with overlapping data copying and unpacking */
  DTFFT_BACKEND_NCCL_PIPELINED = CONF_DTFFT_BACKEND_NCCL_PIPELINED,

/** cuFFTMp backend */
  DTFFT_BACKEND_CUFFTMP = CONF_DTFFT_BACKEND_CUFFTMP,

/** cuFFTMp backend that uses additional buffer to avoid extra copy and gain performance */
  DTFFT_BACKEND_CUFFTMP_PIPELINED = CONF_DTFFT_BACKEND_CUFFTMP_PIPELINED
} dtfft_backend_t;


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
dtfft_get_stream(dtfft_plan_t plan, dtfft_stream_t *stream);


/**
 * @brief Returns selected GPU backend during autotune if `effort` is `::DTFFT_PATIENT`.
 *
 * If `effort` passed to any create function is `::DTFFT_ESTIMATE` or `::DTFFT_MEASURE`
 * returns value set by `::dtfft_set_config` or default value, which is `::DTFFT_BACKEND_NCCL`.
 *
 * @param[in]        plan           Plan handle
 * @param[out]       backend     Selected backend
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_backend(dtfft_plan_t plan, dtfft_backend_t *backend);


/**
 * @brief Returns plan execution platform .
 *
 * @param[in]        plan           Plan handle
 * @param[out]       platform       Plan platform
 *
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_get_platform(dtfft_plan_t plan, dtfft_platform_t *platform);


/**
 * @brief Returns null terminated string with name of backend provided as argument.
 *
 * @param[in]         backend    Backend to represent
 *
 * @return Character representation of backend.
 */
const char *
dtfft_get_backend_string(const dtfft_backend_t backend);
#endif


/** Struct that can be used to set additional configuration parameters to dtFFT
 * @see dtfft_create_config, dtfft_set_config
*/
typedef struct {
/**
 * @brief Should dtFFT use Z-slab optimization or not.
 *
 * @details Default is `true`
 *
 * One should consider disabling Z-slab optimization in order to resolve `::DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error
 * OR when underlying FFT implementation of 2D plan is too slow.
 *
 * In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
 */
  bool enable_z_slab;

#ifdef DTFFT_WITH_CUDA
/**
 * @brief Selects platform to execute plan.
 *
 * Default is `::DTFFT_PLATFORM_HOST`
 *
 * @details This option is only defined in a build with device support.
 * Even when dtFFT is built with device support, it does not necessarily mean that all plans must be device-related.
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
 */
  dtfft_stream_t stream;

/**
 * @brief Backend that will be used by dtFFT when `effort` is `::DTFFT_ESTIMATE` or `::DTFFT_MEASURE`.
 *
 * @details Default is `::DTFFT_BACKEND_NCCL`
 */
  dtfft_backend_t backend;

/**
 * @brief Should MPI GPU Backends be enabled when `effort` is `::DTFFT_PATIENT` or not.
 *
 * @details Default is `false`
 *
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
 * @brief Should pipelined GPU backends be enabled when `effort` is `::DTFFT_PATIENT` or not.
 *
 * @details Default is `true`
 *
 * @note Pipelined backends require additional buffer that user has no control over.
 */
  bool enable_pipelined_backends;

/**
 * @brief Should NCCL Backends be enabled when `effort` is `::DTFFT_PATIENT` or not.
 * @details Default is true.
 */
  bool enable_nccl_backends;

/**
 * @brief Should NVSHMEM Backends be enabled when `effort` is `::DTFFT_PATIENT` or not.
 * @details Default is true.
 *
 */
  bool enable_nvshmem_backends;
#endif
} dtfft_config_t;

/**
 * @brief Sets default values to config
 *
 * @param[out]  config  Config to set default values into
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_create_config(dtfft_config_t *config);

/**
 * @brief Set configuration values to dtFFT. In order to take effect should be called before plan creation
 *
 * @param[in]   config  Config to set
 * @return `::DTFFT_SUCCESS` on success or error code on failure.
 */
dtfft_error_t
dtfft_set_config(dtfft_config_t config);


#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT_H
