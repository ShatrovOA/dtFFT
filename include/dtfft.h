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

#ifdef DTFFT_WITH_CUDA
#include <cuda_runtime_api.h> // cudaStream_t
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include <stdint.h>
#include <stdbool.h> // bool
#include <stdio.h>   // fprintf, stderr, size_t


#define DTFFT_VERSION_MAJOR CONF_DTFFT_VERSION_MAJOR
#define DTFFT_VERSION_MINOR CONF_DTFFT_VERSION_MINOR
#define DTFFT_VERSION_PATCH CONF_DTFFT_VERSION_PATCH

#define DTFFT_VERSION_CODE CONF_DTFFT_VERSION_CODE
#define DTFFT_VERSION(X,Y,Z) (X) * 100000 + (Y) * 1000 + (Z)


/* If <complex.h> is included, use the C99 complex type.  Otherwise
   define a type bit-compatible with C99 complex */
#if defined(_Complex_I) && defined(complex) && defined(I)
  typedef double _Complex dtfft_complex;
  typedef float _Complex dtfftf_complex;
#else
  typedef double dtfft_complex[2];
  typedef float dtfftf_complex[2];
#endif

// Structure to hold plan data
typedef struct dtfft_plan_private_t *dtfft_plan_t;

/**
 * @brief This enum lists the different error codes that dtfft can return.
 */
typedef enum {
  // Successful execution
  DTFFT_SUCCESS = CONF_DTFFT_SUCCESS,
  // MPI_Init is not called or MPI_Finalize has already been called
  DTFFT_ERROR_MPI_FINALIZED = CONF_DTFFT_ERROR_MPI_FINALIZED,
  // Plan not created
  DTFFT_ERROR_PLAN_NOT_CREATED = CONF_DTFFT_ERROR_PLAN_NOT_CREATED,
  // Invalid `transpose_type` provided
  DTFFT_ERROR_INVALID_TRANSPOSE_TYPE = CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE,
  // Invalid Number of dimensions provided. Valid options are 2 and 3
  DTFFT_ERROR_INVALID_N_DIMENSIONS = CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS,
  // One or more provided dimension sizes <= 0
  DTFFT_ERROR_INVALID_DIMENSION_SIZE = CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE,
  // Invalid communicator type provided
  DTFFT_ERROR_INVALID_COMM_TYPE = CONF_DTFFT_ERROR_INVALID_COMM_TYPE,
  // Invalid `precision` parameter provided
  DTFFT_ERROR_INVALID_PRECISION = CONF_DTFFT_ERROR_INVALID_PRECISION,
  // Invalid `effort_type` parameter provided
  DTFFT_ERROR_INVALID_EFFORT_FLAG = CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG,
  // Invalid `executor_type` parameter provided
  DTFFT_ERROR_INVALID_EXECUTOR_TYPE = CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE,
  // Number of dimensions in provided Cartesian communicator > Number of dimension passed to `create` subroutine
  DTFFT_ERROR_INVALID_COMM_DIMS = CONF_DTFFT_ERROR_INVALID_COMM_DIMS,
  // Passed Cartesian communicator with number of processes in 1st (fastest varying) dimension > 1
  DTFFT_ERROR_INVALID_COMM_FAST_DIM = CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM,
  // For R2R plan, `kinds` parameter must be passed if `executor_type` != `DTFFT_EXECUTOR_NONE`
  DTFFT_ERROR_MISSING_R2R_KINDS = CONF_DTFFT_ERROR_MISSING_R2R_KINDS,
  // Invalid values detected in `kinds` parameter
  DTFFT_ERROR_INVALID_R2R_KINDS = CONF_DTFFT_ERROR_INVALID_R2R_KINDS,
  // Transpose plan is not supported in R2C, use R2R or C2C plan instead
  DTFFT_ERROR_R2C_TRANSPOSE_PLAN = CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN,
  // Inplace transpose is not supported
  DTFFT_ERROR_INPLACE_TRANSPOSE = CONF_DTFFT_ERROR_INPLACE_TRANSPOSE,
  // Invalid `aux` buffer provided
  DTFFT_ERROR_INVALID_AUX = CONF_DTFFT_ERROR_INVALID_AUX,
  // Invalid `dim` passed to `dtfft_get_pencil`
  DTFFT_ERROR_INVALID_DIM = CONF_DTFFT_ERROR_INVALID_DIM,
  // NULL pointer passed
  DTFFT_ERROR_INVALID_USAGE = CONF_DTFFT_ERROR_INVALID_USAGE,
  // Selected `executor_type` do not support R2R FFTs
  DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED = CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED,
  // // cufftMp backends support only 3d plan
  // DTFFT_ERROR_CUFFTMP_2D_PLAN = CONF_DTFFT_ERROR_CUFFTMP_2D_PLAN,
  // Invalid stream provided
  DTFFT_ERROR_GPU_INVALID_STREAM = CONF_DTFFT_ERROR_GPU_INVALID_STREAM,
  // Invalid GPU backend provided
  DTFFT_ERROR_GPU_INVALID_BACKEND = CONF_DTFFT_ERROR_GPU_INVALID_BACKEND,
  // Multiple MPI Processes located on same host share same GPU which is not supported
  DTFFT_ERROR_GPU_NOT_SET = CONF_DTFFT_ERROR_GPU_NOT_SET,
  // When using R2R FFT and executor type is vkFFT and plan uses Z-slab optimization, it is required that types of R2R transform
  // are same in X and Y directions
  DTFFT_ERROR_VKFFT_R2R_2D_PLAN = CONF_DTFFT_ERROR_VKFFT_R2R_2D_PLAN,
  // One of pointers passed to `dtfft_execute` or `dtfft_transpose` cannot be accessed from device
  DTFFT_ERROR_NOT_DEVICE_PTR = CONF_DTFFT_ERROR_NOT_DEVICE_PTR
} dtfft_error_code_t;


/**
 * @brief This enum lists the different execution flags that can be passed to `dtfft_execute`.
 */
typedef enum {
// Perform XYZ --> YXZ --> ZXY plan execution
  DTFFT_TRANSPOSE_OUT = CONF_DTFFT_TRANSPOSE_OUT,
// Perform ZXY --> YXZ --> XYZ plan execution
  DTFFT_TRANSPOSE_IN = CONF_DTFFT_TRANSPOSE_IN
} dtfft_execute_type_t;


/**
 * @brief This enum lists the different execution flags that can be passed to `dtfft_transpose`.
 */
typedef enum {
// Transpose from Fortran X aligned to Fortran Y aligned
  DTFFT_TRANSPOSE_X_TO_Y = CONF_DTFFT_TRANSPOSE_X_TO_Y,
// Transpose from Fortran Y aligned to Fortran X aligned
  DTFFT_TRANSPOSE_Y_TO_X = CONF_DTFFT_TRANSPOSE_Y_TO_X,
// Transpose from Fortran Y aligned to Fortran Z aligned
  DTFFT_TRANSPOSE_Y_TO_Z = CONF_DTFFT_TRANSPOSE_Y_TO_Z,
// Transpose from Fortran Z aligned to Fortran Y aligned
  DTFFT_TRANSPOSE_Z_TO_Y = CONF_DTFFT_TRANSPOSE_Z_TO_Y,
// Transpose from Fortran X aligned to Fortran Z aligned
// (only possible with 3D slab decomposition when slab distributed in Z direction)
  DTFFT_TRANSPOSE_X_TO_Z = CONF_DTFFT_TRANSPOSE_X_TO_Z,
// Transpose from Fortran Z aligned to Fortran X aligned
// (only possible with 3D slab decomposition when slab distributed in Z direction)
  DTFFT_TRANSPOSE_Z_TO_X = CONF_DTFFT_TRANSPOSE_Z_TO_X
} dtfft_transpose_type_t;


/**
 * @brief This enum lists the different precision flags that should be used during plan creation.
 */
typedef enum {
// Use Single precision
  DTFFT_SINGLE = CONF_DTFFT_SINGLE,
// Use Double precision
  DTFFT_DOUBLE = CONF_DTFFT_DOUBLE
} dtfft_precision_t;


/**
 * @brief This enum lists the different effort flags that should be used during plan creation.
 */
typedef enum {
// Create plan as fast as possible
  DTFFT_ESTIMATE = CONF_DTFFT_ESTIMATE,
// Will attempt to find best MPI Grid decomposition
// Passing this flag and MPI Communicator with cartesian topology to `dtfft_create_plan_*` is same as `DTFFT_ESTIMATE`.
  DTFFT_MEASURE = CONF_DTFFT_MEASURE,
// Same as `DTFFT_MEASURE` plus cycle through various send and recieve MPI_Datatypes
// For GPU Build this flag will run autotune procedure to find best backend
  DTFFT_PATIENT = CONF_DTFFT_PATIENT
} dtfft_effort_t;


/**
 * @brief This enum lists available FFT executors. It is populated during library compilation.
 */
typedef enum {
// Create transpose only plan, no executor needed
  DTFFT_EXECUTOR_NONE = CONF_DTFFT_EXECUTOR_NONE,
// Use FFTW3
  DTFFT_EXECUTOR_FFTW3 = CONF_DTFFT_EXECUTOR_FFTW3,
// Use MKL DFTI
  DTFFT_EXECUTOR_MKL = CONF_DTFFT_EXECUTOR_MKL,
// Use GPU Executor CUFFT
  DTFFT_EXECUTOR_CUFFT = CONF_DTFFT_EXECUTOR_CUFFT,
// Use GPU Executor VkFFT
  DTFFT_EXECUTOR_VKFFT = CONF_DTFFT_EXECUTOR_VKFFT
} dtfft_executor_t;


/**
 * @brief This enum lists the different R2R FFT kinds.
 */
typedef enum {
  // DCT-I (Logical N=2*(n-1), inverse is `DTFFT_DCT_1`)
  DTFFT_DCT_1 = CONF_DTFFT_DCT_1,
  // DCT-II (Logical N=2*n, inverse is `DTFFT_DCT_3`)
  DTFFT_DCT_2 = CONF_DTFFT_DCT_2,
  // DCT-III (Logical N=2*n, inverse is `DTFFT_DCT_2`)
  DTFFT_DCT_3 = CONF_DTFFT_DCT_3,
  // DCT-IV (Logical N=2*n, inverse is `DTFFT_DCT_4`)
  DTFFT_DCT_4 = CONF_DTFFT_DCT_4,
  // DST-I (Logical N=2*(n+1), inverse is `DTFFT_DST_1`)
  DTFFT_DST_1 = CONF_DTFFT_DST_1,
  // DST-II (Logical N=2*n, inverse is `DTFFT_DST_3`)
  DTFFT_DST_2 = CONF_DTFFT_DST_2,
  // DST-III (Logical N=2*n, inverse is `DTFFT_DST_2`)
  DTFFT_DST_3 = CONF_DTFFT_DST_3,
  // DST-IV (Logical N=2*n, inverse is `DTFFT_DST_4`)
  DTFFT_DST_4 = CONF_DTFFT_DST_4
} dtfft_r2r_kind_t;


#define DTFFT_CALL(call)                                                      \
do {                                                                          \
    dtfft_error_code_t ierr = call;                                           \
    if( ierr != DTFFT_SUCCESS ) {                                             \
        fprintf(stderr, "dtFFT error in file '%s:%i': %s.\n",                 \
                __FILE__, __LINE__, dtfft_get_error_string( ierr ));          \
        MPI_Abort(MPI_COMM_WORLD, ierr);                                      \
    }                                                                         \
} while (0);


/**
 * @brief Obtains current dtFFT version
 *
 * @return Version
 */
int32_t
dtfft_get_version();


/** @brief Real-to-Real Plan constructor. Must be called after MPI_Init
  *
  * @param[in]      ndims                 Number of dimensions: 2 or 3
  * @param[in]      dims                  Buffer of size `ndims` with global dimensions in reversed order.
  *                                         dims[0] must be fastest varying
  * @param[in]      kinds                 Buffer of size `ndims` with Real FFT kinds in reversed order
  *                                         Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * @param[in]      comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * @param[in]      precision             Precision of transform. One of the
  *                                         - `DTFFT_SINGLE`
  *                                         - `DTFFT_DOUBLE`
  * @param[in]      effort_type           How hard DTFFT should look for best plan. One of the
  *                                         - `DTFFT_ESTIMATE`
  *                                         - `DTFFT_MEASURE`
  *                                         - `DTFFT_PATIENT`
  * @param[in]      executor_type         Type of external FFT executor. One of the
  *                                         - `DTFFT_EXECUTOR_NONE`
  *                                         - `DTFFT_EXECUTOR_FFTW3`
  *                                         - `DTFFT_EXECUTOR_VKFFT` - only GPU
  * @param[out]     plan                  Plan handle ready to be executed
  *
  * @return `DTFFT_SUCCESS` if plan was created, error code otherwise
*/
dtfft_error_code_t
dtfft_create_plan_r2r(
  const int8_t ndims,
  const int32_t *dims,
  const dtfft_r2r_kind_t *kinds,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort_type,
  const dtfft_executor_t executor_type,
  dtfft_plan_t *plan);


/** @brief Complex-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * @param[in]      ndims                 Number of dimensions: 2 or 3
  * @param[in]      dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * @param[in]      comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * @param[in]      precision             Precision of transform. One of the
  *                                         - `DTFFT_SINGLE`
  *                                         - `DTFFT_DOUBLE`
  * @param[in]      effort_type           How hard DTFFT should look for best plan. One of the
  *                                         - `DTFFT_ESTIMATE`
  *                                         - `DTFFT_MEASURE`
  *                                         - `DTFFT_PATIENT`
  * @param[in]      executor_type         Type of external FFT executor. One of the
  *                                         - `DTFFT_EXECUTOR_NONE`
  *                                         - `DTFFT_EXECUTOR_FFTW3`
  *                                         - `DTFFT_EXECUTOR_MKL`
  *                                         - `DTFFT_EXECUTOR_CUFFT` - only GPU
  *                                         - `DTFFT_EXECUTOR_VKFFT` - only GPU
  * @param[out]     plan                  Plan handle ready to be executed
  *
  * @return `DTFFT_SUCCESS` if plan was created, error code otherwise
*/
dtfft_error_code_t
dtfft_create_plan_c2c(
  const int8_t ndims,
  const int32_t *dims,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort_type,
  const dtfft_executor_t executor_type,
  dtfft_plan_t *plan);


#ifndef DTFFT_TRANSPOSE_ONLY
/** @brief Real-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * @param[in]      ndims                 Number of dimensions: 2 or 3
  * @param[in]      dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * @param[in]      comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * @param[in]      precision             Precision of transform. One of the
  *                                         - `DTFFT_SINGLE`
  *                                         - `DTFFT_DOUBLE`
  * @param[in]      effort_type           How hard DTFFT should look for best plan. One of the
  *                                         - `DTFFT_ESTIMATE`
  *                                         - `DTFFT_MEASURE`
  *                                         - `DTFFT_PATIENT`
  * @param[in]      executor_type         Type of external FFT executor. One of the
  *                                         - `DTFFT_EXECUTOR_FFTW3`
  *                                         - `DTFFT_EXECUTOR_MKL`
  *                                         - `DTFFT_EXECUTOR_CUFFT` - only GPU
  *                                         - `DTFFT_EXECUTOR_VKFFT` - only GPU
  * @param[out]     plan                  Plan handle ready to be executed
  *
  * @return `DTFFT_SUCCESS` if plan was created, error code otherwise
  *
  * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C plan instead
*/
dtfft_error_code_t
dtfft_create_plan_r2c(
  const int8_t ndims,
  const int32_t *dims,
  MPI_Comm comm,
  const dtfft_precision_t precision,
  const dtfft_effort_t effort_type,
  const dtfft_executor_t executor_type,
  dtfft_plan_t *plan);
#endif

/** @brief Checks if plan is using Z-slab optimization.
  * If `true` then flags `DTFFT_TRANSPOSE_X_TO_Z` and `DTFFT_TRANSPOSE_Z_TO_X` will be valid to pass to `dtfft_transpose`.
  *
  * @param[in]      plan              Plan handle
  * @param[out]     is_z_slab_enabled Boolean value if Z-slab is used.
  *
  * @return `DTFFT_SUCCESS` if call was without error, error code otherwise
*/
dtfft_error_code_t
dtfft_get_z_slab_enabled(dtfft_plan_t plan, bool *is_z_slab_enabled);


/** @brief Plan execution. Neither `in` nor `out` are allowed to be `NULL`. It is safe to pass same pointer to both `in` and `out`.
  *
  * @param[in]      plan            Plan handle
  * @param[inout]   in              Incoming buffer
  * @param[out]     out             Result buffer
  * @param[in]      execute_type    Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  * @param[inout]   aux             Optional auxiliary buffer. Can be `NULL`.
  *                                 If `NULL` during first call to this function, then auxiliary will be allocated
  *                                 internally and freed after call to `dtfft_destroy`
  *
  * @return `DTFFT_SUCCESS` if plan was executed, error code otherwise
*/
dtfft_error_code_t
dtfft_execute(dtfft_plan_t plan, void *in, void *out, const dtfft_execute_type_t execute_type, void *aux);


/** @brief Transpose data in single dimension, e.g. X align -> Y align
  * \attention `in` and `out` cannot be the same pointers
  *
  * @param[in]      plan            Plan handle
  * @param[in]      in              Incoming buffer
  * @param[out]     out             Transposed buffer
  * @param[in]      transpose_type  Type of transpose:
  *                                   - `DTFFT_TRANSPOSE_X_TO_Y`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_X`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_Y` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_X_TO_Z` (3d plan and Z-slab only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_X` (3d plan and Z-slab only)
  *
  * @return `DTFFT_SUCCESS` if plan was executed, error code otherwise
*/
dtfft_error_code_t
dtfft_transpose(dtfft_plan_t plan, void *in, void *out, const dtfft_transpose_type_t transpose_type);


/** @brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
 *
 * @param[inout]    plan            Plan handle
*/
dtfft_error_code_t
dtfft_destroy(dtfft_plan_t *plan);


/** @brief Get grid decomposition information. Results may differ on different MPI processes
  *
  * Minimum number of bytes that needs allocation:
  *
  * @param[in]      plan            Plan handle
  * @param[out]     in_starts       Starts of local portion of data in 'real' space in reversed order
  * @param[out]     in_counts       Sizes  of local portion of data in 'real' space in reversed order
  * @param[out]     out_starts      Starts of local portion of data in 'fourier' space in reversed order
  * @param[out]     out_counts      Sizes  of local portion of data in 'fourier' space in reversed order
  * @param[out]     alloc_size      Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers.
  *                                 Number of bytes to be allocated:
  *                                   - C2C plan: 2 * `alloc_size` * sizeof(double/float) or `alloc_size` * sizeof(dtfft_complex/dtfftf_complex)
  *                                   - R2R plan: `alloc_size` * sizeof(double/float)
  *                                   - R2C plan: `alloc_size` * sizeof(double/float)
  * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
*/
dtfft_error_code_t
dtfft_get_local_sizes(dtfft_plan_t plan, int32_t *in_starts, int32_t *in_counts, int32_t *out_starts, int32_t *out_counts, size_t *alloc_size);


/** @brief Wrapper around `dtfft_get_local_sizes` to obtain number of elements only
  *
  * @param[in]      plan            Plan handle
  * @param[out]     alloc_size      Minimum number of elements needs to be allocated for `in`, `out` or `aux` buffers.
  *                                 Number of bytes to be allocated:
  *                                   - C2C plan: 2 * `alloc_size` * sizeof(double/float) or `alloc_size` * sizeof(dtfft_complex/dtfftf_complex)
  *                                   - R2R plan: `alloc_size` * sizeof(double/float)
  *                                   - R2C plan: `alloc_size` * sizeof(double/float)
  * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
*/
dtfft_error_code_t
dtfft_get_alloc_size(dtfft_plan_t plan, size_t *alloc_size);


/**
 * @brief Gets the string description of an error code
 *
 * @param[in]       error_code      Error code to convert to string
 * @return Error string explaining error.
 */
const char *
dtfft_get_error_string(const dtfft_error_code_t error_code);


// Structure to hold pencil decomposition info
typedef struct {
  int8_t dim;
  int32_t starts[3];
  int32_t counts[3];
} dtfft_pencil_t;

/**
 * @brief Obtains pencil information from plan. This can be useful when user wants to use own FFT implementation, 
 * that is unavailable in dtFFT.
 * 
 * @param[in]     plan            Plan handle
 * @param[in]     dim             Required dimension:
 *                                  - 1 for XYZ layout
 *                                  - 2 for YXZ layout
 *                                  - 3 for ZXY layout
 * @param[out]    pencil          Pencil data
 * 
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 */
dtfft_error_code_t
dtfft_get_pencil(dtfft_plan_t plan, int8_t dim, dtfft_pencil_t *pencil);


/**
 * @brief Prints plan-related information to stdout
 *
 * @param[in]     plan            Plan handle
 *
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 **/
dtfft_error_code_t
dtfft_report(dtfft_plan_t plan);


#ifdef DTFFT_WITH_CUDA

/**
 * @brief This enum lists the different available backend options.
 */
typedef enum {
  // Backend that uses MPI datatypes
  // Not really recommended to use, since it is a million times slower than other backends
  // Left here just to show how slow MPI Datatypes are for GPU usage
  DTFFT_GPU_BACKEND_MPI_DATATYPE = CONF_DTFFT_GPU_BACKEND_MPI_DATATYPE,
  // MPI peer-to-peer algorithm
  DTFFT_GPU_BACKEND_MPI_P2P = CONF_DTFFT_GPU_BACKEND_MPI_P2P,
  // MPI peer-to-peer algorithm with overlapping data copying and unpacking
  DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED = CONF_DTFFT_GPU_BACKEND_MPI_P2P_PIPELINED,
  // MPI backend using MPI_Alltoallv
  DTFFT_GPU_BACKEND_MPI_A2A = CONF_DTFFT_GPU_BACKEND_MPI_A2A,
  // NCCL backend
  DTFFT_GPU_BACKEND_NCCL = CONF_DTFFT_GPU_BACKEND_NCCL,
  // NCCL backend with overlapping data copying and unpacking
  DTFFT_GPU_BACKEND_NCCL_PIPELINED = CONF_DTFFT_GPU_BACKEND_NCCL_PIPELINED,
} dtfft_gpu_backend_t;


/**
 * @brief Returns stream assosiated with dtfft plan.
 * This can either be steam passed by user to `dtfft_set_config` or stream created internally.
 *
 * @param[in]      plan           Plan handle
 * @param[out]     stream         CUDA stream associated with plan
 *
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 */
dtfft_error_code_t
dtfft_get_stream(dtfft_plan_t plan, cudaStream_t *stream);


/**
 * @brief Returns selected GPU backend during autotune if `effort_type` is `DTFFT_PATIENT`.
 *
 * If `effort_type` passed to any create function is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`
 * returns value set by `dtfft_set_gpu_backend` or default value, which is `DTFFT_GPU_BACKEND_NCCL`.
 *
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 */
dtfft_error_code_t
dtfft_get_gpu_backend(dtfft_plan_t plan, dtfft_gpu_backend_t *backend_id);

/**
 * @brief Returns null terminated string with name of backend provided as argument.
 *
 * @param[in]       backend_id   dtfft_gpu_backend_t
 *
 * @return Character representation of backend.
 */
const char *
dtfft_get_gpu_backend_string(const dtfft_gpu_backend_t backend_id);
#endif

typedef struct {
  /*
  Should dtFFT use Z-slab optimization or not.
  Default is true.
  One should consider disabling Z-slab optimization in order to resolve `DTFFT_ERROR_VKFFT_R2R_2D_PLAN` error 
  OR when underlying FFT implementation of 2D plan is too slow.
  In all other cases it is considered that Z-slab is always faster, since it reduces number of data transpositions.
  */
  bool enable_z_slab;

#ifdef DTFFT_WITH_CUDA
  /*
  Main CUDA stream that will be used in dtFFT.
  This parameter is a placeholder for user to set custom stream.
  Stream that is actually used by dtFFT plan is returned by `dtfft_get_stream` function.
  When user sets stream he is responsible of destroying it.
  Stream must not be destroyed before call to `dtfft_destroy`.
  */
  cudaStream_t stream;
  /*
  Backend that will be used by dtFFT when `effort_type` is `DTFFT_ESTIMATE` or `DTFFT_MEASURE`.
  Default is `DTFFT_GPU_BACKEND_NCCL`
  */
  dtfft_gpu_backend_t gpu_backend;
  /*
  Should MPI GPU Backends be enabled when `effort_type` is `DTFFT_PATIENT` or not.
  Default is false.

  MPI Backends are disabled by default during autotuning process due to OpenMPI Bug https://github.com/open-mpi/ompi/issues/12849
  It was noticed that during plan autotuning GPU memory not being freed completely.
  For example:
  1024x1024x512 C2C, double precision, single GPU, using Z-slab optimization, with MPI backends enabled, plan autotuning will leak 8Gb GPU memory.
  Without Z-slab optimization, running on 4 GPUs, will leak 24Gb on each of the GPUs.

  One of the workarounds is to disable MPI Backends by default, which is done here.

  Other is to pass "--mca btl_smcuda_use_cuda_ipc 0" to `mpiexec`,
  but it was noticed that disabling CUDA IPC seriously affects overall performance of MPI algorithms
  */
  bool enable_mpi_backends;
  /*
  Should pipelined GPU backends be enabled when `effort_type` is `DTFFT_PATIENT` or not.
  Default is true.

  Pipelined backends require additional buffer that user has no control over.
  */
  bool enable_pipelined_backends;
  /*
  Should NCCL Backends be enabled when `effort_type` is `DTFFT_PATIENT` or not.
  Default is true.
  */
  bool enable_nccl_backends;
  /*
  Should NCCL Backends be enabled when `effort_type` is `DTFFT_PATIENT` or not.
  Default is true.

  Unused. Reserved for future.
  */
  bool enable_nvshmem_backends;
#endif
} dtfft_config_t;

/**
 * @brief Sets default values to config
 * 
 *
 * @param[out]  config  Config to set default values into
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 */
dtfft_error_code_t
dtfft_create_config(dtfft_config_t *config);

/**
 * @brief Set configuration values to dtFFT. In order to take effect should be called before plan creation
 * 
 * @param[in]   config  Config to set
 * @return `DTFFT_SUCCESS` if call was successfull, error code otherwise
 */
dtfft_error_code_t
dtfft_set_config(dtfft_config_t config);


#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT
