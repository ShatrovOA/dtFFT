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

#ifdef __cplusplus
extern "C" {
#endif

#include "dtfft_config.h"
#include <stddef.h> // size_t
#include <mpi.h>

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
typedef struct dtfft_plan_t *dtfft_plan;

// Error codes
#define DTFFT_SUCCESS CONF_DTFFT_SUCCESS
#define DTFFT_ERROR_MPI_FINALIZED CONF_DTFFT_ERROR_MPI_FINALIZED
#define DTFFT_ERROR_PLAN_NOT_CREATED CONF_DTFFT_ERROR_PLAN_NOT_CREATED
#define DTFFT_ERROR_INVALID_TRANSPOSE_TYPE CONF_DTFFT_ERROR_INVALID_TRANSPOSE_TYPE
#define DTFFT_ERROR_INVALID_N_DIMENSIONS CONF_DTFFT_ERROR_INVALID_N_DIMENSIONS
#define DTFFT_ERROR_INVALID_DIMENSION_SIZE CONF_DTFFT_ERROR_INVALID_DIMENSION_SIZE
#define DTFFT_ERROR_INVALID_COMM_TYPE CONF_DTFFT_ERROR_INVALID_COMM_TYPE
#define DTFFT_ERROR_INVALID_PRECISION CONF_DTFFT_ERROR_INVALID_PRECISION
#define DTFFT_ERROR_INVALID_EFFORT_FLAG CONF_DTFFT_ERROR_INVALID_EFFORT_FLAG
#define DTFFT_ERROR_INVALID_EXECUTOR_TYPE CONF_DTFFT_ERROR_INVALID_EXECUTOR_TYPE
#define DTFFT_ERROR_INVALID_COMM_DIMS CONF_DTFFT_ERROR_INVALID_COMM_DIMS
#define DTFFT_ERROR_INVALID_COMM_FAST_DIM CONF_DTFFT_ERROR_INVALID_COMM_FAST_DIM
#define DTFFT_ERROR_MISSING_R2R_KINDS CONF_DTFFT_ERROR_MISSING_R2R_KINDS
#define DTFFT_ERROR_INVALID_R2R_KINDS CONF_DTFFT_ERROR_INVALID_R2R_KINDS
#define DTFFT_ERROR_R2C_TRANSPOSE_PLAN CONF_DTFFT_ERROR_R2C_TRANSPOSE_PLAN
#define DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED CONF_DTFFT_ERROR_R2R_FFT_NOT_SUPPORTED
#define DTFFT_ERROR_CUFFTMP_2D_PLAN CONF_DTFFT_ERROR_CUFFTMP_2D_PLAN


#define DTFFT_CALL(call)                                                      \
do {                                                                          \
    int ierr = call;                                                          \
    if( ierr != DTFFT_SUCCESS ) {                                             \
        fprintf(stderr, "dtFFT error in file '%s:%i': %s.\n",                 \
                __FILE__, __LINE__, dtfft_get_error_string( ierr ) );         \
        MPI_Abort(MPI_COMM_WORLD, ierr);                                      \
    }                                                                         \
} while (0);


// dtFFT transpose_type flags

// Perform XYZ --> YXZ --> ZXY transposition
#define DTFFT_TRANSPOSE_OUT CONF_DTFFT_TRANSPOSE_OUT
// Perform ZXY --> YXZ --> XYZ transposition
#define DTFFT_TRANSPOSE_IN CONF_DTFFT_TRANSPOSE_IN

// Flags for transpose only plans

// Transpose from Fortran X aligned to Fortran Y aligned
#define DTFFT_TRANSPOSE_X_TO_Y CONF_DTFFT_TRANSPOSE_X_TO_Y
// Transpose from Fortran Y aligned to Fortran X aligned
#define DTFFT_TRANSPOSE_Y_TO_X CONF_DTFFT_TRANSPOSE_Y_TO_X
// Transpose from Fortran Y aligned to Fortran Z aligned
#define DTFFT_TRANSPOSE_Y_TO_Z CONF_DTFFT_TRANSPOSE_Y_TO_Z
// Transpose from Fortran Z aligned to Fortran Y aligned
#define DTFFT_TRANSPOSE_Z_TO_Y CONF_DTFFT_TRANSPOSE_Z_TO_Y
// Transpose from Fortran X aligned to Fortran Z aligned
// (only possible with 3D slab decomposition when slab distributed in Z direction)
#define DTFFT_TRANSPOSE_X_TO_Z CONF_DTFFT_TRANSPOSE_X_TO_Z
// Transpose from Fortran Z aligned to Fortran X aligned
// (only possible with 3D slab decomposition when slab distributed in Z direction)
#define DTFFT_TRANSPOSE_Z_TO_X CONF_DTFFT_TRANSPOSE_Z_TO_X

/*
  dtFFT Precision flags
*/
// Use Single precision
#define DTFFT_SINGLE CONF_DTFFT_SINGLE
// Use Double precision
#define DTFFT_DOUBLE CONF_DTFFT_DOUBLE

/*
  dtFFT Effort flags
*/
// Create plan as fast as possible
#define DTFFT_ESTIMATE CONF_DTFFT_ESTIMATE
// Will attempt to find best MPI Grid decompostion
// Passing this flag and MPI Communicator with cartesian topology to `dtfft_create_plan_*` makes dtFFT do nothing.
#define DTFFT_MEASURE CONF_DTFFT_MEASURE
// Same as `DTFFT_MEASURE` plus cycle through various send and recieve MPI_Datatypes
#define DTFFT_PATIENT CONF_DTFFT_PATIENT

/*
  dtFFT types of external executors
*/
// Create transpose only plan, no executor needed
#define DTFFT_EXECUTOR_NONE CONF_DTFFT_EXECUTOR_NONE
#ifndef DTFFT_WITHOUT_FFTW
// Use FFTW3
#define DTFFT_EXECUTOR_FFTW3 CONF_DTFFT_EXECUTOR_FFTW3
#endif
#ifdef DTFFT_WITH_MKL
// Use MKL DFTI
#define DTFFT_EXECUTOR_MKL CONF_DTFFT_EXECUTOR_MKL
#endif
#ifdef DTFFT_WITH_CUFFT
// Use GPU Executor cuFFT
#define DTFFT_EXECUTOR_CUFFT CONF_DTFFT_EXECUTOR_CUFFT
#endif
// #ifdef DTFFT_WITH_KFR
// // Use KFR
// #define DTFFT_EXECUTOR_KFR CONF_DTFFT_EXECUTOR_KFR
// #endif
#ifdef DTFFT_WITH_VKFFT
// Use GPU Executor VkFFT
#define DTFFT_EXECUTOR_VKFFT CONF_DTFFT_EXECUTOR_VKFFT
#endif

// R2R Transform kinds

#define DTFFT_DCT_1 CONF_DTFFT_DCT_1
#define DTFFT_DCT_2 CONF_DTFFT_DCT_2
#define DTFFT_DCT_3 CONF_DTFFT_DCT_3
#define DTFFT_DCT_4 CONF_DTFFT_DCT_4
#define DTFFT_DST_1 CONF_DTFFT_DST_1
#define DTFFT_DST_2 CONF_DTFFT_DST_2
#define DTFFT_DST_3 CONF_DTFFT_DST_3
#define DTFFT_DST_4 CONF_DTFFT_DST_4

/** \brief Real-to-Real Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  *                                     dims[0] must be fastest varying
  * \param[in]    kinds                 Buffer of size `ndims` with Real FFT kinds in reversed order
  *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform. One of the
  *                                       - `DTFFT_SINGLE`
  *                                       - `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan. One of the
  *                                       - `DTFFT_ESTIMATE`
  *                                       - `DTFFT_MEASURE`
  *                                       - `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                       - `DTFFT_EXECUTOR_NONE`
  *                                       - `DTFFT_EXECUTOR_FFTW3`
  *                                       - `DTFFT_EXECUTOR_KFR`
  *                                       - `DTFFT_EXECUTOR_VKFFT`
  * \param[out]   plan                  Plan handle ready to be executed
  *
  * \return `DTFFT_SUCCESS` if plan was created, error code otherwise
  *
  * \note Executor `DTFFT_EXECUTOR_KFR` only supports DCT types 2 and 3
*/
extern
int
dtfft_create_plan_r2r(const int ndims, const int *dims, const int *kinds, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type, dtfft_plan *plan);



/** \brief Complex-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform. One of the
  *                                       - `DTFFT_SINGLE`
  *                                       - `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan. One of the
  *                                       - `DTFFT_ESTIMATE`
  *                                       - `DTFFT_MEASURE`
  *                                       - `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                       - `DTFFT_EXECUTOR_NONE`
  *                                       - `DTFFT_EXECUTOR_FFTW3`
  *                                       - `DTFFT_EXECUTOR_MKL`
  *                                       - `DTFFT_EXECUTOR_KFR`
  *                                       - `DTFFT_EXECUTOR_VKFFT`
  * \param[out]   plan                  Plan handle ready to be executed
  *
  * \return `DTFFT_SUCCESS` if plan was created, error code otherwise
*/
extern
int
dtfft_create_plan_c2c(const int ndims, const int *dims, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type, dtfft_plan *plan);



/** \brief Real-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform. One of the
  *                                       - `DTFFT_SINGLE`
  *                                       - `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan. One of the
  *                                       - `DTFFT_ESTIMATE`
  *                                       - `DTFFT_MEASURE`
  *                                       - `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                       - `DTFFT_EXECUTOR_FFTW3`
  *                                       - `DTFFT_EXECUTOR_MKL`
  *                                       - `DTFFT_EXECUTOR_KFR`
  *                                       - `DTFFT_EXECUTOR_VKFFT`
  * \param[out]   plan                  Plan handle ready to be executed
  *
  * \return `DTFFT_SUCCESS` if plan was created, error code otherwise
  *
  * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C plan instead
*/
extern
int
dtfft_create_plan_r2c(const int ndims, const int *dims, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type, dtfft_plan *plan);



/** \brief Plan execution. Neither `in` nor `out` are allowed to be `NULL`. It is safe to pass same pointer to both `in` and `out`.
  *
  * \param[in]      plan            Plan handle
  * \param[inout]   in              Incoming buffer
  * \param[out]     out             Result buffer
  * \param[in]      transpose_type  Type of transform:
  *                                   - `DTFFT_TRANSPOSE_OUT`
  *                                   - `DTFFT_TRANSPOSE_IN`
  * \param[inout]   aux             Optional auxiliary buffer. Can be `NULL`.
  *                                 If `NULL` during first call to this function, then it will be allocated internally and freed after call to `dtfft_destroy`
  *
  * \return `DTFFT_SUCCESS` if plan was executed, error code otherwise
*/
extern
int
dtfft_execute(dtfft_plan plan, void *in, void *out, const int transpose_type, void *aux);



/** \brief Transpose data in single dimension, e.g. X align -> Y align
  * \attention `in` and `out` cannot be the same pointers
  *
  * \param[in]      plan            Plan handle
  * \param[in]      in              Incoming buffer
  * \param[out]     out             Transposed buffer
  * \param[in]      transpose_type  Type of transpose:
  *                                   - `DTFFT_TRANSPOSE_X_TO_Y`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_X`
  *                                   - `DTFFT_TRANSPOSE_Y_TO_Z` (3d plan only)
  *                                   - `DTFFT_TRANSPOSE_Z_TO_Y` (3d plan only)
  *
  * \return `DTFFT_SUCCESS` if plan was executed, error code otherwise
*/
extern
int
dtfft_transpose(dtfft_plan plan, const void *in, void *out, const int transpose_type);



/** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
 *
 * \param[inout]    plan            Plan handle
*/
extern
void
dtfft_destroy(dtfft_plan *plan);



/** \brief Get grid decomposition information. Results may differ on different MPI processes
  *
  * Minimum number of bytes that needs allocation:
  * \return Execution error code
  *
  * \param[in]      plan            Plan handle
  * \param[out]     in_starts       Starts of local portion of data in 'real' space in reversed order
  * \param[out]     in_counts       Sizes  of local portion of data in 'real' space in reversed order
  * \param[out]     out_starts      Starts of local portion of data in 'fourier' space in reversed order
  * \param[out]     out_counts      Sizes  of local portion of data in 'fourier' space in reversed order
  * \param[out]     alloc_size      Minimum number of elements needs to be allocated
  *                                   - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(dtfft_complex/dtfftf_complex)
  *                                   - R2R plan: alloc_size * sizeof(double/float)
  *                                   - R2C plan: alloc_size * sizeof(double/float)
  *
*/
extern
int
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, size_t *alloc_size);



/** \brief Wrapper around `dtfft_get_local_sizes` to obtain number of elements only
  * \return Execution error code
  *
  * \param[in]      plan            Plan handle
  * \param[out]     alloc_size      Minimum number of elements needs to be allocated
  *                                   - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(dtfft_complex/dtfftf_complex)
  *                                   - R2R plan: alloc_size * sizeof(double/float)
  *                                   - R2C plan: alloc_size * sizeof(double/float)
*/
extern
int
dtfft_get_alloc_size(dtfft_plan plan, size_t *alloc_size);


extern
const char *
dtfft_get_error_string(const int error_code);

extern
void dtfft_profile_report();


#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT
