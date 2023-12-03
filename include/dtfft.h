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

#ifndef DTFFT_H
#define DTFFT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <mpi.h>

// Aligned Complex datatype, double precision
typedef double dtfft_complex[2];
// Aligned Complex datatype, single precision
typedef float dtfftf_complex[2];

// Structure to hold plan data
typedef struct dtfft_plan_t *dtfft_plan;


// dtFFT transpose_type flags
extern int C_DTFFT_TRANSPOSE_OUT;
// Perform XYZ --> YXZ --> ZXY transposition
#define DTFFT_TRANSPOSE_OUT C_DTFFT_TRANSPOSE_OUT
extern int C_DTFFT_TRANSPOSE_IN;
// Perform ZXY --> YXZ --> XYZ transposition
#define DTFFT_TRANSPOSE_IN C_DTFFT_TRANSPOSE_IN

// Flags for transpose only plans
extern int C_DTFFT_TRANSPOSE_X_TO_Y;
// Transpose from Fortran X aligned to Fortran Y aligned
#define DTFFT_TRANSPOSE_X_TO_Y C_DTFFT_TRANSPOSE_X_TO_Y
extern int C_DTFFT_TRANSPOSE_Y_TO_X;
// Transpose from Fortran Y aligned to Fortran X aligned
#define DTFFT_TRANSPOSE_Y_TO_X C_DTFFT_TRANSPOSE_Y_TO_X
extern int C_DTFFT_TRANSPOSE_Y_TO_Z;
// Transpose from Fortran Y aligned to Fortran Z aligned
#define DTFFT_TRANSPOSE_Y_TO_Z C_DTFFT_TRANSPOSE_Y_TO_Z
extern int C_DTFFT_TRANSPOSE_Z_TO_Y;
// Transpose from Fortran Z aligned to Fortran Y aligned
#define DTFFT_TRANSPOSE_Z_TO_Y C_DTFFT_TRANSPOSE_Z_TO_Y

// dtFFT Precision flags
extern int C_DTFFT_SINGLE;
// Use Single precision
#define DTFFT_SINGLE C_DTFFT_SINGLE
extern int C_DTFFT_DOUBLE;
// Use Double precision
#define DTFFT_DOUBLE C_DTFFT_DOUBLE

// dtFFT Effort flags
extern int C_DTFFT_ESTIMATE;
// Create plan as fast as possible
#define DTFFT_ESTIMATE C_DTFFT_ESTIMATE
extern int C_DTFFT_MEASURE;
// Find Grid decompostion with fastest MPI calls
#define DTFFT_MEASURE C_DTFFT_MEASURE
extern int C_DTFFT_PATIENT;
// Same as above plus cycle through various send and recieve MPI_Datatypes
#define DTFFT_PATIENT C_DTFFT_PATIENT

// dtFFT types of external executors
extern int C_DTFFT_EXECUTOR_NONE;
// Create transpose only plan, no executor needed
#define DTFFT_EXECUTOR_NONE C_DTFFT_EXECUTOR_NONE
extern int C_DTFFT_EXECUTOR_FFTW3;
// Use FFTW3
#define DTFFT_EXECUTOR_FFTW3 C_DTFFT_EXECUTOR_FFTW3
extern int C_DTFFT_EXECUTOR_MKL;
// Use MKL DFTI
#define DTFFT_EXECUTOR_MKL C_DTFFT_EXECUTOR_MKL
// extern int C_DTFFT_EXECUTOR_CUFFT;
// Use cufft
// #define DTFFT_EXECUTOR_CUFFT C_DTFFT_EXECUTOR_CUFFT
// extern int C_DTFFT_EXECUTOR_KFR;
// Use KFR
// #define DTFFT_EXECUTOR_KFR C_DTFFT_EXECUTOR_KFR

// R2R Transform kinds
extern int C_DTFFT_DCT_1;
#define DTFFT_DCT_1 C_DTFFT_DCT_1
extern int C_DTFFT_DCT_2;
#define DTFFT_DCT_2 C_DTFFT_DCT_2
extern int C_DTFFT_DCT_3;
#define DTFFT_DCT_3 C_DTFFT_DCT_3
extern int C_DTFFT_DCT_4;
#define DTFFT_DCT_4 C_DTFFT_DCT_4
extern int C_DTFFT_DST_1;
#define DTFFT_DST_1 C_DTFFT_DST_1
extern int C_DTFFT_DST_2;
#define DTFFT_DST_2 C_DTFFT_DST_2
extern int C_DTFFT_DST_3;
#define DTFFT_DST_3 C_DTFFT_DST_3
extern int C_DTFFT_DST_4;
#define DTFFT_DST_4 C_DTFFT_DST_4


/** \brief Real-to-Real Plan constructor. Must be called after MPI_Init
  * 
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    in_kinds              Buffer of size `ndims` with Real FFT kinds in reversed order, forward transform.
  *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    out_kinds             Buffer of size `ndims` with Real FFT kinds in reversed order, backward transform
  *                                     Can be NULL if `executor_type` == `DTFFT_EXECUTOR_NONE`
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_KFR`
  *
  * \return Plan handle ready to be executed
  * 
  * \note Parameter `effort_flag` is not yet used and reserved for future.
*/
extern
dtfft_plan
dtfft_create_plan_r2r(const int ndims, const int *dims, const int *in_kinds, const int *out_kinds, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type);

/** \brief Complex-to-Complex Plan constructor. Must be called after MPI_Init
  *
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_NONE`
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
  *                                     - `DTFFT_EXECUTOR_KFR`
  *
  * \return Plan handle ready to be executed
  * 
  * \note Parameter `effort_flag` is not yet used and reserved for future.
*/
extern
dtfft_plan
dtfft_create_plan_c2c(const int ndims, const int *dims, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type);

/** \brief Real-to-Complex Plan constructor. Must be called after MPI_Init
  * 
  * \param[in]    ndims                 Number of dimensions: 2 or 3
  * \param[in]    dims                  Buffer of size `ndims` with global dimensions in reversed order.
  * \param[in]    comm                  MPI communicator: `MPI_COMM_WORLD` or Cartesian communicator
  * \param[in]    precision             Precision of transform: `DTFFT_SINGLE` or `DTFFT_DOUBLE`
  * \param[in]    effort_flag           How hard DTFFT should look for best plan: `DTFFT_ESTIMATE`, `DTFFT_MEASURE` or `DTFFT_PATIENT`
  * \param[in]    executor_type         Type of external FFT executor. One of the
  *                                     - `DTFFT_EXECUTOR_FFTW3`
  *                                     - `DTFFT_EXECUTOR_MKL`
  *                                     - `DTFFT_EXECUTOR_KFR`
  *
  * \return Plan handle ready to be executed
  * 
  * \note Parameter `effort_flag` is not yet used and reserved for future.
  * \note Parameter `executor_type` cannot be `DTFFT_EXECUTOR_NONE`. Use C2C plan instead
*/
extern
dtfft_plan
dtfft_create_plan_r2c(const int ndims, const int *dims, MPI_Comm comm, const int precision, const int effort_flag, const int executor_type);

/** \brief Plan execution. Neither `in` nor `out` are allowed to be `NULL`.
  * \attention Inplace execution is possible only in these plans:
  *     - R2R 3d
  *     - C2C 3d
  *     - R2C 2d
  * 
  * \param[in]      plan            Plan handle
  * \param[inout]   in              Incoming buffer
  * \param[out]     out             Result buffer
  * \param[in]      transpose_type  Type of transform: `DTFFT_TRANSPOSE_OUT` or `DTFFT_TRANSPOSE_IN`
  * \param[inout]   aux             Optional auxiliary buffer. Can be `NULL`
*/
extern
void
dtfft_execute(dtfft_plan plan, void *in, void *out, const int transpose_type, void *aux);

/** \brief Transpose data in single dimension, e.g. X align -> Y align 
  * \attention `in` and `out` cannot be the same pointers
  *
  * \param[in]      plan            Plan handle
  * \param[in]      in              Incoming vector
  * \param[out]     out             Transposed vector
  * \param[in]      transpose_type  Type of transpose: `DTFFT_TRANSPOSE_X_TO_Y`, `DTFFT_TRANSPOSE_Y_TO_X`
  *                                 `DTFFT_TRANSPOSE_Y_TO_Z` or `DTFFT_TRANSPOSE_Z_TO_Y`
*/
extern
void
dtfft_transpose(dtfft_plan plan, const void *in, void *out, const int transpose_type);

/** \brief Plan Destructor. To fully clean all internal memory, this should be called before MPI_Finalize
 * 
 * \param[inout]    plan            Plan handle
*/
extern
void
dtfft_destroy(dtfft_plan plan);

/** \brief Get grid decomposition information. Results may differ on different MPI processes
  *
  * Minimum number of bytes that needs allocation:
  * - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(dtfft_complex/dtfftf_complex)
  * - R2R plan: alloc_size * sizeof(double/float)
  * - R2C plan: alloc_size * sizeof(double/float)
  * \return Minimum number of elements needs to be allocated
  * 
  * \param[inout]   plan            Plan handle
  * \param[out]     in_starts       Starts of local portion of data in 'real' space in reversed order
  * \param[out]     in_counts       Sizes  of local portion of data in 'real' space in reversed order
  * \param[out]     out_starts      Starts of local portion of data in 'fourier' space in reversed order
  * \param[out]     out_counts      Sizes  of local portion of data in 'fourier' space in reversed order
  *
*/
extern
size_t
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts);

/** \brief Wrapper around `dtfft_get_local_sizes`
  * \return Minimum number of elements needs to be allocated
  *
  * Minimum number of bytes that needs allocation:
  * - C2C plan: 2 * alloc_size * sizeof(double/float) or alloc_size * sizeof(dtfft_complex/dtfftf_complex)
  * - R2R plan: alloc_size * sizeof(double/float)
  * - R2C plan: alloc_size * sizeof(double/float)
*/
extern
size_t
dtfft_get_alloc_size(dtfft_plan plan);

/** \brief Get minimal size needed for optional aux buffer. Results may differ on different MPI processes
  *
  * Minimum number of bytes that needs allocation:
  * - C2C plan: 2 * aux_size * sizeof(double/float) or aux_size * sizeof(dtfft_complex/dtfftf_complex)
  * - R2R plan: aux_size * sizeof(double/float)
  * - R2C plan: 2 * aux_size * sizeof(double/float) or aux_size * sizeof(dtfft_complex/dtfftf_complex)
  * \return Minimum number of elements needs to be allocated
  *
*/
extern
size_t
dtfft_get_aux_size(dtfft_plan plan);


#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT
