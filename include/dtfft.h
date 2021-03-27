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

#include <mpi.h>
#include <stddef.h>

// Complex datatypes
typedef double dtfft_complex[2];
typedef float dtfftf_complex[2];

// DTFFT transpose_type flags
#define DTFFT_TRANSPOSE_OUT 101
#define DTFFT_TRANSPOSE_IN 102

// DTFFT types of external executers
#define DTFFT_EXECUTOR_FFTW3 200
#define DTFFT_EXECUTOR_MKL 201
#define DTFFT_EXECUTOR_CUFFT 202

// DTFFT Effort flags
// Reserved for future. Currently it doesn't matter which one to pass
#define DTFFT_ESTIMATE 0
#define DTFFT_MEASURE 1
#define DTFFT_PATIENT 2

typedef struct
{
  int plan_type;  // Plan type for error check
  void *plan;     // Pointer to fortran handle
} dtfft_plan;

extern 
dtfft_plan 
dtfft_create_plan_r2r_3d(MPI_Comm comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effor_flag, int executor_type);

extern
dtfft_plan 
dtfft_create_plan_r2r_2d(MPI_Comm comm, int ny, int nx, int *in_kinds, int *out_kinds, int effor_flag, int executor_type);

extern 
dtfft_plan 
dtfft_create_plan_f_r2r_3d(MPI_Comm comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effor_flag, int executor_type);

extern
dtfft_plan 
dtfft_create_plan_f_r2r_2d(MPI_Comm comm, int ny, int nx, int *in_kinds, int *out_kinds, int effor_flag, int executor_type);

extern 
dtfft_plan 
dtfft_create_plan_c2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effor_flag, int executor_type);

extern
dtfft_plan 
dtfft_create_plan_c2c_2d(MPI_Comm comm, int ny, int nx, int effor_flag, int executor_type);

extern 
dtfft_plan 
dtfft_create_plan_f_c2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effor_flag, int executor_type);

extern
dtfft_plan 
dtfft_create_plan_f_c2c_2d(MPI_Comm comm, int ny, int nx, int effor_flag, int executor_type);

extern
dtfft_plan
dtfft_create_plan_r2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type);

extern
dtfft_plan
dtfft_create_plan_r2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type);

extern
dtfft_plan
dtfft_create_plan_f_r2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type);

extern
dtfft_plan
dtfft_create_plan_f_r2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type);


extern 
void 
dtfft_execute_r2r(dtfft_plan plan, double *in, double *out, int transpose_type, double *work);

extern 
void 
dtfft_execute_f_r2r(dtfft_plan plan, float *in, float *out, int transpose_type, float *work);

extern 
void
dtfft_execute_c2c(dtfft_plan plan, dtfft_complex *in, dtfft_complex *out, int transpose_type, dtfft_complex *work);

extern
void
dtfft_execute_f_c2c(dtfft_plan plan, dtfftf_complex *in, dtfftf_complex *out, int transpose_type, dtfftf_complex *work);

extern
void 
dtfft_execute_r2c(dtfft_plan plan, double *in, dtfft_complex *out, dtfft_complex *work);

extern
void 
dtfft_execute_f_r2c(dtfft_plan plan, float *in, dtfftf_complex *out, dtfftf_complex *work);

extern
void 
dtfft_execute_c2r(dtfft_plan plan, dtfft_complex *in, double *out, dtfft_complex *work);

extern
void 
dtfft_execute_f_c2r(dtfft_plan plan, dtfftf_complex *in, float *out, dtfftf_complex *work);


extern
void
dtfft_destroy(dtfft_plan plan);

extern
int 
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts);

extern
int
dtfft_get_worker_size(dtfft_plan plan, int *starts, int *counts);


#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT
