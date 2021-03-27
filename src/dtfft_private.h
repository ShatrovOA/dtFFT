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

#ifndef DTFFT_PRIVATE_H
#define DTFFT_PRIVATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <mpi.h>
#include "../include/dtfft.h"


#define DTFFT_PLAN_R2R_3D 0
#define DTFFT_PLAN_C2C_3D 2
#define DTFFT_PLAN_R2R_2D 4
#define DTFFT_PLAN_C2C_2D 6
#define DTFFT_PLAN_R2C_3D 8
#define DTFFT_PLAN_R2C_2D 10


#define DTFFT_PLAN_F_R2R_3D 1
#define DTFFT_PLAN_F_C2C_3D 3
#define DTFFT_PLAN_F_R2R_2D 5
#define DTFFT_PLAN_F_C2C_2D 7
#define DTFFT_PLAN_F_R2C_3D 9
#define DTFFT_PLAN_F_R2C_2D 11

// Plan creators
// R2R plans
extern
void
dtfft_create_plan_r2r_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_r2r_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_r2r_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_r2r_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type);
// C2C plans
extern
void
dtfft_create_plan_c2c_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_c2c_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_c2c_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_c2c_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int effort_flag, int executor_type);
// R2C plan
extern
void
dtfft_create_plan_r2c_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_r2c_2d_c(void *plan, MPI_Fint comm, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_r2c_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int effort_flag, int executor_type);
extern
void
dtfft_create_plan_f_r2c_3d_c(void *plan, MPI_Fint comm, int nz, int ny, int nx, int effort_flag, int executor_type);


// Plan Executors
// R2R
extern 
void 
dtfft_execute_r2r_3d_c(void *plan, double *in, double *out, int transpose_type, double *work);
extern 
void 
dtfft_execute_r2r_2d_c(void *plan, double *in, double *out, int transpose_type);
extern 
void 
dtfft_execute_f_r2r_3d_c(void *plan, float *in, float *out, int transpose_type, float *work);
extern 
void 
dtfft_execute_f_r2r_2d_c(void *plan, float *in, float *out, int transpose_type);

// C2C
extern 
void 
dtfft_execute_c2c_3d_c(void *plan, dtfft_complex *in, dtfft_complex *out, int transpose_type, dtfft_complex *work);
extern 
void 
dtfft_execute_c2c_2d_c(void *plan, dtfft_complex *in, dtfft_complex *out, int transpose_type);
extern 
void 
dtfft_execute_f_c2c_3d_c(void *plan, dtfftf_complex *in, dtfftf_complex *out, int transpose_type, dtfftf_complex *work);
extern 
void 
dtfft_execute_f_c2c_2d_c(void *plan, dtfftf_complex *in, dtfftf_complex *out, int transpose_type);

// R2C
extern 
void 
dtfft_execute_r2c_2d_c(void *plan, double *in, dtfft_complex *out, dtfft_complex *work);
extern 
void 
dtfft_execute_f_r2c_2d_c(void *plan, float *in, dtfftf_complex *out, dtfftf_complex *work);
extern 
void 
dtfft_execute_c2r_2d_c(void *plan, dtfft_complex *in, double *out, dtfft_complex *work);
extern 
void 
dtfft_execute_f_c2r_2d_c(void *plan, dtfftf_complex *in, float *out, dtfftf_complex *work);
extern 
void 
dtfft_execute_r2c_3d_c(void *plan, double *in, dtfft_complex *out, dtfft_complex *work);
extern 
void 
dtfft_execute_f_r2c_3d_c(void *plan, float *in, dtfftf_complex *out, dtfftf_complex *work);
extern 
void 
dtfft_execute_c2r_3d_c(void *plan, dtfft_complex *in, double *out, dtfft_complex *work);
extern 
void 
dtfft_execute_f_c2r_3d_c(void *plan, dtfftf_complex *in, float *out, dtfftf_complex *work);

// Plan destructors
extern 
void 
dtfft_destroy_r2r_3d_c(void *plan);
extern 
void 
dtfft_destroy_r2r_2d_c(void *plan);
extern 
void 
dtfft_destroy_c2c_3d_c(void *plan);
extern 
void 
dtfft_destroy_c2c_2d_c(void *plan);
extern
void
dtfft_destroy_r2c_2d_c(void *plan);
extern
void
dtfft_destroy_r2c_3d_c(void *plan);

// Local sizes
extern
void 
dtfft_get_local_sizes_r2r_3d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);
extern
void 
dtfft_get_local_sizes_r2r_2d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);
extern
void 
dtfft_get_local_sizes_c2c_3d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);
extern
void 
dtfft_get_local_sizes_c2c_2d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);
extern
void
dtfft_get_local_sizes_r2c_3d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);
extern
void
dtfft_get_local_sizes_r2c_2d_c(void *plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts, int *alloc_size);

// Work buffer
extern
void
dtfft_get_worker_size_r2r_3d_c(void *plan, int *starts, int *counts, int *alloc_size);
extern
void
dtfft_get_worker_size_c2c_3d_c(void *plan, int *starts, int *counts, int *alloc_size);
extern
void
dtfft_get_worker_size_r2c_3d_c(void *plan, int *starts, int *counts, int *alloc_size);
extern
void
dtfft_get_worker_size_r2c_2d_c(void *plan, int *starts, int *counts, int *alloc_size);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // DTFFT
