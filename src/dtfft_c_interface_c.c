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
#include <mpi.h>
#include "../include/dtfft.h"
#include "dtfft_private.h"
#include <stdio.h>

// This subroutine translates C communicator to Fortran
// It reverses arrays and creates new comm that will be passed to dtFFT
MPI_Fint dtfft_translate_c_comm(MPI_Comm comm)
{
  int status;
  MPI_Topo_test(comm, &status);
  if (status == MPI_UNDEFINED) return MPI_Comm_c2f(comm);
  if (status == MPI_CART) {
    int ndims;
    MPI_Cartdim_get(comm, &ndims);
    int dims[ndims], fdims[ndims];
    int periods[ndims], fperiods[ndims];
    int coords[ndims];
    MPI_Cart_get(comm, ndims, dims, periods, coords);
    int d;
    for (d = 0; d < ndims; d++) {
      fdims[d] = dims[ndims - 1 - d];
      fperiods[d] = periods[ndims - 1- d];
    }
    MPI_Comm inv_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, fdims, fperiods, 1, &inv_comm);
    return MPI_Comm_c2f(inv_comm);
  } else {
    printf("Error at 'dtfft_translate_c_comm', unknown communicator type..\n");
    printf("MPI_COMM_WORLD will be used..\n");
    return MPI_Comm_c2f(MPI_COMM_WORLD);
  }
};


dtfft_plan 
dtfft_create_plan_r2r_3d(MPI_Comm comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type) 
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_R2R_3D;
  dtfft_create_plan_r2r_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_r2r_2d(MPI_Comm comm, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_R2R_2D;
  dtfft_create_plan_r2r_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, in_kinds, out_kinds, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_f_r2r_3d(MPI_Comm comm, int nz, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type) 
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_R2R_3D;
  dtfft_create_plan_f_r2r_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, in_kinds, out_kinds, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_f_r2r_2d(MPI_Comm comm, int ny, int nx, int *in_kinds, int *out_kinds, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_R2R_2D;
  dtfft_create_plan_f_r2r_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, in_kinds, out_kinds, effort_flag, executor_type);
  return plan;
};


dtfft_plan 
dtfft_create_plan_c2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type) 
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_C2C_3D;
  dtfft_create_plan_c2c_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_c2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_C2C_2D;
  dtfft_create_plan_c2c_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_f_c2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type) 
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_C2C_3D;
  dtfft_create_plan_f_c2c_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, effort_flag, executor_type);
  return plan;
};

dtfft_plan 
dtfft_create_plan_f_c2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_C2C_2D;
  dtfft_create_plan_f_c2c_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, effort_flag, executor_type);
  return plan;
};

dtfft_plan
dtfft_create_plan_r2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_R2C_3D;
  dtfft_create_plan_r2c_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, effort_flag, executor_type);
  return plan;
}

dtfft_plan
dtfft_create_plan_r2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_R2C_2D;
  dtfft_create_plan_r2c_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, effort_flag, executor_type);
  return plan;
}

dtfft_plan
dtfft_create_plan_f_r2c_3d(MPI_Comm comm, int nz, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_R2C_3D;
  dtfft_create_plan_f_r2c_3d_c(&plan.plan, dtfft_translate_c_comm(comm), nz, ny, nx, effort_flag, executor_type);
  return plan;
}

dtfft_plan
dtfft_create_plan_f_r2c_2d(MPI_Comm comm, int ny, int nx, int effort_flag, int executor_type)
{
  dtfft_plan plan;
  plan.plan_type = DTFFT_PLAN_F_R2C_2D;
  dtfft_create_plan_f_r2c_2d_c(&plan.plan, dtfft_translate_c_comm(comm), ny, nx, effort_flag, executor_type);
  return plan;
}


void 
dtfft_execute_r2r(dtfft_plan plan, double *in, double *out, int transpose_type, double *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2R_2D) {
      dtfft_execute_r2r_2d_c(plan.plan, in, out, transpose_type);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_R2R_3D) {
      dtfft_execute_r2r_3d_c(plan.plan, in, out, transpose_type, work);
      return;
    }
  }
};

void 
dtfft_execute_f_r2r(dtfft_plan plan, float *in, float *out, int transpose_type, float *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_F_R2R_2D) {
      dtfft_execute_f_r2r_2d_c(plan.plan, in, out, transpose_type);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_F_R2R_3D) {
      dtfft_execute_f_r2r_3d_c(plan.plan, in, out, transpose_type, work);
      return;
    }
  }
};

void 
dtfft_execute_c2c(dtfft_plan plan, dtfft_complex *in, dtfft_complex *out, int transpose_type, dtfft_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_C2C_2D) {
      dtfft_execute_c2c_2d_c(plan.plan, in, out, transpose_type);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_C2C_3D) {
      dtfft_execute_c2c_3d_c(plan.plan, in, out, transpose_type, work);
      return;
    }
  }
};

void 
dtfft_execute_f_c2c(dtfft_plan plan, dtfftf_complex *in, dtfftf_complex *out, int transpose_type, dtfftf_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_F_C2C_2D) {
      dtfft_execute_f_c2c_2d_c(plan.plan, in, out, transpose_type);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_F_C2C_3D) {
      dtfft_execute_f_c2c_3d_c(plan.plan, in, out, transpose_type, work);
      return;
    }
  }
};

void 
dtfft_execute_r2c(dtfft_plan plan, double *in, dtfft_complex *out, dtfft_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2C_2D) {
      dtfft_execute_r2c_2d_c(plan.plan, in, out, work);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_R2C_3D) {
      dtfft_execute_r2c_3d_c(plan.plan, in, out, work);
      return;
    }
  }
};

void 
dtfft_execute_f_r2c(dtfft_plan plan, float *in, dtfftf_complex *out, dtfftf_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_F_R2C_2D) {
      dtfft_execute_f_r2c_2d_c(plan.plan, in, out, work);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_F_R2C_3D) {
      dtfft_execute_f_r2c_3d_c(plan.plan, in, out, work);
      return;
    }
  }
};

void 
dtfft_execute_c2r(dtfft_plan plan, dtfft_complex *in, double *out, dtfft_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2C_2D) {
      dtfft_execute_c2r_2d_c(plan.plan, in, out, work);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_R2C_3D) {
      dtfft_execute_c2r_3d_c(plan.plan, in, out, work);
      return;
    }
  }
};

void 
dtfft_execute_f_c2r(dtfft_plan plan, dtfftf_complex *in, float *out, dtfftf_complex *work)
{
  if (plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_F_R2C_2D) {
      dtfft_execute_f_c2r_2d_c(plan.plan, in, out, work);
      return;
    } else if (plan.plan_type == DTFFT_PLAN_F_R2C_3D) {
      dtfft_execute_f_c2r_3d_c(plan.plan, in, out, work);
      return;
    }
  }
};

void
dtfft_destroy(dtfft_plan plan)
{
  if(plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2R_3D || plan.plan_type == DTFFT_PLAN_F_R2R_3D) dtfft_destroy_r2r_3d_c(plan.plan);
    if(plan.plan_type == DTFFT_PLAN_R2R_2D || plan.plan_type == DTFFT_PLAN_F_R2R_2D) dtfft_destroy_r2r_2d_c(plan.plan);
    if(plan.plan_type == DTFFT_PLAN_C2C_3D || plan.plan_type == DTFFT_PLAN_F_C2C_3D) dtfft_destroy_c2c_3d_c(plan.plan);
    if(plan.plan_type == DTFFT_PLAN_C2C_2D || plan.plan_type == DTFFT_PLAN_F_C2C_2D) dtfft_destroy_c2c_2d_c(plan.plan);
    if(plan.plan_type == DTFFT_PLAN_R2C_3D || plan.plan_type == DTFFT_PLAN_F_R2C_3D) dtfft_destroy_r2c_3d_c(plan.plan);
    if(plan.plan_type == DTFFT_PLAN_R2C_2D || plan.plan_type == DTFFT_PLAN_F_R2C_2D) dtfft_destroy_r2c_2d_c(plan.plan);
  }
  plan.plan = NULL;
  plan.plan_type = -1;
}

int 
dtfft_get_local_sizes(dtfft_plan plan, int *in_starts, int *in_counts, int *out_starts, int *out_counts)
{
  int alloc_size;
  if(plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2R_3D || plan.plan_type == DTFFT_PLAN_F_R2R_3D) {
      dtfft_get_local_sizes_r2r_3d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_R2R_2D || plan.plan_type == DTFFT_PLAN_F_R2R_2D) {
      dtfft_get_local_sizes_r2r_2d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_C2C_3D || plan.plan_type == DTFFT_PLAN_F_C2C_3D) {
      dtfft_get_local_sizes_c2c_3d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_C2C_2D || plan.plan_type == DTFFT_PLAN_F_C2C_2D) {
      dtfft_get_local_sizes_c2c_2d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_R2C_3D || plan.plan_type == DTFFT_PLAN_F_R2C_3D) {
      dtfft_get_local_sizes_r2c_3d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size); 
    } else if(plan.plan_type == DTFFT_PLAN_R2C_2D || plan.plan_type == DTFFT_PLAN_F_R2C_2D) {
      dtfft_get_local_sizes_r2c_2d_c(plan.plan, in_starts, in_counts, out_starts, out_counts, &alloc_size);
    }
  } else {
    printf("Error at 'dtfft_get_local_sizes', plan has not been created...\n");
    return -1;
  }
  return alloc_size;
};

int
dtfft_get_worker_size(dtfft_plan plan, int *starts, int *counts)
{
  int alloc_size;
  if(plan.plan) {
    if(plan.plan_type == DTFFT_PLAN_R2R_3D || plan.plan_type == DTFFT_PLAN_F_R2R_3D) {
      dtfft_get_worker_size_r2r_3d_c(plan.plan, starts, counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_C2C_3D || plan.plan_type == DTFFT_PLAN_F_C2C_3D) {
      dtfft_get_worker_size_c2c_3d_c(plan.plan, starts, counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_R2C_2D || plan.plan_type == DTFFT_PLAN_F_R2C_2D) {
      dtfft_get_worker_size_r2c_2d_c(plan.plan, starts, counts, &alloc_size);
    } else if(plan.plan_type == DTFFT_PLAN_R2C_3D || plan.plan_type == DTFFT_PLAN_F_R2C_3D) {
      dtfft_get_worker_size_r2c_3d_c(plan.plan, starts, counts, &alloc_size);
    } 
  } else {
    printf("Error at 'get_worker_size', plan has not been created...\n");
    return -1;
  }
  return alloc_size;
}
