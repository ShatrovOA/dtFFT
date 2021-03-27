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

#include <dtfft.h>
#include <mpi.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) 
{
  dtfft_plan plan;
  int nx = 128, ny = 200, nz = 313;
  float *inout, *check, *work, err, temp, max_err;
  int i,j,k, comm_rank, comm_size;
  int in_counts[3], out_counts[3];
  MPI_Comm grid_comm;
  int f_kinds[3] = {FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10};
  int b_kinds[3] = {FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01};


  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: r2r_3d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  int ndims = 2;
  int dims[ndims]; dims[0] = 0; dims[1] = 0;
  int periods[ndims]; periods[0] = 0; periods[1] = 0;
  MPI_Dims_create(comm_size, ndims, dims);
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &grid_comm);

  // Create plan
  plan = dtfft_create_plan_f_r2r_3d(grid_comm, nz, ny, nx, f_kinds, b_kinds, DTFFT_ESTIMATE, DTFFT_EXECUTOR_FFTW3);

  int alloc_size = dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts);

  inout = (float*) malloc(sizeof(float) * alloc_size);
  check = (float*) malloc(sizeof(float) * in_counts[0] * in_counts[1] * in_counts[2]);

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++)
    inout[i] = check[i] = 55;

  int work_size = dtfft_get_worker_size(plan, NULL, NULL);
  work = (float*) malloc(sizeof(float) * work_size);

  double tf = 0.0 - MPI_Wtime();
  dtfft_execute_f_r2r(plan, inout, inout, DTFFT_TRANSPOSE_OUT, work);
  tf += MPI_Wtime();

  for (i = 0; i < out_counts[0] * out_counts[1] * out_counts[2]; i++)
    inout[i] /= (float) (8 * nx * ny * nz);

  double tb = 0.0 - MPI_Wtime();
  dtfft_execute_f_r2r(plan, inout, inout, DTFFT_TRANSPOSE_IN, work);
  tb += MPI_Wtime();

  double t_sum;
  MPI_Allreduce(&tf, &t_sum, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
  tf = t_sum / (double) comm_size;
  MPI_Allreduce(&tb, &t_sum, 1, MPI_DOUBLE, MPI_SUM, grid_comm);
  tb = t_sum / (double) comm_size;

  if(comm_rank == 0) {
    printf("Forward execution time: %f\n", tf);
    printf("Backward execution time: %f\n", tb);
    printf("----------------------------------------\n");
  }

  err = -1.0;
  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    temp = fabs(check[i] - inout[i]);
    if (temp > err) err = temp;
  }

  MPI_Allreduce(&err, &max_err, 1, MPI_FLOAT, MPI_MAX, grid_comm);

  if(comm_rank == 0) {
    if(max_err < 1e-3) {
      printf("Test 'r2r_3d_float_c' PASSED!\n");
    } else {
      printf("Test 'r2r_3d_float_c' FAILED, error = %f\n", max_err);
    }
    printf("----------------------------------------\n");
  }

  dtfft_destroy(plan);
  
  MPI_Finalize();
  return 0;
}