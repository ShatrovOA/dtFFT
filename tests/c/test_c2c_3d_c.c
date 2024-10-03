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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char *argv[])
{
  dtfft_plan plan;
  int nx = 32, ny = 32, nz = 32;
  dtfft_complex *in, *out, *check, *aux;
  int i,comm_rank, comm_size;
  int in_counts[3], out_counts[3], n[3] = {nz, ny, nx};
  int grid_dims[3] = {1, 0, 0}, periods[3] = {0, 0, 0};
  MPI_Comm grid_comm;

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_3d       |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

#ifdef DTFFT_WITH_MKL
  int executor_type = DTFFT_EXECUTOR_MKL;
#elif !defined(DTFFT_WITHOUT_FFTW)
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  MPI_Dims_create(comm_size, 3, grid_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 3, grid_dims, periods, 0, &grid_comm);
  // Create plan
  DTFFT_CALL( dtfft_create_plan_c2c(3, n, grid_comm, DTFFT_DOUBLE, DTFFT_PATIENT, executor_type, &plan) )

  size_t aux_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &aux_size) )

  in = (dtfft_complex*) malloc(sizeof(dtfft_complex) * in_counts[0] * in_counts[1] * in_counts[2]);
  out = (dtfft_complex*) malloc(sizeof(dtfft_complex) * out_counts[0] * out_counts[1] * out_counts[2]);
  aux = (dtfft_complex*) malloc(sizeof(dtfft_complex) * aux_size);
  check = (dtfft_complex*) malloc(sizeof(dtfft_complex) * in_counts[0] * in_counts[1] * in_counts[2]);

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i][0] = check[i][0] = 1.0;
    in[i][1] = check[i][1] = 3.0;
  }

  double tf = 0.0 - MPI_Wtime();
#ifdef DTFFT_TRANSPOSE_ONLY
  dtfft_transpose(plan, in, aux, DTFFT_TRANSPOSE_X_TO_Y);
  dtfft_transpose(plan, aux, out, DTFFT_TRANSPOSE_Y_TO_Z);
#else
  dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux);
#endif
  tf += MPI_Wtime();

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i][0] = -1.0;
    in[i][1] = -2.0;
  }

#ifndef DTFFT_TRANSPOSE_ONLY
  for (i = 0; i < out_counts[0] * out_counts[1] * out_counts[2]; i++) {
    out[i][0] /= (double) (nx * ny * nz);
    out[i][1] /= (double) (nx * ny * nz);
  }
#endif
  double tb = 0.0 - MPI_Wtime();
  dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, NULL);
  tb += MPI_Wtime();

  double t_sum;
  MPI_Allreduce(&tf, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tf = t_sum / (double) comm_size;
  MPI_Allreduce(&tb, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tb = t_sum / (double) comm_size;

  if(comm_rank == 0) {
    printf("Forward execution time: %f\n", tf);
    printf("Backward execution time: %f\n", tb);
    printf("----------------------------------------\n");
  }

  double local_error = -1.0;
  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    double real_error = fabs(check[i][0] - in[i][0]);
    double cmplx_error = fabs(check[i][1] - in[i][1]);
    double error = fmax(real_error, cmplx_error);
    local_error = error > local_error ? error : local_error;
  }

  double global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-10) {
      printf("Test 'c2c_3d_c' PASSED!\n");
    } else {
      printf("Test 'c2c_3d_c' FAILED, error = %f\n", global_error);
      return -1;
    }
    printf("----------------------------------------\n");
  }
  dtfft_destroy(&plan);

  MPI_Finalize();
  return 0;
}