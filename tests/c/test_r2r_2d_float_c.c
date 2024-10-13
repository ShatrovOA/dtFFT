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

int main(int argc, char *argv[]) {

  dtfft_plan plan;
  int nx = 32, ny = 32;
  float *in, *out, *check;
  int i,j, comm_rank, comm_size;
  int in_counts[2], out_counts[2], n[2] = {ny, nx};
  int kinds[2] = {DTFFT_DST_1, DTFFT_DST_2};

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: r2r_2d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d\n", nx, ny);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

#ifdef DTFFT_WITH_FFTW
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2r(2, n, kinds, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor_type, &plan) )

  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, NULL) )

  in = (float*) malloc(sizeof(float) * in_counts[0] * in_counts[1]);
  out = (float*) malloc(sizeof(float) * out_counts[0] * out_counts[1]);
  check = (float*) malloc(sizeof(float) * in_counts[0] * in_counts[1]);

  for (i = 0; i < in_counts[1]; i++) { // x direction
    for (j = 0; j < in_counts[0]; j++) { // y direction
        in[i * in_counts[0] + j] = check[i * in_counts[0] + j] = 15.0;
    }
  }

  double tf = 0.0 - MPI_Wtime();
  dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, NULL);
  tf += MPI_Wtime();

#ifndef DTFFT_TRANSPOSE_ONLY
  for (i = 0; i < out_counts[1]; i++) { // y direction
    for (j = 0; j < out_counts[0]; j++) { // x direction
        out[i * out_counts[0] + j] /= (float) (4 * nx * (ny + 1));
    }
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

  float local_error = -1.0;
  for (i = 0; i < in_counts[1]; i++) {
    for (j = 0; j < in_counts[0]; j++) {
      float error = fabs(check[i * in_counts[0] + j] - in[i * in_counts[0] + j]);
      local_error = error > local_error ? error : local_error;
    }
  }
  float global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-5) {
      printf("Test 'r2r_2d_float_c' PASSED!\n");
    } else {
      printf("Test 'r2r_2d_float_c' FAILED, error = %e\n", global_error);
      return -1;
    }
    printf("----------------------------------------\n");
  }

  dtfft_destroy(&plan);

  MPI_Finalize();
  return 0;
}