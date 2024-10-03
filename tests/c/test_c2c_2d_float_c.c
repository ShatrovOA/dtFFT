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

#include <complex.h>
#include <dtfft.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[])
{
  int i;
  int nx = 11, ny = 39;

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_2d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d\n", nx, ny);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  // Create plan
  int n[2] = {ny, nx};

#ifndef DTFFT_WITHOUT_FFTW
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  dtfft_plan plan;
  DTFFT_CALL( dtfft_create_plan_c2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_MEASURE, executor_type, &plan) )

  int in_counts[2], out_counts[2];
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, NULL) )

  dtfftf_complex *in, *out, *check;
  in = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * in_counts[0] * in_counts[1]);
  out = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * out_counts[0] * out_counts[1]);
  check = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * in_counts[0] * in_counts[1]);

  
  for (i = 0; i < in_counts[0] * in_counts[1]; i++) {
    in[i] = check[i] = (float)rand() / (float)(RAND_MAX) - (float)rand() / (float)(RAND_MAX) * I;
  }

  double tf = 0.0 - MPI_Wtime();
#ifdef DTFFT_TRANSPOSE_ONLY
  dtfft_transpose(plan, in, out, DTFFT_TRANSPOSE_X_TO_Y);
#else
  dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, NULL);
#endif
  tf += MPI_Wtime();

  for (i = 0; i < in_counts[0] * in_counts[1]; i++) {
    in[i] = -1.0 + 1.0 * I;
  }

#ifndef DTFFT_TRANSPOSE_ONLY
  for (i = 0; i < out_counts[0] * out_counts[1]; i++) {
    out[i] /= (float) (nx * ny);
  }
#endif

  double tb = 0.0 - MPI_Wtime();
#ifdef DTFFT_TRANSPOSE_ONLY
  dtfft_transpose(plan, out, in, DTFFT_TRANSPOSE_Y_TO_X);
#else
  dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, NULL);
#endif
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
  for (i = 0; i < in_counts[0] * in_counts[1]; i++) {
    float real_error = fabs(crealf(check[i]) - crealf(in[i]));
    float cmplx_error = fabs(cimagf(check[i]) - cimagf(in[i]));
    float error = fmax(real_error, cmplx_error);
    local_error = error > local_error ? error : local_error;
  }

  float global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-5) {
      printf("Test 'c2c_2d_float_c' PASSED!\n");
    } else {
      printf("Test 'c2c_2d_float_c' FAILED, error = %f\n", global_error);
      return -1;
    }
    printf("----------------------------------------\n");
  }

  dtfft_destroy(&plan);

  MPI_Finalize();
  return 0;
}