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
  dtfft_plan plan;
  int nx = 16, ny = 32, nz = 70;
  double *in, *check;
  dtfft_complex *out;
  int i, comm_rank, comm_size;
  int in_counts[3], out_counts[3], n[3] = {nz, ny, nx};

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: r2c_3d       |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

#ifdef DTFFT_WITH_MKL
  int executor_type = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_VKFFT)
  int executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  if(comm_rank == 0) {
    printf("No available executors found, skipping test...\n");
  }
  MPI_Finalize();
  return 0;

  int executor_type = DTFFT_EXECUTOR_NONE;
#endif
  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(3, n, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor_type, &plan) )

  // Get local sizes
  size_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )

  // Allocate buffers
  in = (double*) malloc(sizeof(double) * alloc_size);
  out = (dtfft_complex*) malloc(sizeof(double) * alloc_size);
  check = (double*) malloc(sizeof(double) * alloc_size);
  // Allocate work buffer
  dtfft_complex *aux = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size);

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i] = check[i] =  1.0;
  }

  // Forward transpose
  double tf = 0.0 - MPI_Wtime();
  dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux);
  tf += MPI_Wtime();

  // Clean input buffer for possible error check
  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i] = -1.0;
  }

  // Normalize
  for (i = 0; i < out_counts[0] * out_counts[1] * out_counts[2]; i++) {
    out[i] /= (double) (nx * ny * nz);
  }

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
  dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, aux);
  tb += MPI_Wtime();

  double t_sum;
  // Aggregate execution time and find average per processor
  MPI_Allreduce(&tf, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tf = t_sum / (double) comm_size;
  MPI_Allreduce(&tb, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tb = t_sum / (double) comm_size;

  if(comm_rank == 0) {
    printf("Forward execution time: %f\n", tf);
    printf("Backward execution time: %f\n", tb);
    printf("----------------------------------------\n");
  }

  // Check error
  double local_error = -1.0;
  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    double error = fabs(check[i] - in[i]);
    local_error = error > local_error ? error : local_error;
  }
  double global_error;
  // Find maximum error
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if(comm_rank == 0) {
    if(global_error < 1e-10) {
      printf("Test 'r2c_3d_c' PASSED!\n");
    } else {
      printf("Test 'r2c_3d_c' FAILED, error = %E\n", global_error);
      return -1;
    }
    printf("----------------------------------------\n");
  }

  // Destroy plan
  dtfft_destroy(&plan);

  // Deallocate buffers
  free(in);
  free(out);
  free(aux);
  free(check);

  MPI_Finalize();
  return 0;
}