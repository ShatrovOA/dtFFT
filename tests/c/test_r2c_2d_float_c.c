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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) 
{
  dtfft_plan plan;
  int nx = 35, ny = 44;
  float *inout, *check;
  dtfftf_complex *work;
  int i, comm_rank, comm_size;
  int in_counts[2], out_counts[2];

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: r2c_2d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d\n", nx, ny);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  // Create plan
  plan = dtfft_create_plan_f_r2c_2d(MPI_COMM_WORLD, ny, nx, DTFFT_ESTIMATE, DTFFT_EXECUTOR_FFTW3);

  // Get local sizes
  int alloc_size = dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts);

  // Allocate buffers
  inout = (float*) malloc(2 * sizeof(float) * alloc_size);
  check = (float*) malloc(sizeof(float) * in_counts[0] * in_counts[1]);

  for (i = 0; i < in_counts[0] * in_counts[1]; i++) {
    inout[i] = check[i] =  1.0;
  }

  int work_size = dtfft_get_worker_size(plan, NULL, NULL);
  work = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * work_size);

  // Forward (DTFFT_TRANSPOSE_OUT) transpose
  double tf = 0.0 - MPI_Wtime();
  dtfft_execute_f_r2c(plan, inout, inout, work);
  tf += MPI_Wtime();

  // Normalize
  for (i = 0; i < 2 * out_counts[0] * out_counts[1]; i++) {
    inout[i] /= (float) (nx * ny);
  }

  // Backward (DTFFT_TRANSPOSE_IN) transpose
  double tb = 0.0 - MPI_Wtime();
  dtfft_execute_f_c2r(plan, inout, inout, work);
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
  float err = -1.0, temp, max_error;
  for (i = 0; i < in_counts[0] * in_counts[1]; i++) {
    temp = fabs(check[i] - inout[i]);
    if (temp > err) err = temp;
  }
  // Find maximum error
  MPI_Allreduce(&err, &max_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  if(comm_rank == 0) {
    if(max_error < 1e-5) {
      printf("Test 'r2c_2d_float_c' PASSED!\n");
    } else {
      printf("Test 'r2c_2d_float_c' FAILED, error = %E\n", max_error);
    }
    printf("----------------------------------------\n");
  }
  
  // Destroy plan
  dtfft_destroy(plan);

  // Deallocate buffers
  free(inout);
  free(check);
  
  MPI_Finalize();
  return 0;
}