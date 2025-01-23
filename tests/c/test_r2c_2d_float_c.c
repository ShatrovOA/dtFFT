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
#include "test_utils.h"

int main(int argc, char *argv[])
{
  dtfft_plan_t plan;
  int32_t nx = 35, ny = 44;
  float *inout, *check, *work;
  int i, comm_rank, comm_size;
  int32_t in_counts[2], out_counts[2], n[2] = {ny, nx};

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

  dtfft_executor_t executor_type;
#ifdef DTFFT_WITH_MKL
  executor_type = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_CUFFT)
  executor_type = DTFFT_EXECUTOR_CUFFT;
#elif defined(DTFFT_WITH_VKFFT)
  executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  if(comm_rank == 0) {
    printf("No available executors found, skipping test...\n");
  }
  MPI_Finalize();
  return 0;
#endif

  assign_device_to_process();

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor_type, &plan) )

  // Get local sizes
  int64_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )

  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];
  // Allocate buffers
  inout = (float*) malloc(sizeof(float) * alloc_size);
  check = (float*) malloc(sizeof(float) * in_size);
  work =  (float*) malloc(sizeof(float) * alloc_size);

#pragma acc enter data create(inout[0:alloc_size-1], check[0:in_size-1], work[0:alloc_size-1])

#pragma acc parallel loop present(inout, check)
  for (i = 0; i < in_size; i++) {
    inout[i] = check[i] = (float)i / (float)nx / (float)ny;
  }

  // Forward transpose
  double tf = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(inout, work)
  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_TRANSPOSE_OUT, work) );

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tf += MPI_Wtime();

  // Normalize
#pragma acc parallel loop present(inout)
  for (i = 0; i < 2 * out_size; i++) {
    inout[i] /= (float) (nx * ny);
  }

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(inout, work)
  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_TRANSPOSE_IN, work) );

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tb += MPI_Wtime();


  float local_error = -1.0;
#pragma acc parallel loop present(inout, check) reduction(max:local_error)
  for (i = 0; i < in_size; i++) {
    float error = fabs(check[i] - inout[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, NULL, local_error, tf, tb);

  // Destroy plan
  dtfft_destroy(&plan);

  // Deallocate buffers
  free(inout);
  free(check);

  MPI_Finalize();
  return 0;
}