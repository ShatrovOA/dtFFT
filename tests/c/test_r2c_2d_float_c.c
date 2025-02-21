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
#ifdef DTFFT_TRANSPOSE_ONLY
  return 0;
#else
  dtfft_plan_t plan;
  int32_t nx = 35, ny = 44;
  float *inout, *check, *work;
  int comm_rank, comm_size;
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

  dtfft_executor_t executor;
#ifdef DTFFT_WITH_MKL
  executor = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_CUFFT)
  executor = DTFFT_EXECUTOR_CUFFT;
#elif defined(DTFFT_WITH_VKFFT)
  executor = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3;
#endif

  assign_device_to_process();

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor, &plan) )

  // Get local sizes
  size_t alloc_size, el_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  DTFFT_CALL( dtfft_get_element_size(plan, &el_size) )

  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];
  // Allocate buffers
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&inout) )
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&work) )

  check = (float*) malloc(el_size * in_size);

#pragma acc enter data create(inout[0:alloc_size-1], check[0:in_size-1], work[0:alloc_size-1])

#pragma acc parallel loop present(inout, check)
  for (size_t i = 0; i < in_size; i++) {
    inout[i] = check[i] = (float)i / (float)nx / (float)ny;
  }

  // Forward transpose
  double tf = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(inout, work)
  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_EXECUTE_FORWARD, work) );

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tf += MPI_Wtime();

  // Normalize
#pragma acc parallel loop present(inout)
  for (size_t i = 0; i < 2 * out_size; i++) {
    inout[i] /= (float) (nx * ny);
  }

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(inout, work)
  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_EXECUTE_BACKWARD, work) );

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tb += MPI_Wtime();


  float local_error = -1.0;
#pragma acc parallel loop present(inout, check) reduction(max:local_error)
  for (size_t i = 0; i < in_size; i++) {
    float error = fabs(check[i] - inout[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, NULL, local_error, tf, tb);

  // Deallocate buffers
  DTFFT_CALL( dtfft_mem_free(plan, inout) )
  DTFFT_CALL( dtfft_mem_free(plan, work) )
  free(check);
  // Destroy plan
  DTFFT_CALL( dtfft_destroy(&plan) )

  MPI_Finalize();
  return 0;
#endif
}