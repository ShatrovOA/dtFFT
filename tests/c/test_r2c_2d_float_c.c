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
#include <string.h>
#include <math.h>
#include "test_utils.h"

int main(int argc, char *argv[])
{
#if defined(DTFFT_TRANSPOSE_ONLY)
  printf("FFT Support is disabled in this build, skipping test\n");
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

  dtfft_executor_t executor = DTFFT_EXECUTOR_NONE;
#ifdef DTFFT_WITH_MKL
  executor = DTFFT_EXECUTOR_MKL;
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3;
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
# if defined(DTFFT_WITH_CUFFT)
    executor = DTFFT_EXECUTOR_CUFFT;
# elif defined(DTFFT_WITH_VKFFT)
    executor = DTFFT_EXECUTOR_VKFFT;
# else
    executor = DTFFT_EXECUTOR_NONE;
# endif
  }
#endif

  if ( executor == DTFFT_EXECUTOR_NONE ) {
    if ( comm_rank == 0 ) printf("Could not find valid R2C FFT executor, skipping test\n");
    MPI_Finalize();
    return 0;
  }

  attach_gpu_to_process();

#if defined(DTFFT_WITH_CUDA)
  dtfft_config_t conf;
  dtfft_create_config(&conf);
#ifdef DTFFT_WITH_NVSHMEM
  conf.backend = DTFFT_BACKEND_CUFFTMP;
#else
  conf.backend = DTFFT_BACKEND_MPI_A2A;
#endif
  dtfft_set_config(conf);
#endif

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor, &plan) )
  DTFFT_CALL( dtfft_report(plan) )
  // Get local sizes
  size_t alloc_size, el_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  DTFFT_CALL( dtfft_get_element_size(plan, &el_size) )

  if ( el_size != sizeof(float) ) {
    fprintf(stderr, "el_size != sizeof(float)\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];
  // Allocate buffers
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&inout) )
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&work) )

  check = (float*) malloc(el_size * in_size);
  setTestValuesFloat(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )
  floatH2D(check, inout, in_size, (int32_t)platform);
#else
  floatH2D(check, inout, in_size);
#endif

  // Forward transpose
  double tf = 0.0 - MPI_Wtime();

  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_EXECUTE_FORWARD, work) );

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  scaleComplexFloat((int32_t)executor, inout, out_size, nx * ny, (int32_t)platform, NULL);
#else
  scaleComplexFloat((int32_t)executor, inout, out_size, nx * ny);
#endif

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, inout, inout, DTFFT_EXECUTE_BACKWARD, work) );

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportFloat(nx * ny, tf, tb, inout, in_size, check, (int32_t)platform);
#else
  checkAndReportFloat(nx * ny, tf, tb, inout, in_size, check);
#endif

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