/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
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
  int32_t nx = 16, ny = 32, nz = 70;
  double *in, *check;
  dtfft_complex *out, *aux;
  int comm_rank, comm_size;
  int32_t in_starts[3], in_counts[3], out_counts[3], n[3] = {nz, ny, nx};

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

  dtfft_executor_t executor = DTFFT_EXECUTOR_NONE;
#ifdef DTFFT_WITH_MKL
  executor = DTFFT_EXECUTOR_MKL;
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3;
#endif

  attach_gpu_to_process();

  dtfft_config_t config;
  DTFFT_CALL( dtfft_create_config(&config) )

#ifdef DTFFT_WITH_CUDA
  cudaStream_t stream;
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
# if defined(DTFFT_WITH_VKFFT)
    executor = DTFFT_EXECUTOR_VKFFT;
# elif defined (DTFFT_WITH_CUFFT)
    executor = DTFFT_EXECUTOR_CUFFT;
# else
    executor = DTFFT_EXECUTOR_NONE;
# endif
    CUDA_SAFE_CALL( cudaStreamCreate(&stream) )
    config.stream = (dtfft_stream_t)stream;
    config.backend = DTFFT_BACKEND_MPI_P2P_PIPELINED;
  }
#endif

  if ( executor == DTFFT_EXECUTOR_NONE ) {
    if ( comm_rank == 0 ) printf("Could not find valid R2C FFT executor, skipping test\n");
    MPI_Finalize();
    return 0;
  }

  config.enable_z_slab = false;
  DTFFT_CALL( dtfft_set_config(&config) )

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(3, n, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor, &plan) )
  DTFFT_CALL( dtfft_get_local_sizes(plan, in_starts, in_counts, NULL, NULL, NULL) )
  DTFFT_CALL( dtfft_destroy(&plan) )

  // Recreate plan with pencil
  dtfft_pencil_t pencil;
  pencil.ndims = 3;
  for (int i = 0; i < 3; i++) {
    pencil.starts[i] = in_starts[i];
    pencil.counts[i] = in_counts[i];
  }
  DTFFT_CALL( dtfft_create_plan_r2c_pencil(&pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, executor, &plan) )
  DTFFT_CALL( dtfft_report(plan) )

  // Get local sizes
  size_t alloc_size, el_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  DTFFT_CALL( dtfft_get_element_size(plan, &el_size) )
  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];
  size_t out_size = out_counts[0] * out_counts[1] * out_counts[2];

  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&in) )
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&out) )
  DTFFT_CALL( dtfft_mem_alloc(plan, el_size * alloc_size, (void**)&aux) )

  // Allocate buffers
  check = (double*) malloc(el_size * in_size);
  setTestValuesDouble(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )
  doubleH2D(check, in, in_size, (int32_t)platform);
#else
  doubleH2D(check, in, in_size);
#endif

  // Forward transpose
  double tf = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_EXECUTE_FORWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
  }
#endif
  tf += MPI_Wtime();

  // Clean input buffer for possible error check
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaMemset(in, -1, in_size * el_size) )
  } else {
    for (size_t i = 0; i < in_size; i++) {
      in[i] = -1.0;
    }
  }
#else
  for (size_t i = 0; i < in_size; i++) {
    in[i] = -1.0;
  }
#endif

  // Normalize
#if defined(DTFFT_WITH_CUDA)
  scaleComplexDouble((int32_t)executor, out, out_size, nx * ny * nz, (int32_t)platform, stream);
#else
  scaleComplexDouble((int32_t)executor, out, out_size, nx * ny * nz);
#endif

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_EXECUTE_BACKWARD, aux) );

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportDouble(nx * ny * nz, tf, tb, in, in_size, check, (int32_t)platform);
#else
  checkAndReportDouble(nx * ny * nz, tf, tb, in, in_size, check);
#endif

  // Deallocate buffers
  free(check);
  DTFFT_CALL( dtfft_mem_free(plan, in) )
  DTFFT_CALL( dtfft_mem_free(plan, out) )
  DTFFT_CALL( dtfft_mem_free(plan, aux) )

  // Destroy plan
  DTFFT_CALL( dtfft_destroy(&plan) )

  MPI_Finalize();
  return 0;
#endif
}