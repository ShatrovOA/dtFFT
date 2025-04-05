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
  int32_t in_counts[3], out_counts[3], n[3] = {nz, ny, nx};

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

  dtfft_executor_t executor;
#ifdef DTFFT_WITH_MKL
  executor = DTFFT_EXECUTOR_MKL;
#elif defined (DTFFT_WITH_FFTW)
  executor = DTFFT_EXECUTOR_FFTW3;
#else
# if !defined(DTFFT_WITH_CUDA)
  if(comm_rank == 0) {
    printf("Missing HOST FFT Executor\n");
  }
  MPI_Finalize();
  return 0;
# endif
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
# if defined(DTFFT_WITH_VKFFT)
    executor = DTFFT_EXECUTOR_VKFFT;
# elif defined (DTFFT_WITH_CUFFT)
    executor = DTFFT_EXECUTOR_CUFFT;
# else
    if(comm_rank == 0) {
      printf("Missing CUDA FFT Executor\n");
    }
    MPI_Finalize();
    return 0;
# endif
  }
#endif

  dtfft_config_t config;
  DTFFT_CALL( dtfft_create_config(&config) )

  config.enable_z_slab = false;
#if defined(DTFFT_WITH_CUDA)
  config.backend = DTFFT_BACKEND_MPI_P2P_PIPELINED;
  config.platform = DTFFT_PLATFORM_CUDA;
#endif
  DTFFT_CALL( dtfft_set_config(config) )

  attach_gpu_to_process();
  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(3, n, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor, &plan) )

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

  for (size_t i = 0; i < in_size; i++) {
    check[i] = (double)(i) / (double)(in_size);
  }

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )

  cudaStream_t stream;

  if ( platform == DTFFT_PLATFORM_CUDA ) {
    dtfft_stream_t dtfftStream;
    DTFFT_CALL( dtfft_get_stream(plan, &dtfftStream) )
    stream = (cudaStream_t)dtfftStream;
    CUDA_SAFE_CALL( cudaMemcpy(in, check, el_size * in_size, cudaMemcpyHostToDevice) )
  } else {
    memcpy(in, check, el_size * in_size);
  }
#else
  memcpy(in, check, el_size * in_size);
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
  if ( executor != DTFFT_EXECUTOR_NONE ) {
#if defined(DTFFT_WITH_CUDA)
    if ( platform == DTFFT_PLATFORM_CUDA ) {
      scaleComplexDouble(out, out_size, nx * ny * nz, stream);
      CUDA_SAFE_CALL( cudaDeviceSynchronize() )
    } else {
      scaleComplexDoubleHost(out, out_size, nx * ny * nz);
    }
#else
    scaleComplexDoubleHost(out, out_size, nx * ny * nz);
#endif
  }

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_EXECUTE_BACKWARD, aux) );

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
  }
#endif
  tb += MPI_Wtime();

  double local_error;
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    dtfftf_complex *h_in;

    h_in = (dtfftf_complex*) malloc(el_size * in_size);
    CUDA_SAFE_CALL( cudaMemcpy(h_in, in, el_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkDouble(check, h_in, in_size);
    free(h_in);
  } else {
    local_error = checkDouble(check, in, in_size);
  }
#else
  local_error = checkDouble(check, in, in_size);
#endif

  reportDouble(&tf, &tb, &local_error, &nx, &ny, &nz);

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