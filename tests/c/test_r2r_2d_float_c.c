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

#include <dtfft.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "test_utils.h"

int main(int argc, char *argv[]) {

  dtfft_plan_t plan;
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  int32_t nx = 2048, ny = 64;
#else
  int32_t nx = 32, ny = 32;
#endif
  float *in, *out, *check;
  int comm_rank, comm_size;
  int32_t in_counts[2], out_counts[2], n[2] = {ny, nx};
  dtfft_r2r_kind_t kinds[2] = {DTFFT_DST_1, DTFFT_DST_2};

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

  dtfft_executor_t executor = DTFFT_EXECUTOR_NONE;
#ifdef DTFFT_WITH_FFTW
  executor = DTFFT_EXECUTOR_FFTW3;
#endif
#ifdef DTFFT_WITH_CUDA
  bool is_cuda_platform = false;
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
    is_cuda_platform = true;
# if defined(DTFFT_WITH_VKFFT)
    executor = DTFFT_EXECUTOR_VKFFT;
# else
    executor = DTFFT_EXECUTOR_NONE;
# endif
  }
#endif

  dtfft_config_t config;
  DTFFT_CALL( dtfft_create_config(&config) )
  config.backend = DTFFT_BACKEND_MPI_P2P_PIPELINED;

#if defined(DTFFT_WITH_CUDA)
  if ( is_cuda_platform ) {
#if defined(DTFFT_WITH_NCCL)
    config.backend = DTFFT_BACKEND_NCCL_PIPELINED;
#elif defined(DTFFT_WITH_NVSHMEM)
    config.backend = DTFFT_BACKEND_CUFFTMP;
#endif
  }
  config.platform = DTFFT_PLATFORM_CUDA;
#endif

  DTFFT_CALL( dtfft_set_config(&config) )

  attach_gpu_to_process();

  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2r(2, n, kinds, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor, &plan) )

  size_t element_size;
  DTFFT_CALL( dtfft_get_element_size(plan, &element_size) )
  if ( element_size != sizeof(float) ) {
    fprintf(stderr, "element_size != sizeof(float)\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  size_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(float) * alloc_size, (void**)&in) )
  DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(float) * alloc_size, (void**)&out) )

  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];

  check = (float*) malloc(sizeof(float) * in_size);
  setTestValuesFloat(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )
  floatH2D(check, in, in_size, (int32_t)platform);
#else
  floatH2D(check, in, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  dtfft_execute(plan, in, out, DTFFT_EXECUTE_FORWARD, NULL);
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

  size_t scale_size = 4 * nx * (ny + 1);
#if defined(DTFFT_WITH_CUDA)
  scaleFloat((int32_t)executor, out, out_size, scale_size, (int32_t)platform, NULL);
#else
  scaleFloat((int32_t)executor, out, out_size, scale_size);
#endif

  double tb = 0.0 - MPI_Wtime();
  dtfft_execute(plan, out, in, DTFFT_EXECUTE_BACKWARD, NULL);
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportFloat(nx * ny, tf, tb, in, in_size, check, (int32_t)platform);
#else
  checkAndReportFloat(nx * ny, tf, tb, in, in_size, check);
#endif

  DTFFT_CALL( dtfft_mem_free(plan, in) )
  DTFFT_CALL( dtfft_mem_free(plan, out) )
  free(check);

  DTFFT_CALL( dtfft_destroy(&plan) )

  MPI_Finalize();
  return 0;
}