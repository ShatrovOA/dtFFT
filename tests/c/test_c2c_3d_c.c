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
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include "test_utils.h"


int main(int argc, char *argv[])
{
  dtfft_plan_t plan;
  int32_t nx = 512, ny = 64, nz = 32;
  dtfft_complex *in, *out, *check, *aux;
  int comm_rank, comm_size;
  int32_t in_counts[3], in_starts[3], out_counts[3];

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if ( comm_size != 2 ) {
    if ( comm_rank == 0 ) {
      printf("This test requires exactly 2 MPI processes.\n");
    }
    MPI_Finalize();
    return 0;
  }

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_3d       |\n");
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
#ifdef DTFFT_WITH_CUDA
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
# if defined (DTFFT_WITH_VKFFT)
  executor = DTFFT_EXECUTOR_VKFFT;
# elif defined (DTFFT_WITH_CUFFT)
  executor = DTFFT_EXECUTOR_CUFFT;
# else
  executor = DTFFT_EXECUTOR_NONE;
# endif
  }
#endif

  attach_gpu_to_process();

#if defined(DTFFT_WITH_CUDA)
  dtfft_config_t conf;
  dtfft_create_config(&conf);
  conf.platform = DTFFT_PLATFORM_CUDA;
  // We want to use managed memory here.
  // Disabling symmetric heap possibilities.
  conf.enable_nvshmem_backends = false;
  dtfft_set_config(conf);
#endif

  dtfft_pencil_t pencil;
  pencil.ndims = 3;
  if ( comm_rank == 0 ) {
    pencil.starts[0] = 0; pencil.starts[1] = 0; pencil.starts[2] = nx / 4;
    pencil.counts[0] = nz; pencil.counts[1] = ny; pencil.counts[2] = 3 * nx / 4;
  } else {
    pencil.starts[0] = 0; pencil.starts[1] = 0; pencil.starts[2] = 0;
    pencil.counts[0] = nz; pencil.counts[1] = ny; pencil.counts[2] = nx / 4;
  }

  // Create plan
  DTFFT_CALL( dtfft_create_plan_c2c_pencil(&pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor, &plan) )
  int8_t ndims = 0;
  const int32_t *dims;
  DTFFT_CALL( dtfft_get_dims(plan, &ndims, &dims) )
  if ( ndims != 3 || dims[0] != nz || dims[1] != ny || dims[2] != nx ) {
    fprintf(stderr, "Plan created with wrong dimensions: ndims = %d: %d, %d, %d.\n", ndims, dims[0], dims[1], dims[2]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  DTFFT_CALL( dtfft_report(plan) )
  DTFFT_CALL( dtfft_get_local_sizes(plan, in_starts, in_counts, NULL, out_counts, NULL) )
  for (int i = 0; i < 3; i++) {
    if ( in_starts[i] != pencil.starts[i] || in_counts[i] != pencil.counts[i] ) {
      fprintf(stderr, "Plan reported wrong decomposition.\n");
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
  size_t alloc_bytes;
  DTFFT_CALL( dtfft_get_alloc_bytes(plan, &alloc_bytes) )

  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];
  size_t out_size = out_counts[0] * out_counts[1] * out_counts[2];

  check = (dtfft_complex*) malloc(sizeof(dtfft_complex) * in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&in, alloc_bytes, cudaMemAttachGlobal) )
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&out, alloc_bytes, cudaMemAttachGlobal) )
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&aux, alloc_bytes, cudaMemAttachGlobal) )
  } else {
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&in) )
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&out) )
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&aux) )
  }
#else
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&in) )
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&out) )
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&aux) )
#endif

  setTestValuesComplexDouble(check, in_size);
#if defined(DTFFT_WITH_CUDA)
  complexDoubleH2D(check, in, in_size, (int32_t)platform);
#else
  complexDoubleH2D(check, in, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();

  if ( executor == DTFFT_EXECUTOR_NONE ) {
    bool is_z_slab;
    DTFFT_CALL( dtfft_get_z_slab_enabled(plan, &is_z_slab) )
    if ( is_z_slab ) {
      DTFFT_CALL( dtfft_transpose(plan, in, out, DTFFT_TRANSPOSE_X_TO_Z) )
    } else {
      DTFFT_CALL( dtfft_transpose(plan, in, aux, DTFFT_TRANSPOSE_X_TO_Y) )
      DTFFT_CALL( dtfft_transpose(plan, aux, out, DTFFT_TRANSPOSE_Y_TO_Z) )
    }
  } else {
    DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_EXECUTE_FORWARD, aux) )
  }
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  scaleComplexDouble((int32_t)executor, out, out_size, nx * ny * nz, (int32_t)platform, NULL);
#else
  scaleComplexDouble((int32_t)executor, out, out_size, nx * ny * nz);
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_EXECUTE_BACKWARD, aux) )
  #if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportComplexDouble(nx * ny * nz, tf, tb, in, in_size, check, (int32_t)platform);
#else
  checkAndReportComplexDouble(nx * ny * nz, tf, tb, in, in_size, check);
#endif

  free(check);

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaFree(in) )
    CUDA_SAFE_CALL( cudaFree(out) )
    CUDA_SAFE_CALL( cudaFree(aux) )
  } else {
    DTFFT_CALL( dtfft_mem_free(plan, in) )
    DTFFT_CALL( dtfft_mem_free(plan, out) )
    DTFFT_CALL( dtfft_mem_free(plan, aux) )
  }
#else
  DTFFT_CALL( dtfft_mem_free(plan, in) )
  DTFFT_CALL( dtfft_mem_free(plan, out) )
  DTFFT_CALL( dtfft_mem_free(plan, aux) )
#endif

  DTFFT_CALL( dtfft_destroy(&plan) )

  MPI_Finalize();
  return 0;
}