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
#include <complex.h>
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
  int32_t nx = 512, ny = 64, nz = 96;
  double complex *in, *out, *temp, *check, *aux;
  int comm_rank, comm_size;
  int32_t in_counts[3], in_starts[3], out_counts[3];

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if ( comm_size != 2 && comm_size != 4 && comm_size !=8 ) {
    if ( comm_rank == 0 ) {
      printf("This test requires exactly 2, 4 or 8 MPI processes.\n");
    }
    MPI_Finalize();
    return 0;
  }

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| dtFFT test C interface: c2c_3d       |\n");
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

  dtfft_config_t conf;
  DTFFT_CALL( dtfft_create_config(&conf) )
  conf.backend = DTFFT_BACKEND_MPI_P2P_PIPELINED;
  conf.enable_z_slab = false;
  conf.enable_y_slab = false;
  conf.enable_fourier_reshape = true;
  conf.transpose_mode = DTFFT_TRANSPOSE_MODE_UNPACK;

#if defined(DTFFT_WITH_CUDA)
  conf.platform = DTFFT_PLATFORM_CUDA;
  // We want to use managed memory here.
  // Disabling symmetric heap possibilities.
  conf.enable_nvshmem_backends = false;
#endif

  DTFFT_CALL( dtfft_set_config(&conf) )

  executor = DTFFT_EXECUTOR_NONE;

  dtfft_pencil_t pencil;
  pencil.ndims = 3;
  
  // Calculate grid dimensions based on comm_size
  int grid_z = (comm_size == 8) ? 2 : 1;
  int grid_y = (comm_size >= 4) ? 2 : 1;
  int grid_x = comm_size / (grid_z * grid_y);
  
  // Calculate rank position in 3D grid
  int rank_z = comm_rank / (grid_y * grid_x);
  int rank_y = (comm_rank / grid_x) % grid_y;
  int rank_x = comm_rank % grid_x;
  
  // Set starts and counts based on rank position
  pencil.starts[0] = rank_z * (nz / grid_z);
  pencil.starts[1] = rank_y * (ny / grid_y);
  pencil.starts[2] = rank_x * (nx / grid_x);
  
  pencil.counts[0] = nz / grid_z;
  pencil.counts[1] = ny / grid_y;
  pencil.counts[2] = nx / grid_x;
  
  // Special handling for comm_size == 2 (uneven split)
  if (comm_size == 2) {
    if (comm_rank == 0) {
      pencil.starts[2] = nx / 4;
      pencil.counts[2] = 3 * nx / 4;
    } else {
      pencil.starts[2] = 0;
      pencil.counts[2] = nx / 4;
    }
  }
  if ( comm_size == 8 ) {
    executor = DTFFT_EXECUTOR_NONE;
  }

  // Create plan
  DTFFT_CALL( dtfft_create_plan_c2c_pencil(&pencil, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor, &plan) )
  int8_t ndims = 0;
  const int32_t *dims;
  DTFFT_CALL( dtfft_get_dims(plan, &ndims, &dims) )
  if ( ndims != 3 || dims[0] != nz || dims[1] != ny || dims[2] != nx ) {
    fprintf(stderr, "Plan created with wrong dimensions: ndims = %d: %dx%dx%d.\n", ndims, dims[0], dims[1], dims[2]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  const int *grid_dims;
  DTFFT_CALL( dtfft_get_grid_dims(plan, NULL, &grid_dims) )
  if ( comm_size == 2 ) {
    if ( grid_dims[0] != 1 || grid_dims[1] != 1 || grid_dims[2] != 2 ) {
      fprintf(stderr, "Plan created with wrong grid dimensions: %dx%dx%d.\n", grid_dims[0], grid_dims[1], grid_dims[2]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  } else if ( comm_size == 4 ) {
    if ( grid_dims[0] != 1 || grid_dims[1] != 2 || grid_dims[2] != 2 ) {
      fprintf(stderr, "Plan created with wrong grid dimensions: %dx%dx%d.\n", grid_dims[0], grid_dims[1], grid_dims[2]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  } else if ( comm_size == 8 ) {
    if ( grid_dims[0] != 1 || grid_dims[1] != 2 || grid_dims[2] != 4 ) {
      fprintf(stderr, "Plan created with wrong grid dimensions: %dx%dx%d.\n", grid_dims[0], grid_dims[1], grid_dims[2]);
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }

  DTFFT_CALL( dtfft_report(plan) )
  DTFFT_CALL( dtfft_get_local_sizes(plan, in_starts, in_counts, NULL, out_counts, NULL) )
  if ( comm_size != 8 ) {
    for (int i = 0; i < 3; i++) {
      if ( in_starts[i] != pencil.starts[i] || in_counts[i] != pencil.counts[i] ) {
        fprintf(stderr, "Plan reported wrong decomposition.\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
      }
    }
  }
  size_t alloc_bytes;
  DTFFT_CALL( dtfft_get_alloc_bytes(plan, &alloc_bytes) )
  size_t aux_bytes;
  DTFFT_CALL( dtfft_get_aux_bytes(plan, &aux_bytes) )
  size_t aux_bytes_transpose;
  DTFFT_CALL( dtfft_get_aux_bytes_transpose(plan, &aux_bytes_transpose) )
  if ( comm_size == 8) {
    size_t aux_bytes_reshape;
    DTFFT_CALL( dtfft_get_aux_bytes_reshape(plan, &aux_bytes_reshape) )
  }

  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];
  size_t out_size = out_counts[0] * out_counts[1] * out_counts[2];

  check = (double complex *) malloc(sizeof(double complex) * in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&in, alloc_bytes, cudaMemAttachGlobal) )
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&temp, alloc_bytes, cudaMemAttachGlobal) )
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&out, alloc_bytes, cudaMemAttachGlobal) )
    CUDA_SAFE_CALL( cudaMallocManaged((void**)&aux, aux_bytes, cudaMemAttachGlobal) )
  } else {
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&in) )
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&temp) )
    DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&out) )
    DTFFT_CALL( dtfft_mem_alloc(plan, aux_bytes, (void**)&aux) )
  }
#else
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&in) )
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&temp) )
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&out) )
  DTFFT_CALL( dtfft_mem_alloc(plan, aux_bytes, (void**)&aux) )
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
      DTFFT_CALL( dtfft_transpose(plan, in, out, DTFFT_TRANSPOSE_X_TO_Z, aux) )
    } else {
      if ( comm_size == 8 ) {
        DTFFT_CALL( dtfft_reshape(plan, in, out, DTFFT_RESHAPE_X_BRICKS_TO_PENCILS, aux) )
        DTFFT_CALL( dtfft_transpose(plan, out, temp, DTFFT_TRANSPOSE_X_TO_Y, aux) )
        DTFFT_CALL( dtfft_transpose(plan, temp, in, DTFFT_TRANSPOSE_Y_TO_Z, aux) )
        DTFFT_CALL( dtfft_reshape(plan, in, out, DTFFT_RESHAPE_Z_PENCILS_TO_BRICKS, aux) )
      } else {
        DTFFT_CALL( dtfft_transpose(plan, in, temp, DTFFT_TRANSPOSE_X_TO_Y, aux) )
        DTFFT_CALL( dtfft_transpose(plan, temp, out, DTFFT_TRANSPOSE_Y_TO_Z, aux) )
      }
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