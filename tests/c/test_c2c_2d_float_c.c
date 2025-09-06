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
#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  int32_t nx = 3333, ny = 4444;
#else
  int32_t nx = 11, ny = 39;
#endif

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_2d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d\n", nx, ny);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  // Create plan
  int32_t n[2] = {ny, nx};

  dtfft_executor_t executor = DTFFT_EXECUTOR_NONE;
#ifdef DTFFT_WITH_FFTW
  executor = DTFFT_EXECUTOR_FFTW3;
#elif defined(DTFFT_WITH_MKL)
  executor = DTFFT_EXECUTOR_MKL;
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = getenv("DTFFT_PLATFORM");

  if ( platform_env == NULL || strcmp(platform_env, "cuda") == 0 )
  {
# if defined( DTFFT_WITH_CUFFT )
    executor = DTFFT_EXECUTOR_CUFFT;
# elif defined( DTFFT_WITH_VKFFT )
    executor = DTFFT_EXECUTOR_VKFFT;
# else
    executor = DTFFT_EXECUTOR_NONE;
# endif
  }
#endif

#if defined(DTFFT_WITH_CUDA)
  dtfft_config_t conf;
  DTFFT_CALL( dtfft_create_config(&conf) )
  conf.platform = DTFFT_PLATFORM_CUDA;
  DTFFT_CALL( dtfft_set_config(&conf) )
#endif

  attach_gpu_to_process();

  dtfft_plan_t plan;
  DTFFT_CALL( dtfft_create_plan_c2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_PATIENT, executor, &plan) )

  int32_t in_counts[2], out_counts[2];
  size_t alloc_size, element_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  DTFFT_CALL( dtfft_get_element_size(plan, &element_size) )
  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];

  if ( element_size != (size_t)sizeof(dtfftf_complex) ) {
    fprintf(stderr, "element_size /= sizeof(dtfftf_complex)\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )

  cudaStream_t stream;

  if ( platform == DTFFT_PLATFORM_CUDA ) {
    DTFFT_CALL( dtfft_get_stream(plan, (dtfft_stream_t*)&stream) )
  }
#endif

  dtfftf_complex *in, *out, *check;

  dtfft_mem_alloc(plan, element_size * alloc_size, (void**)&in);
  dtfft_mem_alloc(plan, element_size * alloc_size, (void**)&out);
  check = (dtfftf_complex*) malloc(element_size * in_size);
  setTestValuesComplexFloat(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  complexFloatH2D(check, in, in_size, (int32_t)platform);
#else
  complexFloatH2D(check, in, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_EXECUTE_FORWARD, NULL) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_HOST ) {
    for (size_t i = 0; i < alloc_size; i++) {
      in[i] = -1.0 + 1.0 * I;
    }
  }
#else
  for (size_t i = 0; i < alloc_size; i++) {
    in[i] = -1.0 + 1.0 * I;
  }
#endif


#if defined(DTFFT_WITH_CUDA)
  scaleComplexFloat((int32_t)executor, out, out_size, nx * ny, (int32_t)platform, stream);
#else
  scaleComplexFloat((int32_t)executor, out, out_size, nx * ny);
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_EXECUTE_BACKWARD, NULL) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
  }
#endif
  tb += MPI_Wtime();


#if defined(DTFFT_WITH_CUDA)
  checkAndReportComplexFloat(nx * ny, tf, tb, in, in_size, check, (int32_t)platform);
#else
  checkAndReportComplexFloat(nx * ny, tf, tb, in, in_size, check);
#endif


  // Free memory before plan destruction
  dtfft_mem_free(plan, in);
  dtfft_mem_free(plan, out);
  free(check);

  DTFFT_CALL( dtfft_destroy(&plan) )

  MPI_Finalize();
  return 0;
}