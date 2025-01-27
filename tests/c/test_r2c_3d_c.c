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
#include <math.h>
#include "test_utils.h"

int main(int argc, char *argv[])
{
#ifdef DTFFT_TRANSPOSE_ONLY
  return 0;
#else
  dtfft_plan_t plan;
  int32_t nx = 16, ny = 32, nz = 70;
  double *in, *check;
  dtfft_complex *out, *aux;
  int comm_rank, comm_size;
  int32_t in_counts[3], out_counts[3], n[3] = {nz, ny, nx};
#ifdef DTFFT_WITH_CUDA
  double *d_in, *d_out;
#endif

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

  dtfft_executor_t executor_type;
#ifdef DTFFT_WITH_MKL
  executor_type = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_VKFFT)
  executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined (DTFFT_WITH_CUFFT)
  executor_type = DTFFT_EXECUTOR_CUFFT;
#endif

  assign_device_to_process();
  // Create plan
  DTFFT_CALL( dtfft_create_plan_r2c(3, n, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor_type, &plan) )

  // Get local sizes
  size_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];
  size_t out_size = out_counts[0] * out_counts[1] * out_counts[2];

  // Allocate buffers
  check = (double*) malloc(sizeof(double) * in_size);
  in = (double*) malloc(sizeof(double) * alloc_size);
  out = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size / 2);
  // `alloc_size / 2` is used here, since `dtfft_get_local_sizes` for R2C plan returns number of elements for real values
  // and `out` is defined as complex buffer
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_in,  sizeof(double) * alloc_size) )
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_out, sizeof(dtfft_complex) * alloc_size / 2) )
  CUDA_SAFE_CALL( cudaMalloc((void**)&aux, sizeof(dtfft_complex) * alloc_size / 2) )
#else
  aux = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size / 2);
#endif

  for (size_t i = 0; i < in_size; i++) {
    in[i] = check[i] = (double)(i) / (double)(in_size);
  }
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaMemcpy( d_in, in, sizeof(double) * in_size, cudaMemcpyHostToDevice) )
  cudaStream_t stream;
  DTFFT_CALL( dtfft_get_stream(plan, &stream) )
#endif
  // Forward transpose
  double tf = 0.0 - MPI_Wtime();
#ifdef DTFFT_WITH_CUDA
  DTFFT_CALL( dtfft_execute(plan, d_in, d_out, DTFFT_TRANSPOSE_OUT, aux) )
  CUDA_SAFE_CALL( cudaMemcpyAsync(out, d_out, sizeof(dtfft_complex) * out_size, cudaMemcpyDeviceToHost, stream) )
  CUDA_SAFE_CALL( cudaMemcpyAsync(in, d_in, sizeof(double) * in_size, cudaMemcpyDeviceToHost, stream) )
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
#else
  DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux) );
#endif
  tf += MPI_Wtime();

  // Clean input buffer for possible error check
  for (size_t i = 0; i < in_size; i++) {
    in[i] = -1.0;
  }

  // Normalize
  for (size_t i = 0; i < out_size; i++) {
    out[i] /= (double) (nx * ny * nz);
  }

  // Backward transpose
  double tb = 0.0 - MPI_Wtime();
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaMemcpyAsync(d_out, out, sizeof(dtfft_complex) * out_size, cudaMemcpyHostToDevice, stream) )
  CUDA_SAFE_CALL( cudaMemcpyAsync(d_in, in, sizeof(double) * in_size, cudaMemcpyHostToDevice, stream) )
  DTFFT_CALL( dtfft_execute(plan, d_out, d_in, DTFFT_TRANSPOSE_IN, aux) );
  CUDA_SAFE_CALL( cudaMemcpyAsync(in, d_in, sizeof(double) * in_size, cudaMemcpyDeviceToHost, stream) )
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
#else
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, aux) );
#endif
  tb += MPI_Wtime();

  // Check error
  double local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    double error = fabs(check[i] - in[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_double(&nx, &ny, &nz, local_error, tf, tb);

  // Deallocate buffers
  free(in);
  free(out);
  free(check);
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaFree(d_in) )
  CUDA_SAFE_CALL( cudaFree(d_out) )
  CUDA_SAFE_CALL( cudaFree(aux) )
#else
  free(aux);
#endif

  // Destroy plan
  DTFFT_CALL( dtfft_destroy(&plan) );

  MPI_Finalize();
  return 0;
#endif
}