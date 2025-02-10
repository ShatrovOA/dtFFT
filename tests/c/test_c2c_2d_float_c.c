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
#ifdef DTFFT_WITH_CUDA
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

#ifdef DTFFT_WITH_FFTW
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined( DTFFT_WITH_CUFFT )
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_CUFFT;
#else
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_NONE;
#endif

  assign_device_to_process();

  dtfft_plan_t plan;
  DTFFT_CALL( dtfft_create_plan_c2c(2, n, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_PATIENT, executor_type, &plan) )

  int32_t in_counts[2], out_counts[2];
  size_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )
  size_t in_size = in_counts[0] * in_counts[1];
  size_t out_size = out_counts[0] * out_counts[1];

#ifdef DTFFT_WITH_CUDA
  cudaStream_t stream;
  DTFFT_CALL( dtfft_get_stream(plan, &stream) )
#endif

  dtfftf_complex *in, *out, *check;
  in = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * alloc_size);
  out = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * alloc_size);
  check = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * in_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = check[i] = (float)rand() / (float)(RAND_MAX) - (float)rand() / (float)(RAND_MAX) * I;
  }

#pragma acc enter data create(out[0:alloc_size - 1]) copyin(in[0:alloc_size - 1])

  double tf = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(in, out)
  DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, NULL) )
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
#endif
  tf += MPI_Wtime();


#pragma acc parallel loop present(in)
  for (size_t i = 0; i < alloc_size; i++) {
    in[i] = -1.0 + 1.0 * I;
  }

  if ( executor_type != DTFFT_EXECUTOR_NONE ) {
#pragma acc parallel loop present(out)
    for (size_t i = 0; i < out_size; i++) {
      out[i] /= (float) (nx * ny);
    }
  }

  double tb = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(in, out)
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, NULL) )
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) )
#endif
  tb += MPI_Wtime();

#pragma acc update self(in[0:in_size - 1])

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float real_error = fabs(crealf(check[i]) - crealf(in[i]));
    float cmplx_error = fabs(cimagf(check[i]) - cimagf(in[i]));
    float error = fmax(real_error, cmplx_error);
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, NULL, local_error, tf, tb);

#pragma acc exit data delete(in, out)

  dtfft_destroy(&plan);

  MPI_Finalize();
  return 0;
}