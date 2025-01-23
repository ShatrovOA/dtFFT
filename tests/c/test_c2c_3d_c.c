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
  size_t i;
  int32_t in_counts[3], out_counts[3], n[3] = {nz, ny, nx};
  int grid_dims[3] = {1, 0, 0}, periods[3] = {0, 0, 0};
  MPI_Comm grid_comm;

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_3d       |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  dtfft_executor_t executor_type;
#ifdef DTFFT_WITH_MKL
  executor_type = DTFFT_EXECUTOR_MKL;
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined (DTFFT_WITH_VKFFT)
  executor_type = DTFFT_EXECUTOR_VKFFT;
#else
  executor_type = DTFFT_EXECUTOR_NONE;
#endif

  assign_device_to_process();

  MPI_Dims_create(comm_size, 3, grid_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 3, grid_dims, periods, 0, &grid_comm);
  // Create plan
  DTFFT_CALL( dtfft_create_plan_c2c(3, n, grid_comm, DTFFT_DOUBLE, DTFFT_PATIENT, executor_type, &plan) )

  int64_t alloc_size;
  DTFFT_CALL( dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts, &alloc_size) )

  size_t in_size, out_size;
  in_size = in_counts[0] * in_counts[1] * in_counts[2];
  out_size = out_counts[0] * out_counts[1] * out_counts[2];

  check = (dtfft_complex*) malloc(sizeof(dtfft_complex) * in_size);

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaMallocManaged((void**)&in, sizeof(dtfft_complex) * alloc_size, cudaMemAttachGlobal) )
  CUDA_SAFE_CALL( cudaMallocManaged((void**)&out, sizeof(dtfft_complex) * alloc_size, cudaMemAttachGlobal) )
  CUDA_SAFE_CALL( cudaMallocManaged((void**)&aux, sizeof(dtfft_complex) * alloc_size, cudaMemAttachGlobal) )
#else
  in = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size);
  out = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size);
  aux = (dtfft_complex*) malloc(sizeof(dtfft_complex) * alloc_size);
#endif

  for (i = 0; i < in_size; i++) {
    in[i][0] = check[i][0] = 1.0;
    in[i][1] = check[i][1] = 3.0;
  }

  double tf = 0.0 - MPI_Wtime();

  if ( executor_type == DTFFT_EXECUTOR_NONE ) {
    bool is_z_slab;
    DTFFT_CALL( dtfft_get_z_slab(plan, &is_z_slab) )
    if ( is_z_slab ) {
      DTFFT_CALL( dtfft_transpose(plan, in, out, DTFFT_TRANSPOSE_X_TO_Z) )
    } else {
      DTFFT_CALL( dtfft_transpose(plan, in, aux, DTFFT_TRANSPOSE_X_TO_Y) )
      DTFFT_CALL( dtfft_transpose(plan, aux, out, DTFFT_TRANSPOSE_Y_TO_Z) )
    }
  } else {
    DTFFT_CALL( dtfft_execute(plan, in, out, DTFFT_TRANSPOSE_OUT, aux) )
  }
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tf += MPI_Wtime();

  for (i = 0; i < in_size; i++) {
    in[i][0] = -1.0;
    in[i][1] = -2.0;
  }

  if ( executor_type != DTFFT_EXECUTOR_NONE ) {
    for (i = 0; i < out_size; i++) {
      out[i][0] /= (double) (nx * ny * nz);
      out[i][1] /= (double) (nx * ny * nz);
    }
  }
  double tb = 0.0 - MPI_Wtime();
  DTFFT_CALL( dtfft_execute(plan, out, in, DTFFT_TRANSPOSE_IN, aux) )
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tb += MPI_Wtime();

  double local_error = -1.0;
  for (i = 0; i < in_size; i++) {
    double real_error = fabs(check[i][0] - in[i][0]);
    double cmplx_error = fabs(check[i][1] - in[i][1]);
    double error = fmax(real_error, cmplx_error);
    local_error = error > local_error ? error : local_error;
  }

  report_double(&nx, &ny, &nz, local_error, tf, tb);

  dtfft_destroy(&plan);

  free(check);

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaFree(in) )
  CUDA_SAFE_CALL( cudaFree(out) )
  CUDA_SAFE_CALL( cudaFree(aux) )
#else
  free(in);
  free(out);
  free(aux);
#endif

  MPI_Finalize();
  return 0;
}