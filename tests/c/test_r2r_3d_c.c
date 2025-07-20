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

#include "test_utils.h"

int main(int argc, char *argv[]) 
{
  int32_t nx = 4, ny = 64, nz = 16;
  double *inout, *check, *aux;
  int comm_rank, comm_size;
  int32_t n[3] = {nz, ny, nx};
  dtfft_pencil_t pencils[3];


  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: r2r_3d       |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  attach_gpu_to_process();

  // Create plan
  dtfft_plan_t plan;

  DTFFT_CALL( dtfft_create_plan_r2r(3, n, NULL, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, DTFFT_EXECUTOR_NONE, &plan) )

  size_t alloc_bytes;
  DTFFT_CALL( dtfft_get_alloc_bytes(plan, &alloc_bytes) )

  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&inout) )
  DTFFT_CALL( dtfft_mem_alloc(plan, alloc_bytes, (void**)&aux) )

  // Obtain pencil information (optional)
  for ( int i = 0; i < 3; i++ ) {
    dtfft_get_pencil(plan, i + 1, &pencils[i]);
  }

  size_t in_size = pencils[0].size;

  check = (double*) malloc(in_size * sizeof(double));
  setTestValuesDouble(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  dtfft_platform_t platform;
  DTFFT_CALL( dtfft_get_platform(plan, &platform) )

  doubleH2D(check, inout, in_size, (int32_t)platform);
#else
  doubleH2D(check, inout, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  /*
    Run custom Forward FFT X direction using pencils[0] information
  */
  DTFFT_CALL( dtfft_transpose(plan, inout, aux, DTFFT_TRANSPOSE_X_TO_Y) )
  /*
    Run custom Forward FFT Y direction using pencils[1] information
  */
  DTFFT_CALL( dtfft_transpose(plan, aux, inout, DTFFT_TRANSPOSE_Y_TO_Z) )
  /*
    Run custom Forward FFT Z direction using pencils[2] information
  */
#if defined(DTFFT_WITH_CUDA)
  if ( platform == DTFFT_PLATFORM_CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();


  double tb = 0.0 - MPI_Wtime();
  /*
    Run custom Backward FFT Z direction using pencils[2] information
  */
  DTFFT_CALL( dtfft_transpose(plan, inout, aux, DTFFT_TRANSPOSE_Z_TO_Y) )
  /*
    Run custom Backward FFT Y direction using pencils[1] information
  */
  DTFFT_CALL( dtfft_transpose(plan, aux, inout, DTFFT_TRANSPOSE_Y_TO_X) )
  /*
    Run custom Backward FFT X direction using pencils[0] information
  */
#if defined(DTFFT_WITH_CUDA)
 if ( platform == DTFFT_PLATFORM_CUDA ) {
   CUDA_SAFE_CALL( cudaDeviceSynchronize() )
 }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportDouble(nx * ny, tf, tb, inout, in_size, check, (int32_t)platform);
#else
  checkAndReportDouble(nx * ny, tf, tb, inout, in_size, check);
#endif

  DTFFT_CALL( dtfft_mem_free(plan, inout) )
  DTFFT_CALL( dtfft_mem_free(plan, aux) )

  DTFFT_CALL( dtfft_destroy(&plan) )
  free(check);

  MPI_Finalize();
  return 0;
}