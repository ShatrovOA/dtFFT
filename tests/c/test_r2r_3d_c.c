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
  int32_t in_counts[3], out_counts[3], n[3] = {nz, ny, nx};
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

  dtfft_executor_t executor = DTFFT_EXECUTOR_NONE;

  // Create plan
  dtfft_plan_t plan;

  DTFFT_CALL( dtfft_create_plan_r2r(3, n, NULL, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, executor, &plan) )

  size_t alloc_size;
  DTFFT_CALL( dtfft_get_alloc_size(plan, &alloc_size) )

  DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(double) * alloc_size, (void**)&inout) )
  DTFFT_CALL( dtfft_mem_alloc(plan, sizeof(double) * alloc_size, (void**)&aux) )

  check = (double*) malloc(sizeof(double) * alloc_size);

  // Obtain pencil information (optional)
  for ( int i = 0; i < 3; i++ ) {
    dtfft_get_pencil(plan, i + 1, &pencils[i]);
  }

  memcpy(in_counts, pencils[0].counts, 3 * sizeof(int32_t));
  memcpy(out_counts, pencils[2].counts, 3 * sizeof(int32_t));

  int64_t in_size = in_counts[0] * in_counts[1] * in_counts[2];
  int64_t out_size = out_counts[0] * out_counts[1] * out_counts[2];

  for (int i = 0; i < in_size; i++)
    inout[i] = check[i] = (double)(i) / (double)(in_size);

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
  tf += MPI_Wtime();

  if ( executor != DTFFT_EXECUTOR_NONE ) {
    // Perform scaling. Scaling value may very depending on FFT type
    for (int i = 0; i < out_size; i++)
      inout[i] /= (double) (8 * nx * (ny - 1) * (nz - 1));
  }

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
  tb += MPI_Wtime();

  double local_error = -1.0;
  for (int i = 0; i < in_size; i++) {
    double error = fabs(check[i] - inout[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_double(&nx, &ny, &nz, local_error, tf, tb);

  DTFFT_CALL( dtfft_mem_free(plan, inout) )
  DTFFT_CALL( dtfft_mem_free(plan, aux) )

  DTFFT_CALL( dtfft_destroy(&plan) )
  free(check);

  MPI_Finalize();
  return 0;
}