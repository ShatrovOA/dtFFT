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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) 
{
  dtfft_plan plan;
  int nx = 64, ny = 128, nz = 256;
  dtfftf_complex *in, *out, *check, *work;
  float m_err, temp1, temp2, temp;
  int i,j,k, comm_rank, comm_size;
  int in_counts[3], out_counts[3];

  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if(comm_rank == 0) {
    printf("----------------------------------------\n");
    printf("| DTFFT test C interface: c2c_3d_float |\n");
    printf("----------------------------------------\n");
    printf("Nx = %d, Ny = %d, Nz = %d\n", nx, ny, nz);
    printf("Number of processors: %d\n", comm_size);
    printf("----------------------------------------\n");
  }

  // Create 1D communicator (slab decomposition)
  int ndims = 1;
  int dims[ndims]; dims[0] = 0;
  int periods[ndims]; periods[0] = 0;
  MPI_Dims_create(comm_size, ndims, dims);
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &grid_comm);

  // Create plan
  plan = dtfft_create_plan_f_c2c_3d(grid_comm, nz, ny, nx, DTFFT_ESTIMATE, DTFFT_EXECUTOR_FFTW3);

  dtfft_get_local_sizes(plan, NULL, in_counts, NULL, out_counts);

  in = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * in_counts[0] * in_counts[1] * in_counts[2]);
  out = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * out_counts[0] * out_counts[1] * out_counts[2]);
  check = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * in_counts[0] * in_counts[1] * in_counts[2]);

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i][0] = check[i][0] = 1.0;
    in[i][1] = check[i][1] = 3.0;
  }

  int work_size = dtfft_get_worker_size(plan, NULL, NULL);
  work = (dtfftf_complex*) malloc(sizeof(dtfftf_complex) * work_size);

  double tf = 0.0 - MPI_Wtime();
  dtfft_execute_f_c2c(plan, in, out, DTFFT_TRANSPOSE_OUT, work);
  tf += MPI_Wtime();

  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    in[i][0] = -1.0;
    in[i][1] = -2.0;
  }

  for (i = 0; i < out_counts[0] * out_counts[1] * out_counts[2]; i++) {
    out[i][0] /= (float) (nx * ny * nz);
    out[i][1] /= (float) (nx * ny * nz);
  }

  double tb = 0.0 - MPI_Wtime();
  dtfft_execute_f_c2c(plan, out, in, DTFFT_TRANSPOSE_IN, work);
  tb += MPI_Wtime();

  double t_sum;
  MPI_Allreduce(&tf, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tf = t_sum / (double) comm_size;
  MPI_Allreduce(&tb, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tb = t_sum / (double) comm_size;

  if(comm_rank == 0) {
    printf("Forward execution time: %f\n", tf);
    printf("Backward execution time: %f\n", tb);
    printf("----------------------------------------\n");
  }

  m_err = -1.0;
  for (i = 0; i < in_counts[0] * in_counts[1] * in_counts[2]; i++) {
    temp1 = fabs(check[i][0] - in[i][0]);
    temp2 = fabs(check[i][1] - in[i][1]);
    temp = fabs(temp1 + temp2);
    if (temp > m_err) m_err = temp;
  }

  MPI_Allreduce(&m_err, &m_err, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  
  if(comm_rank == 0) {
    if(m_err < 1e-3) {
      printf("Test 'c2c_3d_float_c' PASSED!\n");
    } else {
      printf("Test 'c2c_3d_float_c' FAILED, error = %f\n", m_err);
      return -1;
    }
    printf("----------------------------------------\n");
  }
  
  dtfft_destroy(plan);
  
  MPI_Finalize();
  return 0;
}