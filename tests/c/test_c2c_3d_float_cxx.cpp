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

#include <dtfft.hpp>
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <complex>
#include <vector>
#include <numeric>

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int nx = 6, ny = 22, nz = 59;

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: c2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
  }

  // Create plan
  // Nz is fastest varying index
  vector<int> dims = {nz, ny, nx};

  MPI_Comm grid_comm;
  // It is assumed that data is not distributed in Nz direction
  // So 2d communicator is created here
  int comm_dims[2] = {0, 0};
  const int comm_periods[2] = {0, 0};
  MPI_Dims_create(comm_size, 2, comm_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, comm_dims, comm_periods, 1, &grid_comm);

#ifdef DTFFT_WITH_VKFFT
  int executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  dtfft::PlanC2C plan(dims, grid_comm, DTFFT_SINGLE, DTFFT_MEASURE, executor_type);
  vector<int> in_counts(3);
  plan.get_local_sizes(NULL, in_counts.data());

  size_t in_size = std::accumulate(in_counts.begin(), in_counts.end(), 1, multiplies<int>());

  size_t alloc_size;
  plan.get_alloc_size(&alloc_size);

  vector<complex<float>> in(alloc_size),
                          out(alloc_size),
                          check(alloc_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = complex<float> ( (float)(i) / (float)(nx) / (float)(ny) / (float)(nz),
                            -(float)(i) / (float)(nx) / (float)(ny) / (float)(nz));
    check[i] = in[i];
  }

  double tf = 0.0 - MPI_Wtime();
  plan.execute(in, out, DTFFT_TRANSPOSE_OUT);
  tf += MPI_Wtime();

  std::fill(in.begin(), in.end(), complex<float>(-1., -1.));
#ifndef DTFFT_TRANSPOSE_ONLY
  float scaler = 1. / (float) (nx * ny * nz);
  for ( auto & element: out) {
    element *= scaler;
  }
#endif
  double tb = 0.0 - MPI_Wtime();
  plan.execute(out, in, DTFFT_TRANSPOSE_IN);
  tb += MPI_Wtime();

  double t_sum;
  MPI_Allreduce(&tf, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tf = t_sum / (double) comm_size;
  MPI_Allreduce(&tb, &t_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  tb = t_sum / (double) comm_size;

  if(comm_rank == 0) {
    cout << "Forward execution time: " << tf << endl;
    cout << "Backward execution time: " << tb << endl;
    cout << "----------------------------------------" << endl;
  }

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(complex<float>(in[i] - check[i]));
    local_error = error > local_error ? error : local_error;
  }

  float global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-5) {
      cout << "Test 'c2c_3d_float_cxx' PASSED!" << endl;
    } else {
      cout << "Test 'c2c_3d_float_cxx' FAILED, error = " << global_error << endl;
      return -1;
    }
    cout << "----------------------------------------" << endl;
  }
  plan.destroy();

  MPI_Finalize();
  return 0;
}