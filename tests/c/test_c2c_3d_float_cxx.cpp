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
#include "test_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int32_t nx = 6, ny = 22, nz = 59;

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: c2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
#ifdef DTFFT_WITH_CUDA
    cout << "This test is using C++ vectors, skipping it for GPU build" << endl;
#endif
  }
#ifdef DTFFT_WITH_CUDA
  MPI_Finalize();
  return 0;
#endif

  // Create plan
  // Nz is fastest varying index
  vector<int32_t> dims = {nz, ny, nx};

  MPI_Comm grid_comm;
  // It is assumed that data is not distributed in Nz direction
  // So 2d communicator is created here
  int comm_dims[2] = {0, 0};
  const int comm_periods[2] = {0, 0};
  MPI_Dims_create(comm_size, 2, comm_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, comm_dims, comm_periods, 1, &grid_comm);

  dtfft_executor_t executor_type;
#if defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined (DTFFT_WITH_MKL)
  executor_type = DTFFT_EXECUTOR_MKL;
#else
  executor_type = DTFFT_EXECUTOR_NONE;
#endif

  dtfft::PlanC2C plan(dims, grid_comm, DTFFT_SINGLE, DTFFT_MEASURE, executor_type);
  vector<int> in_counts(3);
  plan.get_local_sizes(NULL, in_counts.data());

  size_t in_size = std::accumulate(in_counts.begin(), in_counts.end(), 1, multiplies<int>());

  int64_t alloc_size;
  plan.get_alloc_size(&alloc_size);

  vector<complex<float>> in(alloc_size),
                          out(alloc_size),
                          aux(alloc_size),
                          check(alloc_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = complex<float> { (float)(i) / (float)(nx) / (float)(ny) / (float)(nz),
                            -(float)(i) / (float)(nx) / (float)(ny) / (float)(nz)};
    check[i] = in[i];
  }

  bool is_z_slab;
  plan.get_z_slab(&is_z_slab);
  double tf = 0.0 - MPI_Wtime();

  if ( executor_type == DTFFT_EXECUTOR_NONE ) {
    if ( is_z_slab ) {
      plan.transpose(in, out, DTFFT_TRANSPOSE_X_TO_Z);
    } else {
      plan.transpose(in, aux, DTFFT_TRANSPOSE_X_TO_Y);
      plan.transpose(aux, out, DTFFT_TRANSPOSE_Y_TO_Z);
    }
  } else {
    plan.execute(in, out, DTFFT_TRANSPOSE_OUT, aux);
  }

  tf += MPI_Wtime();

  std::fill(in.begin(), in.end(), complex<float>(-1., -1.));

  if ( executor_type != DTFFT_EXECUTOR_NONE ) {
    float scaler = 1. / (float) (nx * ny * nz);
    for ( auto & element: out) {
      element *= scaler;
    }
  }

  double tb = 0.0 - MPI_Wtime();
  plan.execute(out, in, DTFFT_TRANSPOSE_IN, aux);
  tb += MPI_Wtime();

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(complex<float>(in[i] - check[i]));
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, &nz, local_error, tf, tb);
  plan.destroy();

  MPI_Finalize();
  return 0;
}