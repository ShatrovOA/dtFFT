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

#include "test_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
#ifdef DTFFT_TRANSPOSE_ONLY
  return 0;
#else
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int nx = 19, ny = 32, nz = 55;

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: r2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
  }

  dtfft_executor_t executor_type;
#ifdef DTFFT_WITH_MKL
  executor_type = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_VKFFT)
  executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  executor_type = DTFFT_EXECUTOR_FFTW3;
#else // cuFFT
  executor_type = DTFFT_EXECUTOR_CUFFT;
#endif
  // Create plan
  vector<int32_t> dims = {nz, ny, nx};
  dtfft::PlanR2C plan(dims, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_MEASURE, executor_type);

  vector<int32_t> in_counts(3);
  plan.get_local_sizes(NULL, in_counts.data());
  size_t alloc_size;
  plan.get_alloc_size(&alloc_size);

  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];

  vector<float> in(alloc_size), check(in_size), aux(alloc_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = (float)(i) / (float)(nx) / (float)(ny) / (float)(nz);
    check[i] = in[i];
  }

  double tf = 0.0 - MPI_Wtime();
  plan.execute(in, in, DTFFT_TRANSPOSE_OUT, aux);
  tf += MPI_Wtime();

  float scaler = 1. / (float) (nx * ny * nz);
  for ( auto & element: in) {
    element *= scaler;
  }

  double tb = 0.0 - MPI_Wtime();
  plan.execute(in, in, DTFFT_TRANSPOSE_IN, aux);
  tb += MPI_Wtime();

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(in[i] - check[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, &nz, local_error, tf, tb);
  // Plan must be destroyed before calling MPI_Finalize
  plan.destroy();

  MPI_Finalize();
  return 0;
#endif
}