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


  int nx = 32, ny = 64, nz = 128;

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: r2r_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
  }

#ifndef DTFFT_WITHOUT_FFTW
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined(DTFFT_WITH_VKFFT)
  int executor_type = DTFFT_EXECUTOR_VKFFT;
#else
  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  const int ndims = 3;
  const int dims[] = {nz, ny, nx};
  const int kinds[] = {DTFFT_DCT_2, DTFFT_DCT_2, DTFFT_DCT_2};
  dtfft::PlanR2R plan(ndims, dims, kinds, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_ESTIMATE, executor_type);

  int in_sizes[ndims];
  int out_sizes[ndims];
  size_t alloc_size;

  plan.get_local_sizes(NULL, in_sizes, NULL, out_sizes, &alloc_size);

  size_t in_size = std::accumulate(in_sizes, in_sizes + 3, 1, multiplies<int>());
  size_t out_size = std::accumulate(out_sizes, out_sizes + 3, 1, multiplies<int>());
  float *inout = new float[alloc_size];
  float *check = new float[alloc_size];
  float *aux = new float[alloc_size];

  for (size_t i = 0; i < in_size; i++)
  {
    inout[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    check[i] = inout[i];
  }

  plan.execute(inout, inout, DTFFT_TRANSPOSE_OUT, aux);

#ifndef DTFFT_TRANSPOSE_ONLY
  for (size_t i = 0; i < out_size; i++)
  {
    inout[i] /= (float) (8 * nx * ny * nz);
  }
#endif

  plan.execute(inout, inout, DTFFT_TRANSPOSE_IN, aux);

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(inout[i] - check[i]);
    local_error = error > local_error ? error : local_error;
  }

  float global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-5) {
      cout << "Test 'r2r_3d_float_cxx' PASSED!" << endl;
    } else {
      cout << "Test 'r2r_3d_float_cxx' FAILED, error = " << global_error << endl;
      return -1;
    }
    cout << "----------------------------------------" << endl;
  }

  plan.destroy();

  delete inout;
  delete check;

  MPI_Finalize();
}