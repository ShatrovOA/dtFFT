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
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int nx = 35, ny = 44;

  if(comm_rank == 0) {
    cout << "----------------------------------------" << endl;
    cout << "|   DTFFT test C++ interface: r2c_2d   |" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Nx = " << nx << ", Ny = " << ny           << endl;
    cout << "Number of processors: " << comm_size      << endl;
    cout << "----------------------------------------" << endl;
  }

#ifdef DTFFT_WITH_MKL
  int executor_type = DTFFT_EXECUTOR_MKL;
#elif defined(DTFFT_WITH_VKFFT)
  int executor_type = DTFFT_EXECUTOR_VKFFT;
#elif defined (DTFFT_WITH_FFTW)
  int executor_type = DTFFT_EXECUTOR_FFTW3;
#else
  if(comm_rank == 0) {
    cout << "No available executors found, skipping test..." << endl;
  }
  MPI_Finalize();
  return 0;

  int executor_type = DTFFT_EXECUTOR_NONE;
#endif

  // Create plan
  vector<int> dims = {ny, nx};
  dtfft::PlanR2C plan(dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_ESTIMATE, executor_type);

  vector<int> in_counts(2);
  size_t alloc_size;
  plan.get_alloc_size(&alloc_size);
  plan.get_local_sizes(NULL, in_counts.data());
  int in_size = in_counts[0] * in_counts[1];

  vector<double> in(alloc_size), check(in_size);
  vector<complex<double>> out(alloc_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    check[i] = in[i];
  }

  double tf = 0.0 - MPI_Wtime();
  plan.execute(in, out, DTFFT_TRANSPOSE_OUT);
  tf += MPI_Wtime();

  for ( auto & element: in) {
    element = -1.;
  }

  double scaler = 1. / (double) (nx * ny);
  for ( auto & element: out) {
    element *= scaler;
  }

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

  double local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    double error = abs(in[i] - check[i]);
    local_error = error > local_error ? error : local_error;
  }

  double global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-10) {
      cout << "Test 'r2c_2d_cxx' PASSED!" << endl;
    } else {
      cout << "Test 'r2c_2d_cxx' FAILED, error = " << global_error << endl;
      return -1;
    }
    cout << "----------------------------------------" << endl;
  }
  plan.destroy();

  MPI_Finalize();
  return 0;
}