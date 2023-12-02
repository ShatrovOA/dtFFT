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

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int nx = 1024, ny = 512;

  if(comm_rank == 0) {
    cout << "----------------------------------------" << endl;
    cout << "|   DTFFT test C++ interface: r2r_2d   |" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Nx = " << nx << ", Ny = " << ny           << endl;
    cout << "Number of processors: " << comm_size      << endl;
    cout << "----------------------------------------" << endl;
  }

  // Create plan
  vector<int> dims = {ny, nx};
  vector<int> f_kinds = {DTFFT_DCT_2, DTFFT_DST_2};
  vector<int> b_kinds = {DTFFT_DCT_3, DTFFT_DST_3};
  dtfft::PlanR2R plan(dims, f_kinds, b_kinds);
  size_t alloc_size = plan.get_alloc_size();

  vector<double> in, out, check;

  in.reserve(alloc_size);
  out.reserve(alloc_size);
  check.reserve(alloc_size);

  for (size_t i = 0; i < alloc_size; i++) {
    in.push_back( (double)(i) / (double)(nx) / (double)(ny));
    check.push_back(in.at(i));
    out.push_back(0.0);
  }

  double tf = 0.0 - MPI_Wtime();
  plan.execute(in, out, DTFFT_TRANSPOSE_OUT);
  tf += MPI_Wtime();

  for (size_t i = 0; i < alloc_size; i++) {
    out.at(i) /= (double) (4 * nx * ny);
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
  for (size_t i = 0; i < alloc_size; i++) {
    double error = fabs(in.at(i) - check.at(i));
    local_error = error > local_error ? error : local_error;
  }

  double global_error;
  MPI_Allreduce(&local_error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if(comm_rank == 0) {
    if(global_error < 1e-10) {
      cout << "Test 'r2r_2d_cxx' PASSED!" << endl;
    } else {
      cout << "Test 'r2r_2d_cxx' FAILED, error = " << global_error << endl;
      return -1;
    }
    cout << "----------------------------------------" << endl;
  }
  plan.destroy();

  MPI_Finalize();
  return 0;
}