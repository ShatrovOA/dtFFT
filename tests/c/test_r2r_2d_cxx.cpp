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
using namespace dtfft;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int32_t nx = 19, ny = 44;

  if(comm_rank == 0) {
    cout << "----------------------------------------" << endl;
    cout << "|   DTFFT test C++ interface: r2r_2d   |" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Nx = " << nx << ", Ny = " << ny           << endl;
    cout << "Number of processors: " << comm_size      << endl;
    cout << "----------------------------------------" << endl;
  }

// #ifdef DTFFT_WITH_FFTW
//   dtfft_executor_t executor = DTFFT_EXECUTOR_FFTW3;
// #elif defined(DTFFT_WITH_VKFFT)
//   dtfft_executor_t executor = DTFFT_EXECUTOR_VKFFT;
// #else
  Executor executor = Executor::NONE;
// #endif

  // Create plan
  vector<int32_t> dims = {ny, nx};
  vector<dtfft::R2RKind> kinds = {};
  dtfft::PlanR2R plan(dims, kinds, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, executor);
  size_t alloc_size;
  DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) )

  vector<double> in(alloc_size),
                 out(alloc_size),
                 check(alloc_size);

  for (size_t i = 0; i < alloc_size; i++) {
    in[i] = ((double)(i) / (double)(nx) / (double)(ny));
    check[i] = in[i];
  }

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(in.data(), out.data(), dtfft::ExecuteType::FORWARD) );
  tf += MPI_Wtime();

  Pencil out_pencil;
  DTFFT_CXX_CALL( plan.get_pencil(2, out_pencil) )
  size_t out_size = out_pencil.get_size();

  if ( executor != Executor::NONE ) {
    for (size_t i = 0; i < out_size; i++) {
      out[i] /= (double) (4 * nx * ny);
    }
  }

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(out.data(), in.data(), dtfft::ExecuteType::BACKWARD) )
  tb += MPI_Wtime();

  double local_error = -1.0;
  for (size_t i = 0; i < alloc_size; i++) {
    double error = fabs(in[i] - check[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_double(&nx, &ny, nullptr, local_error, tf, tb);

  DTFFT_CXX_CALL( plan.destroy() )

  MPI_Finalize();
  return 0;
}