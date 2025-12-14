/*
  Copyright (c) 2021 - 2025, Oleg Shatrov
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
#include <cstring>
#include <cstdlib>

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
    cout << "|   dtFFT test C++ interface: r2r_2d   |" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Nx = " << nx << ", Ny = " << ny           << endl;
    cout << "Number of processors: " << comm_size      << endl;
    cout << "----------------------------------------" << endl;
  }

  Executor executor = Executor::NONE;
#ifdef DTFFT_WITH_FFTW
  executor = Executor::FFTW3;
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  if ( platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0 )
  {
    if(comm_rank == 0) {
      cout << "CUDA Platform detected.\n";
      cout << "This test is not designed to run on CUDA.\n";
    }
    MPI_Finalize();
    exit(0);
  }
#endif

  // Create plan
  vector<int32_t> dims = {ny, nx};

  // Create 2D MPI grid decomposition
  int grid_dims[2] = { 0, 0 };
  vector<int32_t> starts(2), counts(2);
  createGridDims(2, dims.data(), grid_dims, starts.data(), counts.data());
  
  auto pencil = Pencil(starts, counts);
  vector<R2RKind> kinds = {R2RKind::DCT_3, R2RKind::DCT_3};

  auto plan = PlanR2R(pencil, kinds, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, executor);
  DTFFT_CXX_CALL( plan.report() )
  size_t alloc_size;
  int32_t out_sizes[2];
  DTFFT_CXX_CALL( plan.get_local_sizes(nullptr, nullptr, nullptr, out_sizes, &alloc_size) )

  vector<double> in(alloc_size),
                 out(alloc_size),
                 check(alloc_size);

  size_t in_size = static_cast<size_t>(counts[0]) * static_cast<size_t>(counts[1]);
  size_t out_size = static_cast<size_t>(out_sizes[0]) * static_cast<size_t>(out_sizes[1]);

  // setTestValuesDouble(check.data(), in_size);
  for (size_t i = 0; i < in_size; i++) {
    in[i] = ((double)(i) / (double)(nx) / (double)(ny));
    check[i] = in[i];
    // in[i] = check[i];
  }

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(in.data(), out.data(), Execute::FORWARD) );
  tf += MPI_Wtime();

  if ( executor != Executor::NONE ) {
    for (size_t i = 0; i < out_size; i++) {
      out[i] /= (double) (4 * nx * ny);
    }
  }

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(out.data(), in.data(), Execute::BACKWARD) )
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportDouble(nx * ny, tf, tb, in.data(), in_size, check.data(), static_cast<int32_t>(Platform::HOST));
#else
  checkAndReportDouble(nx * ny, tf, tb, in.data(), in_size, check.data());
#endif

  DTFFT_CXX_CALL( plan.destroy() )

  MPI_Finalize();
  return 0;
}