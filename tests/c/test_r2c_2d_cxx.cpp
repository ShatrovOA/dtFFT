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
#include <cstring>
#include <cstdlib>
#include <vector>
#include "test_utils.h"

using namespace std;
using namespace dtfft;

int main(int argc, char *argv[])
{
#if defined(DTFFT_TRANSPOSE_ONLY)
  cout << "FFT Support is disabled in this build, skipping test" << endl;
#else
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

#if defined(DTFFT_WITH_CUDA)
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  if ( platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0 )
  {
    cout << "This test is using C++ vectors, skipping it for GPU build" << endl;
    MPI_Finalize();
    return 0;
  }
#endif

  Executor executor = Executor::NONE;
#ifdef DTFFT_WITH_MKL
  executor = Executor::MKL;
#elif defined(DTFFT_WITH_FFTW )
  executor = Executor::FFTW3;
#endif

  if ( executor == Executor::NONE ) {
    if ( comm_rank == 0 ) cout << "Could not find valid R2C FFT executor, skipping test\n";
    MPI_Finalize();
    return 0;
  }

  // Create plan
  vector<int32_t> dims = {ny, nx};
  PlanR2C *plan;
  try {
    plan = new PlanR2C(dims, executor);
  } catch (const Exception& err) {
    cerr << err.what() << endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
    return -1;
  }

  vector<int32_t> in_counts(2);
  size_t alloc_size = plan->get_alloc_size();
  DTFFT_CXX_CALL( plan->get_local_sizes(nullptr, in_counts.data()) );
  size_t in_size = in_counts[0] * in_counts[1];

  vector<double> in(alloc_size), check(in_size);
  vector<complex<double>> out(alloc_size / 2);

  setTestValuesDouble(check.data(), in_size);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = check[i];
  }

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan->execute(in.data(), out.data(), Execute::FORWARD) );
  tf += MPI_Wtime();

  for ( auto & element: in) {
    element = -1.;
  }

  double scaler = 1. / (double) (nx * ny);
  for ( auto & element: out) {
    element *= scaler;
  }

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan->execute(out.data(), in.data(), Execute::BACKWARD) );
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportDouble(nx * ny, tf, tb, in.data(), in_size, check.data(), static_cast<int32_t>(Platform::HOST));
#else
  checkAndReportDouble(nx * ny, tf, tb, in.data(), in_size, check.data());
#endif

  DTFFT_CXX_CALL( plan->destroy() );

  delete plan;

  MPI_Finalize();
  return 0;
#endif
}