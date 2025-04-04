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
#if defined(DTFFT_WITH_CUDA)
    cout << "This test is using C++ vectors, skipping it for GPU build" << endl;
#endif
  }

#if defined(DTFFT_WITH_CUDA)
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  if ( platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0 )
  {
      MPI_Finalize();
      return 0;
  }
#endif

  Executor executor;
#ifdef DTFFT_WITH_MKL
  executor = Executor::MKL;
#elif defined(DTFFT_WITH_FFTW )
  executor = Executor::FFTW3;
#else
# if !defined(DTFFT_WITH_CUDA)
  if(comm_rank == 0) {
    cout << "Missing HOST FFT Executor\n";
  }
  MPI_Finalize();
  return 0;
# endif
#endif

  // Create plan
  vector<int32_t> dims = {ny, nx};
  dtfft::PlanR2C *plan;
  try {
    plan = new dtfft::PlanR2C(dims, executor);
  } catch (const dtfft::Exception& err) {
    cerr << err.what() << endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
    return -1;
  }

  vector<int32_t> in_counts(2);
  size_t alloc_size;
  DTFFT_CXX_CALL( plan->get_alloc_size(&alloc_size) );
  DTFFT_CXX_CALL( plan->get_local_sizes(nullptr, in_counts.data()) );
  size_t in_size = in_counts[0] * in_counts[1];

  vector<double> in(alloc_size), check(in_size);
  vector<complex<double>> out(alloc_size / 2);

  for (size_t i = 0; i < in_size; i++) {
    in[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    check[i] = in[i];
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

  double local_error = checkDouble(check.data(), in.data(), in_size);
  reportDouble(&tf, &tb, &local_error, &nx, &ny, nullptr );

  DTFFT_CXX_CALL( plan->destroy() );

  delete plan;

  MPI_Finalize();
  return 0;
#endif
}