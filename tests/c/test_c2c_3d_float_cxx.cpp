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
#include <numeric>
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

  int32_t nx = 6, ny = 22, nz = 59;

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|dtFFT test C++ interface: c2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
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

  // Create plan
  // Nz is fastest varying index
  vector<int32_t> dims = {nz, ny, nx};

  MPI_Comm grid_comm;
  // It is assumed that data is not distributed in Nz direction
  // So 2D communicator is created here
  int comm_dims[2] = {0, 0};
  const int comm_periods[2] = {0, 0};
  MPI_Dims_create(comm_size, 2, comm_dims);
  MPI_Cart_create(MPI_COMM_WORLD, 2, comm_dims, comm_periods, 1, &grid_comm);

  Executor executor = Executor::NONE;
#if defined (DTFFT_WITH_FFTW)
  executor = Executor::FFTW3;
#elif defined (DTFFT_WITH_MKL)
  executor = Executor::MKL;
#endif

  Config conf;
  conf.set_enable_z_slab(false)
    .set_enable_y_slab(true)
    .set_backend(Backend::MPI_P2P);
  DTFFT_CXX_CALL( set_config(conf) );

  auto plan = new PlanC2C(dims, grid_comm, Precision::SINGLE, Effort::MEASURE, executor);
  DTFFT_CXX_CALL( plan->report() )
  vector<int32_t> in_counts(3), out_counts(3);
  DTFFT_CXX_CALL( plan->get_local_sizes(nullptr, in_counts.data(), nullptr, out_counts.data()) )

  MPI_Comm_free(&grid_comm);

#if defined(DTFFT_WITH_CUDA)
  const Platform platform = plan->get_platform();
  if ( platform == Platform::CUDA ) {
    if ( comm_rank == 0 ) cerr << "Detected CUDA Platform\n";
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
#endif

  size_t in_size = std::accumulate(in_counts.begin(), in_counts.end(), 1, multiplies<int>());
  size_t out_size = std::accumulate(out_counts.begin(), out_counts.end(), 1, multiplies<int>());

  size_t alloc_size = plan->get_alloc_size();
  size_t aux_size = plan->get_aux_size();

  vector<complex<float>> in(alloc_size),
                          out(alloc_size),
                          aux(aux_size),
                          check(alloc_size);

  setTestValuesComplexFloat(check.data(), in_size);
  for (size_t i = 0; i < in_size; i++) {
    in[i] = check[i];
  }

  bool is_z_slab = plan->get_z_slab_enabled();
  bool is_y_slab = plan->get_y_slab_enabled();
  double tf = 0.0 - MPI_Wtime();

  if ( executor == Executor::NONE ) {
    if ( is_z_slab ) {
      DTFFT_CXX_CALL( plan->transpose(in.data(), out.data(), Transpose::X_TO_Z) )
    } else if ( is_y_slab ) {
      DTFFT_CXX_CALL( plan->transpose(in.data(), out.data(), Transpose::X_TO_Y) )
    } else {
      DTFFT_CXX_CALL( plan->transpose(in.data(), aux.data(), Transpose::X_TO_Y) )
      DTFFT_CXX_CALL( plan->transpose(aux.data(), out.data(), Transpose::Y_TO_Z) )
    }
  } else {
    DTFFT_CXX_CALL( plan->forward(in.data(), out.data(), aux.data()) )
  }

  tf += MPI_Wtime();

  std::fill(in.begin(), in.end(), complex<float>{-1., -1.});

#if defined(DTFFT_WITH_CUDA)
  scaleComplexFloat(static_cast<int32_t>(executor), out.data(), out_size, nx * ny * nz, static_cast<int32_t>(platform), NULL);
#else
  scaleComplexFloat(static_cast<int32_t>(executor), out.data(), out_size, nx * ny * nz);
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan->backward(out.data(), in.data(), aux.data()) )
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportComplexFloat(nx * ny * nz, tf, tb, in.data(), in_size, check.data(), static_cast<int32_t>(platform));
#else
  checkAndReportComplexFloat(nx * ny * nz, tf, tb, in.data(), in_size, check.data());
#endif

  DTFFT_CXX_CALL( plan->destroy() )

  delete plan;
  MPI_Finalize();
  return 0;
}