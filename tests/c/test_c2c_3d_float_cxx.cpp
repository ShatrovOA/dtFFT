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
    cout << "|DTFFT test C++ interface: c2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
    cout << "This test is using C++ vectors, skipping it for GPU build" << endl;
#endif
  }
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  MPI_Finalize();
  return 0;
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

  Executor executor;
#if defined (DTFFT_WITH_FFTW)
  executor = Executor::FFTW3;
#elif defined (DTFFT_WITH_MKL)
  executor = Executor::MKL;
#else
  executor = Executor::NONE;
#endif

  Config conf;
  conf.set_enable_z_slab(true);
  DTFFT_CXX_CALL( set_config(conf) );

  Plan *plan = new PlanC2C(dims, grid_comm, Precision::SINGLE, Effort::MEASURE, executor);
  vector<int> in_counts(3);
  DTFFT_CXX_CALL( plan->get_local_sizes(nullptr, in_counts.data()) )

  size_t in_size = std::accumulate(in_counts.begin(), in_counts.end(), 1, multiplies<int>());

  size_t alloc_size;
  plan->get_alloc_size(&alloc_size);

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
  DTFFT_CXX_CALL( plan->get_z_slab_enabled(&is_z_slab) )
  double tf = 0.0 - MPI_Wtime();

  if ( executor == Executor::NONE ) {
    if ( is_z_slab ) {
      DTFFT_CXX_CALL( plan->transpose(in.data(), out.data(), TransposeType::X_TO_Z) )
    } else {
      DTFFT_CXX_CALL( plan->transpose(in.data(), aux.data(), TransposeType::X_TO_Y) )
      DTFFT_CXX_CALL( plan->transpose(aux.data(), out.data(), TransposeType::Y_TO_Z) )
    }
  } else {
    DTFFT_CXX_CALL( plan->execute(in.data(), out.data(), ExecuteType::FORWARD, aux.data()) )
  }

  tf += MPI_Wtime();

  std::fill(in.begin(), in.end(), complex<float>{-1., -1.});

  if ( executor != Executor::NONE ) {
    float scaler = 1. / (float) (nx * ny * nz);
    for ( auto & element: out) {
      element *= scaler;
    }
  }

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan->execute(out.data(), in.data(), ExecuteType::BACKWARD, aux.data()) )
  tb += MPI_Wtime();

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(complex<float>(in[i] - check[i]));
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, &nz, local_error, tf, tb);
  DTFFT_CXX_CALL( plan->destroy() )

  MPI_Finalize();
  return 0;
}