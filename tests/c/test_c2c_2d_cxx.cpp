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
#include <cmath>
#include <iostream>
#include <complex>
#include "test_utils.h"
#include <cstring>

using namespace std;
using namespace dtfft;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  int32_t nx = 31334, ny = 44;
#else
  int32_t nx = 313, ny = 44;
#endif

  if(comm_rank == 0) {
    cout << "----------------------------------------" << endl;
    cout << "|   DTFFT test C++ interface: c2c_2d   |" << endl;
    cout << "----------------------------------------" << endl;
    cout << "Nx = " << nx << ", Ny = " << ny           << endl;
    cout << "Number of processors: " << comm_size      << endl;
    cout << "----------------------------------------" << endl;
    cout << "dtFFT Version = " << Version::get()       << endl;
  }

  attach_gpu_to_process();

#if defined(DTFFT_WITH_CUDA)
  Config config;
  config.set_backend(Backend::MPI_P2P);
  config.set_platform(Platform::CUDA); // Can be changed at runtime via `DTFFT_PLATFORM` environment variable
  // config.set_enable_nvshmem_backends(false);
  set_config(config);
#endif

  // Create plan
  const vector<int32_t> dims = {ny, nx};
  PlanC2C plan(dims, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, Executor::NONE);

  DTFFT_CXX_CALL( plan.report() )

  size_t alloc_size;
  DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) )

  std::vector<dtfft::Pencil> pencils;

  for (int i = 0; i < 2; i++) {
    dtfft::Pencil pencil;
    DTFFT_CXX_CALL( plan.get_pencil(i + 1, pencil) )
    pencils.push_back(pencil);
  }

  size_t in_size = pencils[0].get_size();
  size_t out_size = pencils[1].get_size();

  complex<double> *in, *out;
  auto *check = new complex<double>[in_size];

  for (size_t j = 0; j < in_size; j++) {
    double real = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    double cmplx = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    check[j] = std::complex<double>{real, cmplx};
  }

  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&in) )
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&out) )

#if defined(DTFFT_WITH_CUDA)
  Platform platform;
  DTFFT_CXX_CALL( plan.get_platform(&platform) )

  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaMemcpy(in, check, alloc_size * sizeof(complex<double>), cudaMemcpyHostToDevice) );
  } else {
    std::memcpy(in, check, alloc_size * sizeof(complex<double>));
  }
#else
  std::memcpy(in, check, alloc_size * sizeof(complex<double>));
#endif

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.transpose(in, out, dtfft::Transpose::X_TO_Y) );
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::HOST ) {
    for ( size_t i = 0; i < out_size; i++) {
      in[i] = complex<double>{-1., -1.};
    }
  }
#else
  for ( size_t i = 0; i < out_size; i++) {
    in[i] = complex<double>{-1., -1.};
  }
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.transpose(out, in, dtfft::Transpose::Y_TO_X) );
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  complex<double> *h_in = new complex<double>[in_size];

  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaMemcpy(h_in, in, in_size * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
  } else {
    std::memcpy(h_in, in, in_size * sizeof(complex<double>));
  }
  double local_error = checkComplexDouble(check, h_in, in_size);

  delete[] h_in;
#else
  double local_error = checkComplexDouble(check, in, in_size);
#endif

  DTFFT_CXX_CALL( plan.mem_free(in) )
  DTFFT_CXX_CALL( plan.mem_free(out) )
  delete[] check;

  reportDouble(&tf, &tb, &local_error, &nx, &ny, nullptr);

  dtfft::Error error_code;
  error_code = plan.destroy();
  if ( comm_rank == 0 ) std::cout << dtfft::get_error_string(error_code) << std::endl;
  // Should not catch any signal
  // Simply returning `DTFFT_ERROR_PLAN_NOT_CREATED`
  error_code = plan.execute(nullptr, nullptr, static_cast<dtfft::Execute>(-1));
  if ( comm_rank == 0 ) std::cout << dtfft::get_error_string(error_code) << std::endl;
  MPI_Finalize();
  return 0;
}