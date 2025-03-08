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

using namespace std;
using namespace dtfft;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
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

  assign_device_to_process();

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  Config config;
  config.set_gpu_backend(GPUBackend::MPI_P2P);
  config.set_platform(Platform::CUDA);
  // config.set_enable_nvshmem_backends(false);
  set_config(config);
#endif

  // Create plan
  const vector<int32_t> dims = {ny, nx};
  PlanC2C plan(dims, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, Executor::NONE);

  plan.report();

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

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  complex<double> *d_in, *d_out;

  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&d_in) )
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&d_out) )

  in = new complex<double>[alloc_size];
  out = new complex<double>[alloc_size];
#else
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&in) )
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(complex<double>), (void**)&out) )
#endif

  for (size_t j = 0; j < in_size; j++) {
    double real = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    double cmplx = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    in[j] = std::complex<double>{real, cmplx};
    check[j] = in[j];
  }

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  CUDA_SAFE_CALL( cudaMemcpy(d_in, in, alloc_size * sizeof(complex<double>), cudaMemcpyHostToDevice) );

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.transpose(d_in, d_out, dtfft::TransposeType::X_TO_Y) );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#else
  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.transpose(in, out, dtfft::TransposeType::X_TO_Y) );
#endif
  tf += MPI_Wtime();

  for ( size_t i = 0; i < out_size; i++) {
    in[i] = complex<double>{-1., -1.};
  }

  double tb = 0.0 - MPI_Wtime();
#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  DTFFT_CXX_CALL( plan.transpose(d_out, d_in, dtfft::TransposeType::Y_TO_X) );
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  CUDA_SAFE_CALL( cudaMemcpy(in, d_in, alloc_size * sizeof(complex<double>), cudaMemcpyDeviceToHost) );
#else
  DTFFT_CXX_CALL( plan.transpose(out, in, dtfft::TransposeType::Y_TO_X) );
#endif
  tb += MPI_Wtime();

  double local_error = -1.;
  for (size_t i = 0; i < in_size; i++) {
    double error = abs(complex<double>(in[i] - check[i]));
    local_error = error > local_error ? error : local_error;
  }

#if defined(DTFFT_WITH_CUDA) && defined(__NVCOMPILER)
  DTFFT_CXX_CALL( plan.mem_free(d_in) )
  DTFFT_CXX_CALL( plan.mem_free(d_out) )

  delete[] in;
  delete[] out;
#else
  DTFFT_CXX_CALL( plan.mem_free(in) )
  DTFFT_CXX_CALL( plan.mem_free(out) )
#endif
  delete[] check;

  report_double(&nx, &ny, nullptr, local_error, tf, tb);

  dtfft::ErrorCode error_code;
  error_code = plan.destroy();
  std::cout << dtfft::get_error_string(error_code) << std::endl;
  // Should not catch any signal
  // Simply returning `DTFFT_ERROR_PLAN_NOT_CREATED`
  error_code = plan.execute(nullptr, nullptr, static_cast<dtfft::ExecuteType>(-1));
  std::cout << dtfft::get_error_string(error_code) << std::endl;
  MPI_Finalize();
  return 0;
}