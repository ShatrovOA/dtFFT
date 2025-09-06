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
  int32_t nx = 3133, ny = 44;
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
  config.set_backend(Backend::MPI_P2P)
    .set_platform(Platform::CUDA); // Can be changed at runtime via `DTFFT_PLATFORM` environment variable
  // config.set_enable_nvshmem_backends(false);
  DTFFT_CXX_CALL( set_config(config) )
#endif

  // Create plan
  const vector<int32_t> dims = {ny, nx};
  PlanC2C plan(dims, MPI_COMM_WORLD, Precision::DOUBLE, Effort::PATIENT, Executor::NONE);

  DTFFT_CXX_CALL( plan.report() )

  if ( plan.get_precision() != Precision::DOUBLE ) {
    DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), "reported_precision != Precision::DOUBLE")
  }

  size_t alloc_size = plan.get_alloc_size();

  std::vector<dtfft::Pencil> pencils;

  for (int i = 0; i < 2; i++) {
    dtfft::Pencil pencil = plan.get_pencil(i + 1);
    pencils.push_back(pencil);
  }

  size_t in_size = pencils[0].get_size();
  size_t out_size = pencils[1].get_size();

  complex<double> * in;
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * sizeof(*in), reinterpret_cast<void**>(&in)) )

  auto out = plan.mem_alloc<complex<double>>(alloc_size);
  auto check = new complex<double>[in_size];

  setTestValuesComplexDouble(check, in_size);

#if defined(DTFFT_WITH_CUDA)
  Platform platform = plan.get_platform();
  complexDoubleH2D(check, in, in_size, static_cast<int32_t>(platform));
#else
  complexDoubleH2D(check, in, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.transpose(in, out, Transpose::X_TO_Y) );
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
  DTFFT_CXX_CALL( plan.transpose(out, in, Transpose::Y_TO_X) );
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportComplexDouble(nx * ny, tf, tb, in, in_size, check, static_cast<int32_t>(platform));
#else
  checkAndReportComplexDouble(nx * ny, tf, tb, in, in_size, check);
#endif

  DTFFT_CXX_CALL( plan.mem_free(in) )
  DTFFT_CXX_CALL( plan.mem_free(out) )
  delete[] check;

  dtfft::Error error_code;
  error_code = plan.destroy();
  if ( comm_rank == 0 ) std::cout << dtfft::get_error_string(error_code) << std::endl;
  // Should not catch any signal
  // Simply returning `DTFFT_ERROR_PLAN_NOT_CREATED`
  error_code = plan.execute(in, out, static_cast<dtfft::Execute>(-1));
  if ( comm_rank == 0 ) std::cout << dtfft::get_error_string(error_code) << std::endl;
  MPI_Finalize();
  return 0;
}