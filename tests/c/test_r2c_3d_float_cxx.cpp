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
#include <vector>
#include <cstdlib>
#include <cstring>

#include "test_utils.h"

using namespace std;
using namespace dtfft;

int main(int argc, char *argv[])
{
#ifdef DTFFT_TRANSPOSE_ONLY
  return 0;
#else
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
  int32_t nx = 111, ny = 44, nz = 352;
#else
  int32_t nx = 19, ny = 32, nz = 55;
#endif

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: r2c_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
  }

  Executor executor = Executor::NONE;
#ifdef DTFFT_WITH_MKL
  executor = Executor::MKL;
#elif defined (DTFFT_WITH_FFTW)
  executor = Executor::FFTW3;
#endif

#ifdef DTFFT_WITH_CUDA
  char* platform_env = std::getenv("DTFFT_PLATFORM");
  bool running_cuda = false;

  if ( platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0 )
  {
    running_cuda = true;
# if defined(DTFFT_WITH_CUFFT)
    executor = Executor::CUFFT;
# elif defined(DTFFT_WITH_VKFFT)
    executor = Executor::VKFFT;
# else
    executor = Executor::NONE;
# endif
  }
#endif

  if ( executor == Executor::NONE ) {
    if ( comm_rank == 0 ) cout << "Could not find valid R2C FFT executor, skipping test\n";
    MPI_Finalize();
    return 0;
  }

  Config conf;
  auto backend = Backend::MPI_P2P;
#ifdef DTFFT_WITH_CUDA
  if ( running_cuda ) {
# ifdef DTFFT_WITH_NVSHMEM
    backend = Backend::CUFFTMP_PIPELINED;
# elif defined(DTFFT_WITH_NCCL)
    backend = Backend::NCCL_PIPELINED;
# endif
  }
#endif
  conf.set_backend(backend);

  DTFFT_CXX_CALL( set_config(conf) )

  attach_gpu_to_process();

  // Create plan
  vector<int32_t> dims = {nz, ny, nx};
  auto plan = PlanR2C(dims, executor, MPI_COMM_WORLD, Precision::SINGLE, Effort::MEASURE);
  DTFFT_CXX_CALL( plan.report() )
  vector<int32_t> in_counts(3), out_counts(3);
  DTFFT_CXX_CALL( plan.get_local_sizes(nullptr, in_counts.data()) )
  size_t alloc_size = plan.get_alloc_size();
  size_t element_size = plan.get_element_size();

  Pencil out_pencil = plan.get_pencil(3);
  size_t out_size = out_pencil.get_size();

  if ( element_size != sizeof(float) ) {
    DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), "element_size != sizeof(float)")
  }

  if ( comm_rank == 0 ) {
    cout << "Using executor: " << get_executor_string(plan.get_executor())
         << ", precision: " << get_precision_string(plan.get_precision()) << endl;
  }

  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];

  float *check = new float[in_size];
  setTestValuesFloat(check, in_size);

  float *buf = nullptr;
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * element_size, (void**)&buf) )
  auto aux = plan.mem_alloc<float>(alloc_size);

#if defined(DTFFT_WITH_CUDA)
  auto platform = plan.get_platform();
  auto real_backend = plan.get_backend();
  if ( (backend != real_backend) && (comm_size > 1) && (platform == Platform::CUDA) ) {
    DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), 
                          "Backend mismatch: backend set before plan creation: " + get_backend_string(backend) +
                          ", but plan reports: " + get_backend_string(real_backend));
  }
  floatH2D(check, buf, in_size, static_cast<int32_t>(platform));
#else
  floatH2D(check, buf, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  // Performing inplace transform, but treating input and output as different types
  auto fourier = plan.execute<std::complex<float>>(buf, Execute::FORWARD, aux);
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  scaleComplexFloat(static_cast<int32_t>(executor), fourier, out_size, nx * ny * nz, static_cast<int32_t>(platform), NULL);
#else
  scaleComplexFloat(static_cast<int32_t>(executor), fourier, out_size, nx * ny * nz);
#endif

  double tb = 0.0 - MPI_Wtime();
  auto real = plan.backward<float>(fourier, aux);
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  checkAndReportFloat(nx * ny * nz, tf, tb, real, in_size, check, static_cast<int32_t>(platform));
#else
  checkAndReportFloat(nx * ny * nz, tf, tb, real, in_size, check);
#endif

  delete[] check;
  DTFFT_CXX_CALL( plan.mem_free(buf) )
  DTFFT_CXX_CALL( plan.mem_free(aux) )
  fourier = nullptr;
  real = nullptr;

  // Plan must be destroyed before calling MPI_Finalize
  DTFFT_CXX_CALL( plan.destroy() )

  MPI_Finalize();
  return 0;
#endif
}