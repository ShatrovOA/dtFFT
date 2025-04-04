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

  Executor executor;
#ifdef DTFFT_WITH_MKL
  executor = Executor::MKL;
#elif defined (DTFFT_WITH_FFTW)
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
#ifdef DTFFT_WITH_CUDA
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  if ( platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0 )
  {
# if defined(DTFFT_WITH_CUFFT)
    executor = Executor::CUFFT;
# elif defined(DTFFT_WITH_VKFFT)
    executor = Executor::VKFFT;
# else
    if(comm_rank == 0) {
      cout << "Missing CUDA FFT Executor\n";
    }
    MPI_Finalize();
    return 0;
# endif
  }
#endif

#ifdef DTFFT_WITH_CUDA
  dtfft::Config conf;
#ifdef DTFFT_WITH_NVSHMEM
  conf.set_backend(dtfft::Backend::CUFFTMP);
#else
  conf.set_backend(dtfft::Backend::MPI_P2P);
#endif
  conf.set_platform(dtfft::Platform::CUDA);
  DTFFT_CXX_CALL( dtfft::set_config(conf) )
#endif

  attach_gpu_to_process();

  // Create plan
  vector<int32_t> dims = {nz, ny, nx};
  dtfft::PlanR2C plan(dims, executor, MPI_COMM_WORLD, Precision::SINGLE, Effort::MEASURE);

  vector<int32_t> in_counts(3);
  DTFFT_CXX_CALL( plan.get_local_sizes(nullptr, in_counts.data()) )
  size_t alloc_size;
  DTFFT_CXX_CALL( plan.get_alloc_size(&alloc_size) )
  size_t element_size;
  DTFFT_CXX_CALL( plan.get_element_size(&element_size) )

  dtfft::Pencil out_pencil;
  DTFFT_CXX_CALL( plan.get_pencil(3, out_pencil) )
  size_t out_size = out_pencil.get_size();

  if ( element_size != sizeof(float) ) {
    DTFFT_THROW_EXCEPTION("element_size != sizeof(float)")
  }

  size_t in_size = in_counts[0] * in_counts[1] * in_counts[2];

  float *buf, *check, *aux;

  check = new float[in_size];
  for (size_t i = 0; i < in_size; i++) {
    check[i] = (float)(i) / (float)(nx) / (float)(ny) / (float)(nz);
  }

  DTFFT_CXX_CALL( plan.mem_alloc(element_size * alloc_size, (void**)&buf) )
  DTFFT_CXX_CALL( plan.mem_alloc(element_size * alloc_size, (void**)&aux) )

#if defined(DTFFT_WITH_CUDA)
  dtfft::Platform platform;
  DTFFT_CXX_CALL( plan.get_platform(&platform) )

  if ( platform == dtfft::Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaMemcpy(buf, check, element_size * in_size, cudaMemcpyHostToDevice) )
  } else {
    std::memcpy(buf, check, element_size * in_size);
  }
#else
  std::memcpy(buf, check, element_size * in_size);
#endif

  double tf = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(buf, buf, Execute::FORWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == dtfft::Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
  if ( platform == dtfft::Platform::CUDA ) {
    scaleComplexFloat(buf, out_size, nx * ny * nz, 0);
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  } else {
    scaleComplexFloatHost(buf, out_size, nx * ny * nz);
  }
#else
  scaleComplexFloatHost(buf, out_size, nx * ny * nz);
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(buf, buf, Execute::BACKWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == dtfft::Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaDeviceSynchronize() )
  }
#endif
  tb += MPI_Wtime();

  float local_error;
#if defined(DTFFT_WITH_CUDA)
  if ( platform == dtfft::Platform::CUDA ) {
    float *test = new float[in_size];

    CUDA_SAFE_CALL( cudaMemcpy(test, buf, element_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkFloat(check, test, in_size);
    delete[] test;
  } else {
    local_error = checkFloat(check, buf, in_size);
  }
#else
  local_error = checkFloat(check, buf, in_size);
#endif

  reportSingle(&tf, &tb, &local_error, &nx, &ny, &nz);
  // Plan must be destroyed before calling MPI_Finalize
  DTFFT_CXX_CALL( plan.destroy() )

  MPI_Finalize();
  return 0;
#endif
}