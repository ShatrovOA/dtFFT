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
#include <stdio.h>
#include <complex>
#include <vector>
#include <numeric>
#include <cstring>

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
  const int32_t nx = 256, ny = 512, nz = 1024;
#else
  const int32_t nx = 32, ny = 64, nz = 128;
#endif

  if(comm_rank == 0) {
    cout << "----------------------------------------"          << endl;
    cout << "|DTFFT test C++ interface: r2r_3d_float|"          << endl;
    cout << "----------------------------------------"          << endl;
    cout << "Nx = " << nx << ", Ny = " << ny << ", Nz = " << nz << endl;
    cout << "Number of processors: " << comm_size               << endl;
    cout << "----------------------------------------"          << endl;
  }


  Executor executor = Executor::NONE;
#ifdef DTFFT_WITH_FFTW
  executor = Executor::FFTW3;
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  bool running_cuda = platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0;

  if ( running_cuda )
  {
# if defined(DTFFT_WITH_VKFFT)
    executor = Executor::VKFFT;
# else
    executor = Executor::NONE;
# endif
  }
#endif

  attach_gpu_to_process();

  Config conf;
  conf.set_enable_z_slab(false);

#if defined(DTFFT_WITH_CUDA)
  cudaStream_t stream;
  if ( running_cuda )
  {
    CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
    conf.set_stream((dtfft_stream_t)stream);
  }
  conf.set_enable_mpi_backends(true);
  conf.set_enable_nvshmem_backends(false);
  conf.set_platform(Platform::CUDA);
#endif

  DTFFT_CXX_CALL( set_config(conf) );

  const int8_t ndims = 3;
  const int32_t dims[] = {nz, ny, nx};
  const R2RKind kinds[] = {R2RKind::DCT_2, R2RKind::DCT_3, R2RKind::DCT_2};
  PlanR2R plan(ndims, dims, kinds, MPI_COMM_WORLD, Precision::SINGLE, Effort::PATIENT, executor);

  int32_t in_sizes[ndims];
  int32_t out_sizes[ndims];
  size_t alloc_size;

  DTFFT_CXX_CALL( plan.report() )
  DTFFT_CXX_CALL( plan.get_local_sizes(nullptr, in_sizes, nullptr, out_sizes, &alloc_size) )

  size_t in_size = std::accumulate(in_sizes, in_sizes + 3, 1, multiplies<int>());
  size_t out_size = std::accumulate(out_sizes, out_sizes + 3, 1, multiplies<int>());
  size_t el_size;
  DTFFT_CXX_CALL( plan.get_element_size(&el_size) );

  if ( el_size != sizeof(float) ) {
    DTFFT_THROW_EXCEPTION("el_size != sizeof(float)")
  }

  float *inout, *aux;
  float *check = new float[in_size];

  for (size_t i = 0; i < in_size; i++)
  {
    check[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * el_size, (void**)&inout) )
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * el_size, (void**)&aux) )

#if defined(DTFFT_WITH_CUDA)
  Platform platform;
  DTFFT_CXX_CALL( plan.get_platform(&platform) )

  if ( running_cuda && platform != Platform::CUDA ) {
    DTFFT_THROW_EXCEPTION("running_cuda && platform != Platform::CUDA")
  }

  if ( running_cuda ) {
    CUDA_SAFE_CALL( cudaMemcpyAsync(inout, check, in_size * el_size, cudaMemcpyHostToDevice, stream) );
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
  } else {
    std::memcpy(inout, check, in_size * el_size);
  }
#else
  std::memcpy(inout, check, in_size * el_size);
#endif

  double tf = 0.0 - MPI_Wtime();

  DTFFT_CXX_CALL( plan.execute(inout, inout, Execute::FORWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( running_cuda ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
  }
#endif
  tf += MPI_Wtime();

  if ( executor != Executor::NONE ) {
    size_t scale_value = 8 * nx * ny * nz;
#if defined(DTFFT_WITH_CUDA)
  if ( running_cuda ) {
    scaleFloat(inout, out_size, scale_value, stream);
    CUDA_SAFE_CALL(  cudaStreamSynchronize(stream) )
  } else {
    scaleFloatHost(inout, out_size, scale_value);
  }
#else
  scaleFloatHost(inout, out_size, scale_value);
#endif
  }

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(inout, inout, Execute::BACKWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
  }
#endif
  tb += MPI_Wtime();


  float local_error;
#if defined(DTFFT_WITH_CUDA)
  if ( platform == dtfft::Platform::CUDA ) {
    float *test = new float[in_size];

    CUDA_SAFE_CALL( cudaMemcpy(test, inout, el_size * in_size, cudaMemcpyDeviceToHost) )
    local_error = checkFloat(check, test, in_size);
    delete[] test;
  } else {
    local_error = checkFloat(check, inout, in_size);
  }
#else
  local_error = checkFloat(check, inout, in_size);
#endif

  reportSingle(&tf, &tb, &local_error, &nx, &ny, &nz);


  DTFFT_CXX_CALL( plan.mem_free(inout) )
  DTFFT_CXX_CALL( plan.mem_free(aux) )

#if defined(DTFFT_WITH_CUDA)
  if ( running_cuda ) {
    CUDA_SAFE_CALL( cudaStreamDestroy(stream) );
  }
#endif

  DTFFT_CXX_CALL( plan.destroy() )

  delete[] check;

  MPI_Finalize();
}