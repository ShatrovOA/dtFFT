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

#if defined(DTFFT_WITH_CUDA) && !defined(DTFFT_RUNNING_CICD)
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

  attach_gpu_to_process();
  Executor executor = Executor::NONE;
  Config conf;
  // Different FFT kinds are used. Disabling Z-slab
  conf.set_enable_z_slab(false);

#ifdef DTFFT_WITH_FFTW
  executor = Executor::FFTW3;
#endif
#ifdef DTFFT_WITH_CUDA
  char* platform_env = std::getenv("DTFFT_PLATFORM");

  bool running_cuda = platform_env == nullptr || std::strcmp(platform_env, "cuda") == 0;
  cudaStream_t stream;
  if ( running_cuda )
  {
# if defined(DTFFT_WITH_VKFFT)
    executor = Executor::VKFFT;
# else
    executor = Executor::NONE;
# endif
    CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
    conf.set_stream((dtfft_stream_t)stream)
      .set_enable_mpi_backends(true)
      .set_enable_nvshmem_backends(false);
  }
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
  size_t element_size;
  DTFFT_CXX_CALL( plan.get_element_size(&element_size) );

  if ( element_size != sizeof(float) ) {
    DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), "element_size != sizeof(float)")
  }

  float *inout, *aux;
  float *check = new float[in_size];
  setTestValuesFloat(check, in_size);

  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * element_size, (void**)&inout) )
  DTFFT_CXX_CALL( plan.mem_alloc(alloc_size * element_size, (void**)&aux) )

#if defined(DTFFT_WITH_CUDA)
  Platform platform = plan.get_platform();

  if ( running_cuda && platform != Platform::CUDA ) {
    DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), "running_cuda && platform != Platform::CUDA")
  }

  floatH2D(check, inout, in_size, static_cast<int32_t>(platform));
#else
  floatH2D(check, inout, in_size);
#endif

  double tf = 0.0 - MPI_Wtime();

  DTFFT_CXX_CALL( plan.execute(inout, inout, Execute::FORWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( running_cuda ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
  }
#endif
  tf += MPI_Wtime();

  size_t scale_value = 8 * nx * ny * nz;

#if defined(DTFFT_WITH_CUDA)
  scaleFloat(static_cast<int32_t>(executor), inout, out_size, scale_value, static_cast<int32_t>(platform), stream);
#else
  scaleFloat(static_cast<int32_t>(executor), inout, out_size, scale_value);
#endif

  double tb = 0.0 - MPI_Wtime();
  DTFFT_CXX_CALL( plan.execute(inout, inout, Execute::BACKWARD, aux) )
#if defined(DTFFT_WITH_CUDA)
  if ( platform == Platform::CUDA ) {
    CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
  }
#endif
  tb += MPI_Wtime();


#if defined(DTFFT_WITH_CUDA)
  checkAndReportFloat(nx * ny * nz, tf, tb, inout, in_size, check, static_cast<int32_t>(platform));
#else
  checkAndReportFloat(nx * ny * nz, tf, tb, inout, in_size, check);
#endif


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