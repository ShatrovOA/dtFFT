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

#ifdef DTFFT_WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);

  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#ifdef DTFFT_WITH_CUDA
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

#ifdef DTFFT_WITH_FFTW
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_FFTW3;
#elif defined(DTFFT_WITH_VKFFT)
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_VKFFT;
  dtfft_disable_z_slab();
#else
  dtfft_executor_t executor_type = DTFFT_EXECUTOR_NONE;
#endif

#ifdef DTFFT_WITH_CUDA
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);
  CUDA_SAFE_CALL( cudaSetDevice(local_rank) );

  cudaStream_t stream;
  CUDA_SAFE_CALL( cudaStreamCreate(&stream) );
  dtfft_set_stream(stream);

  dtfft_enable_mpi_backends();
  // dtfft_set_gpu_backend(DTFFT_GPU_BACKEND_NCCL_PIPELINED);
  // dtfft_disable_pipelined_backends();
#endif


  const int8_t ndims = 3;
  const int32_t dims[] = {nz, ny, nx};
  const dtfft_r2r_kind_t kinds[] = {DTFFT_DCT_2, DTFFT_DCT_3, DTFFT_DCT_2};
  dtfft::PlanR2R plan(ndims, dims, kinds, MPI_COMM_WORLD, DTFFT_SINGLE, DTFFT_PATIENT, executor_type);

  int32_t in_sizes[ndims];
  int32_t out_sizes[ndims];
  size_t alloc_size;

  plan.get_local_sizes(NULL, in_sizes, NULL, out_sizes, &alloc_size);

  size_t in_size = std::accumulate(in_sizes, in_sizes + 3, 1, multiplies<int>());
  size_t out_size = std::accumulate(out_sizes, out_sizes + 3, 1, multiplies<int>());
  float *inout = new float[alloc_size];
  float *check = new float[alloc_size];
  float *aux = new float[alloc_size];

#ifdef DTFFT_WITH_CUDA
  float *d_inout, *d_aux;

  CUDA_SAFE_CALL( cudaMalloc((void**)&d_inout, alloc_size * sizeof(float)) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&d_aux, alloc_size * sizeof(float)) );
#endif

  for (size_t i = 0; i < in_size; i++)
  {
    inout[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    check[i] = inout[i];
  }

  double tf = 0.0 - MPI_Wtime();
#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaMemcpyAsync(d_inout, inout, alloc_size * sizeof(float), cudaMemcpyHostToDevice, stream) );
  plan.execute(d_inout, d_inout, DTFFT_TRANSPOSE_OUT, d_aux);
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
#else
  plan.execute(inout, inout, DTFFT_TRANSPOSE_OUT, aux);
#endif
  tf += MPI_Wtime();

  if ( executor_type != DTFFT_EXECUTOR_NONE ) {
#ifdef DTFFT_WITH_CUDA
#pragma acc parallel loop deviceptr(d_inout) vector_length(256) async
    for (size_t i = 0; i < out_size; i++)
    {
      d_inout[i] /= (float) (8 * nx * ny * nz);
    }
    // Clearing host buffer
    for (size_t i = 0; i < in_size; i++)
    {
      inout[i] = (float)(-1);
    }
#pragma acc wait
#else
    for (size_t i = 0; i < out_size; i++)
    {
      inout[i] /= (float) (8 * nx * ny * nz);
    }
#endif
  }

  double tb = 0.0 - MPI_Wtime();
#ifdef DTFFT_WITH_CUDA
  plan.execute(d_inout, d_inout, DTFFT_TRANSPOSE_IN, d_aux);
  CUDA_SAFE_CALL( cudaMemcpyAsync(inout, d_inout, alloc_size * sizeof(float), cudaMemcpyDeviceToHost, stream) );
  CUDA_SAFE_CALL( cudaStreamSynchronize(stream) );
#else
  plan.execute(inout, inout, DTFFT_TRANSPOSE_IN, aux);
#endif
  tb += MPI_Wtime();

  float local_error = -1.0;
  for (size_t i = 0; i < in_size; i++) {
    float error = abs(inout[i] - check[i]);
    local_error = error > local_error ? error : local_error;
  }

  report_float(&nx, &ny, &nz, local_error, tf, tb);

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaFree(d_inout) );
  CUDA_SAFE_CALL( cudaFree(d_aux) );

  CUDA_SAFE_CALL( cudaStreamDestroy(stream) );
#endif

  plan.destroy();

  delete[] inout;
  delete[] check;
  delete[] aux;

  MPI_Finalize();
}