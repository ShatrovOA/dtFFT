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
#include "test_utils.h"

using namespace std;

int main(int argc, char *argv[])
{
  // MPI_Init must be called before calling dtFFT
  MPI_Init(&argc, &argv);
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

#ifdef DTFFT_WITH_CUDA
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
  }

  assign_device_to_process();

  // Create plan
  const vector<int32_t> dims = {ny, nx};
  dtfft::PlanC2C plan = dtfft::PlanC2C(dims, MPI_COMM_WORLD, DTFFT_DOUBLE, DTFFT_PATIENT, DTFFT_EXECUTOR_NONE);

  size_t alloc_size;
  plan.get_alloc_size(&alloc_size);

  dtfft_pencil_t pencils[2];
  for (int i = 0; i < 2; i++) {
    plan.get_pencil(i + 1, &pencils[i]);
  }

  size_t in_size = pencils[0].counts[0] * pencils[0].counts[1];
  size_t out_size = pencils[1].counts[0] * pencils[1].counts[1];

  complex<double> *in = new complex<double>[alloc_size];
  complex<double> *out = new complex<double>[alloc_size];
  complex<double> *check = new complex<double>[in_size];

  for (size_t j = 0; j < in_size; j++) {
    double real = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    double cmplx = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    in[j] = std::complex<double>{real, cmplx};
    check[j] = in[j];
  }

#pragma acc enter data copyin(in[0:alloc_size - 1]) create(out[0:alloc_size - 1])

  double tf = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(in, out)
  DTFFT_CALL( plan.transpose(in, out, DTFFT_TRANSPOSE_X_TO_Y) )

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tf += MPI_Wtime();

#pragma acc parallel loop present(in)
  for ( size_t i = 0; i < out_size; i++) {
    in[i] = complex<double>{-1., -1.};
  }

  double tb = 0.0 - MPI_Wtime();
#pragma acc host_data use_device(in, out)
  DTFFT_CALL( plan.transpose(out, in, DTFFT_TRANSPOSE_Y_TO_X) )

#ifdef DTFFT_WITH_CUDA
  CUDA_SAFE_CALL( cudaDeviceSynchronize() )
#endif
  tb += MPI_Wtime();

#pragma acc update self(in[0:in_size-1])

  double local_error = -1.;
  for (size_t i = 0; i < in_size; i++) {
    double error = abs(complex<double>(in[i] - check[i]));
    local_error = error > local_error ? error : local_error;
  }

#pragma acc exit data delete(in, out)

  delete[] in;
  delete[] out;
  delete[] check;

  report_double(&nx, &ny, nullptr, local_error, tf, tb);

  dtfft_error_code_t error_code;
  error_code = plan.destroy();
  std::cout << dtfft_get_error_string(error_code) << std::endl;
  // Should not catch any signal
  // Simply returning `DTFFT_ERROR_PLAN_NOT_CREATED`
  error_code = plan.execute(NULL, NULL, (dtfft_execute_type_t)-1);
  std::cout << dtfft_get_error_string(error_code) << std::endl;
  MPI_Finalize();
  return 0;
}