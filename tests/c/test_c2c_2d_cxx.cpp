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

#include "dtfft.hpp"
#include "test_utils.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace dtfft;

int main(int argc, char* argv[])
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

    if (comm_rank == 0) {
        cout << "----------------------------------------" << endl;
        cout << "|   dtFFT test C++ interface: c2c_2d   |" << endl;
        cout << "----------------------------------------" << endl;
        cout << "Nx = " << nx << ", Ny = " << ny << endl;
        cout << "Number of processors: " << comm_size << endl;
        cout << "----------------------------------------" << endl;
        cout << "dtFFT Version = " << Version::get() << endl;
    }

    attach_gpu_to_process();

    auto config = Config()
        .set_enable_mpi_backends(true)
        .set_enable_fourier_reshape(true)
        .set_transpose_mode(dtfft::TransposeMode::UNPACK)
        .set_access_mode(dtfft::AccessMode::READ);

#if defined(DTFFT_WITH_CUDA)
    config.set_platform(Platform::CUDA) // Can be changed at runtime via `DTFFT_PLATFORM` environment variable
        .set_enable_nvshmem_backends(false);
#endif
    DTFFT_CXX_CALL(set_config(config))

    // Create 2D MPI grid decomposition
    int grid_dims[2] = { 0, 0 };
    int glob_dims[2] = {ny, nx};
    vector<int32_t> starts(2), counts(2);

    createGridDims(2, glob_dims, grid_dims, starts.data(), counts.data());

    Pencil pencil(starts, counts);

    if (comm_rank == 0) {
        cout << "Grid decomposition: " << grid_dims[0] << " x " << grid_dims[1] << endl;
    }

    bool reshape_required = grid_dims[0] > 1;

    // Create plan using pencil
    PlanC2C plan(pencil, Precision::DOUBLE, Effort::EXHAUSTIVE);

    DTFFT_CXX_CALL(plan.report())

    if (plan.get_precision() != Precision::DOUBLE) {
        DTFFT_THROW_EXCEPTION(static_cast<Error>(-1), "reported_precision != Precision::DOUBLE")
    }

    auto back = plan.get_backend();
    if (comm_rank == 0)
        std::cout << "Using backend: " << get_backend_string(back) << "\n";

    if ( reshape_required ) {
        auto reshape_back = plan.get_reshape_backend();
        if (comm_rank == 0)
            std::cout << "Using reshape backend: " << get_backend_string(reshape_back) << "\n";
    }

    size_t alloc_size = plan.get_alloc_size();
    size_t alloc_bytes = plan.get_alloc_bytes();

    std::vector<dtfft::Pencil> pencils;

    if ( reshape_required )
        pencils.push_back(plan.get_pencil(Layout::X_BRICKS));

    pencils.push_back(plan.get_pencil(Layout::X_PENCILS));
    pencils.push_back(plan.get_pencil(Layout::Y_PENCILS));
    if ( reshape_required )
        pencils.push_back(plan.get_pencil(Layout::Z_BRICKS));

    size_t in_size = pencils[0].get_size();
    size_t out_size = pencils[pencils.size() - 1].get_size();

    complex<double>* in;
    DTFFT_CXX_CALL(plan.mem_alloc(alloc_bytes, reinterpret_cast<void**>(&in)))

    auto work = plan.mem_alloc<complex<double>>(plan.get_aux_size());
    auto out = plan.mem_alloc<complex<double>>(alloc_size);
    auto check = new complex<double>[in_size];
    size_t reshape_work_size;
    auto ierr = plan.get_aux_size_reshape(&reshape_work_size);
    if ( ierr == dtfft::Error::SUCCESS && reshape_work_size > 0 ) {
        if ( comm_rank == 0 ) cout << "reshape_work_size = " << reshape_work_size << "\n";
        auto reshape_bytes = plan.get_aux_bytes_reshape();
        if ( reshape_bytes != reshape_work_size * sizeof(complex<double>) ) {
            DTFFT_THROW_EXCEPTION(static_cast<Error>(-2), "reshape_bytes != reshape_work_size * sizeof(complex<double>)");
        }
    }

    auto transpose_work_size = plan.get_aux_size_transpose();
    if ( comm_rank == 0 ) cout << "transpose_work_size = " << transpose_work_size << "\n";
    auto transpose_bytes = plan.get_aux_bytes_transpose();
    if ( transpose_bytes != transpose_work_size * sizeof(complex<double>) ) {
        DTFFT_THROW_EXCEPTION(static_cast<Error>(-2), "transpose_bytes != transpose_work_size * sizeof(complex<double>)");
    }
    auto work_size = std::max(reshape_work_size, transpose_work_size);

    std::complex<double>* aux = nullptr;
    if (work_size > 0) aux = plan.mem_alloc<complex<double>>(work_size);


    setTestValuesComplexDouble(check, in_size);

#if defined(DTFFT_WITH_CUDA)
    Platform platform = plan.get_platform();
    complexDoubleH2D(check, in, in_size, static_cast<int32_t>(platform));
#else
    complexDoubleH2D(check, in, in_size);
#endif

    double tf = 0.0 - MPI_Wtime();

    if ( reshape_required ) {
        auto request_reshape_xb = plan.reshape_start(in, work, Reshape::X_BRICKS_TO_PENCILS, aux);
        if (comm_rank == 0)
            cout << "Doing stuff while data is being reshaped on host" << endl;
        DTFFT_CXX_CALL(plan.reshape_end(request_reshape_xb));

        auto request_transpose_xy = plan.transpose_start(work, in, Transpose::X_TO_Y, aux);
        if (comm_rank == 0)
            cout << "Doing stuff while data is being transposed on host" << endl;
        DTFFT_CXX_CALL(plan.transpose_end(request_transpose_xy));

        DTFFT_CXX_CALL(plan.reshape(in, out, Reshape::Y_PENCILS_TO_BRICKS, aux));
        if (comm_rank == 0)
            cout << "Converted pencils to bricks" << endl;
    } else {
        DTFFT_CXX_CALL(plan.transpose(in, out, Transpose::X_TO_Y, aux) )
    }
#if defined(DTFFT_WITH_CUDA)
    if (platform == Platform::CUDA) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize())
    }
#endif
    tf += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
    if (platform == Platform::HOST) {
        for (size_t i = 0; i < out_size; i++) {
            in[i] = complex<double> { -1., -1. };
        }
    }
#else
    for (size_t i = 0; i < out_size; i++) {
        in[i] = complex<double> { -1., -1. };
    }
#endif

    double tb = 0.0 - MPI_Wtime();
    DTFFT_CXX_CALL(plan.backward(out, in, work));
    if (comm_rank == 0)
        cout << "Executed backwards using blocking execute" << endl;

#if defined(DTFFT_WITH_CUDA)
    if (platform == Platform::CUDA) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize())
    }
#endif
    tb += MPI_Wtime();

#if defined(DTFFT_WITH_CUDA)
    checkAndReportComplexDouble(nx * ny, tf, tb, in, in_size, check, static_cast<int32_t>(platform));
#else
    checkAndReportComplexDouble(nx * ny, tf, tb, in, in_size, check);
#endif

    DTFFT_CXX_CALL(plan.mem_free(in))
    DTFFT_CXX_CALL(plan.mem_free(out))
    DTFFT_CXX_CALL(plan.mem_free(work))
    delete[] check;

    dtfft::Error error_code;
    error_code = plan.destroy();
    if (comm_rank == 0)
        std::cout << dtfft::get_error_string(error_code) << std::endl;
    // Should not catch any signal
    // Simply returning `DTFFT_ERROR_PLAN_NOT_CREATED`
    error_code = plan.execute(in, out, static_cast<dtfft::Execute>(-1));
    if (comm_rank == 0)
        std::cout << dtfft::get_error_string(error_code) << std::endl;
    MPI_Finalize();
    return 0;
}