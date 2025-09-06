#pragma once

#include "heffte.h"
#include "config.hpp"
#include "dtfft.hpp"

// There are multiple ways to select the GPU tag

// 1. Using the default_backend trait with the tag::gpu for the location
using backend_tag = heffte::backend::cufft;

template<typename T>
void run_heffte_private(heffte::box3d<>& boxin, heffte::box3d<>&boxout){

    MPI_Comm comm = MPI_COMM_WORLD;

    int comm_rank, comm_size;
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Comm_size(comm, &comm_size);

    // define the heffte class and the input and output geometry
    // heffte::plan_options can be specified just as in the backend::fftw
    heffte::fft3d<backend_tag> fft(boxin, boxout, comm);

    size_t alloc_size = std::max(fft.size_inbox(), fft.size_outbox());

    heffte::gpu::vector<std::complex<T>> gpu_input(alloc_size);
    CUDA_CALL( cudaMemset(gpu_input.data(), 0, alloc_size * sizeof(std::complex<T>)) );

    // allocate memory on the device for the output
    heffte::gpu::vector<std::complex<T>> gpu_output(alloc_size);

    // allocate scratch space, this is using the public type alias buffer_container
    // and for the cufft backend this is heffte::gpu::vector
    // for the CPU backends (fftw and mkl) the buffer_container is std::vector
    heffte::fft3d<backend_tag>::buffer_container<std::complex<T>> workspace(fft.size_workspace());

      // Timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CALL(cudaEventCreate(&startEvent));
    CUDA_CALL(cudaEventCreate(&stopEvent));
    float ms;

        // Warmup
    if (comm_rank == 0) {
        printf("Started warmup\n");
    }
    for (int iter = 0; iter < WARMUP_ITERATIONS; iter++) {
        // perform forward fft using arrays and the user-created workspace
        fft.forward(gpu_input.data(), gpu_output.data(), workspace.data(), heffte::scale::none);
        fft.backward(gpu_output.data(), gpu_input.data(), workspace.data(), heffte::scale::none);
    }

    MPI_Barrier(comm);
    if (comm_rank == 0) {
        printf("Ended warmup\n");
    }

    CUDA_CALL(cudaEventRecord(startEvent, fft.stream()));

    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        // perform forward fft using arrays and the user-created workspace
        fft.forward(gpu_input.data(), gpu_output.data(), workspace.data(), heffte::scale::none);
        fft.backward(gpu_output.data(), gpu_input.data(), workspace.data(), heffte::scale::none);
    }

    CUDA_CALL(cudaEventRecord(stopEvent, fft.stream()));
    CUDA_CALL(cudaEventSynchronize(stopEvent));
    CUDA_CALL(cudaEventElapsedTime(&ms, startEvent, stopEvent));


    // Gather timing statistics
    float min_ms, max_ms, avg_ms;
    MPI_Allreduce(&ms, &min_ms, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&ms, &max_ms, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&ms, &avg_ms, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    avg_ms /= (float)comm_size;

    if (comm_rank == 0) {
        printf("min time: %f [ms]\n", min_ms);
        printf("max time: %f [ms]\n", max_ms);
        printf("avg time: %f [ms]\n", avg_ms);
        printf("----------------------------------------\n");
    }

    // Cleanup
    CUDA_CALL(cudaEventDestroy(startEvent));
    CUDA_CALL(cudaEventDestroy(stopEvent));
}

template<typename T>
void run_heffte(const std::vector<int>& dims, bool enable_z_slab) {
    dtfft::Config conf;
    conf.set_enable_z_slab(enable_z_slab)
        .set_enable_log(false)
        .set_platform(dtfft::Platform::HOST);
    DTFFT_CXX_CALL( dtfft::set_config(conf) );

    // Defining boxes as they are retuned by dtFFT
    dtfft::PlanC2C plan(dims, sizeof(T) == 4 ? dtfft::Precision::SINGLE : dtfft::Precision::DOUBLE, dtfft::Effort::ESTIMATE);
    if ( enable_z_slab ) {
        if ( !plan.get_z_slab_enabled() ) return;
    }
    std::array<int, 3> ilower, iupper, icount, olower, oupper, ocount;
    plan.get_local_sizes(ilower.data(), icount.data(), olower.data(), ocount.data());
    for (std::size_t i = 0; i < 3; ++i) {
        iupper[i] = ilower[i] + icount[i] - 1;
        oupper[i] = olower[i] + ocount[i] - 1;
    }
    std::array<int, 3> out_order = {2, 0, 1};

    heffte::box3d<> boxin(ilower, iupper);
    heffte::box3d<> boxout(olower, oupper);

    int comm_rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if (comm_rank == 0) {
        printf("----------------------------------------\n");
        printf("HeFFTe GPU benchmark\n");
        printf("Precision is %s\n", dtfft::get_precision_string(plan.get_precision()).c_str());
        if ( enable_z_slab ) {
            printf("Z-slab is enabled\n");
        }
        printf("----------------------------------------\n");
    }

  run_heffte_private<T>(boxin, boxin);
}